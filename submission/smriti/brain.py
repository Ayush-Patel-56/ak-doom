"""SmritiNet: the Phase 2 brain (docs/03-SYSTEM-DESIGN.md §3.2-3.4).

    rays ──► RayEncoder (circular conv) ─────┐
    entities ─► EntityEncoder (attention) ───┤
    vitals + previous action ─► MLP ─────────┼─► fuse ─┬─► GRU (memory) ─► policy (6 action groups)
    mind-map crop ─► MapEncoder (small CNN) ─┘         │       └─► value (training only)
    notebook slots ─► NotebookReader ◄── query from the GRU state of the previous step
                          └──────────────────────────► (read vector joins the GRU input)

Phase 3 (docs/03-SYSTEM-DESIGN.md §3.3) adds the mind map and the landmark notebook; both are
switched by BrainConfig.use_mindmap / use_notebook (the memory ablation turns them off). Phase 4 adds
the manager and image encoders. The checkpoint stores its own config, and new config fields default
to "off", so Phase 2 checkpoints (no memory heads) still load and behave exactly as before.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import (
    ENTITY_FEATURES,
    MEMORY_KEYS,
    NUM_BUTTONS,
    NUM_ENTITY_TYPES,
    NUM_HIT_TYPES,
    NUM_RAYS,
    NUM_VITALS,
)
from .memory.mindmap import CROP, NUM_CHANNELS
from .memory.notebook import NOTE_FEATURES
from .policy.actions import GROUP_SIZES, NUM_LOGITS

CHECKPOINT_FORMAT = "smriti-brain-v1"

# Divide raw vitals by these so every input is roughly in [-2, 2] (see doom_env.spaces.VITALS_FIELDS).
VITALS_SCALE = np.array(
    [100, 100, 7, 100, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 16, 16, 16], dtype=np.float32
)
ENTITY_SCALE = np.array([1, 1, 1, 16, 16], dtype=np.float32)  # distance/1024, sin, cos, rel vel (units/tic)


@dataclass
class BrainConfig:
    ray_channels: int = 32
    ray_out: int = 128
    entity_dim: int = 64
    entity_heads: int = 4
    vitals_out: int = 32
    hidden: int = 256
    # Phase 3 memory. Defaults are OFF so Phase 2 checkpoints (whose config lacks these keys) load
    # unchanged; Phase 3 training turns them on (training/train.py --memory on).
    use_mindmap: bool = False
    use_notebook: bool = False
    map_out: int = 128
    note_dim: int = 64

    @property
    def uses_memory(self) -> bool:
        return self.use_mindmap or self.use_notebook


class RayEncoder(nn.Module):
    """64 rays form a ring, so convolutions wrap around (circular padding)."""

    def __init__(self, cfg: BrainConfig):
        super().__init__()
        c = cfg.ray_channels
        self.hit_emb = nn.Embedding(NUM_HIT_TYPES, 8)
        self.conv1 = nn.Conv1d(1 + 8, c, 5, padding=2, padding_mode="circular")
        self.conv2 = nn.Conv1d(c, c, 5, stride=2, padding=2, padding_mode="circular")
        self.conv3 = nn.Conv1d(c, c, 3, stride=2, padding=1, padding_mode="circular")
        self.out = nn.Linear(c * (NUM_RAYS // 4), cfg.ray_out)

    def forward(self, dist: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
        x = torch.cat([dist.unsqueeze(-1), self.hit_emb(hit)], dim=-1).transpose(1, 2)  # (N, 9, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return F.relu(self.out(x.flatten(1)))


class EntityEncoder(nn.Module):
    """Any number of visible entities (0-32) -> one vector, via attention with a learned query.

    A learned 'nothing here' token is always present, so an empty view is handled cleanly.
    The attention weights tell us which entity the agent focuses on (for the Mind View dashboard).
    """

    def __init__(self, cfg: BrainConfig):
        super().__init__()
        d = cfg.entity_dim
        self.register_buffer("scale", torch.from_numpy(ENTITY_SCALE), persistent=False)
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, 8)
        self.mlp = nn.Sequential(nn.Linear(ENTITY_FEATURES + 8, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.attn = nn.MultiheadAttention(d, cfg.entity_heads, batch_first=True)

    def forward(self, feat, etype, mask, return_weights: bool = False):
        n = feat.shape[0]
        tokens = self.mlp(torch.cat([feat / self.scale, self.type_emb(etype)], dim=-1))  # (N, 32, d)
        tokens = torch.cat([self.null.expand(n, -1, -1), tokens], dim=1)
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1)
        out, w = self.attn(self.query.expand(n, -1, -1), tokens, tokens, key_padding_mask=pad,
                           need_weights=return_weights)
        out = out.squeeze(1)
        return (out, w.squeeze(1)) if return_weights else out


class MapEncoder(nn.Module):
    """Small CNN on the rotated mind-map crop (NUM_CHANNELS x 32 x 32, uint8 0..255)."""

    def __init__(self, cfg: BrainConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 16, 3, stride=2, padding=1), nn.ReLU(),  # 16 x 16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 8 x 8
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # 4 x 4
            nn.Flatten(),
            nn.Linear(32 * (CROP // 8) ** 2, cfg.map_out), nn.ReLU(),
        )

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        return self.net(crop.float() / 255.0)


class NotebookReader(nn.Module):
    """Attention over the landmark slots. Keys/values come from each slot's egocentric features; the
    query comes from the GRU state, i.e. "what am I looking for right now?" (e.g. the locked door
    after picking up a key). A learned 'nothing noted' slot keeps an empty notebook well-defined."""

    def __init__(self, cfg: BrainConfig):
        super().__init__()
        d = cfg.note_dim
        self.slot = nn.Sequential(nn.Linear(NOTE_FEATURES, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.q = nn.Linear(cfg.hidden, d)
        self.null_k = nn.Parameter(torch.zeros(1, 1, d))
        self.null_v = nn.Parameter(torch.zeros(1, 1, d))
        self.scale = d ** -0.5

    def keys_values(self, notes: torch.Tensor, mask: torch.Tensor):
        """notes (N, S, F), mask (N, S) -> keys, values (N, S+1, d), valid (N, S+1). Batched over time."""
        n = notes.shape[0]
        e = self.slot(notes)
        k = torch.cat([self.null_k.expand(n, -1, -1), self.k(e)], dim=1)
        v = torch.cat([self.null_v.expand(n, -1, -1), self.v(e)], dim=1)
        valid = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        return k, v, valid

    def read(self, state: torch.Tensor, k, v, valid, return_weights: bool = False):
        q = self.q(state).unsqueeze(1)  # (N, 1, d)
        att = (q * k).sum(-1) * self.scale  # (N, S+1)
        att = att.masked_fill(~valid, float("-inf"))
        w = torch.softmax(att, dim=-1)
        out = (w.unsqueeze(-1) * v).sum(1)
        return (out, w) if return_weights else out


class SmritiNet(nn.Module):
    def __init__(self, cfg: BrainConfig | None = None):
        super().__init__()
        self.cfg = cfg or BrainConfig()
        c = self.cfg
        self.rays = RayEncoder(c)
        self.entities = EntityEncoder(c)
        self.register_buffer("vitals_scale", torch.from_numpy(VITALS_SCALE), persistent=False)
        self.vitals = nn.Sequential(nn.Linear(NUM_VITALS + NUM_BUTTONS, 64), nn.ReLU(), nn.Linear(64, c.vitals_out),
                                    nn.ReLU())
        fuse_in = c.ray_out + c.entity_dim + c.vitals_out
        if c.use_mindmap:
            self.mindmap = MapEncoder(c)
            fuse_in += c.map_out
        self.fuse = nn.Sequential(nn.Linear(fuse_in, c.hidden), nn.ReLU())
        if c.use_notebook:
            self.notebook = NotebookReader(c)
        self.gru = nn.GRUCell(c.hidden + (c.note_dim if c.use_notebook else 0), c.hidden)
        self.policy = nn.Linear(c.hidden, NUM_LOGITS)
        self.value = nn.Linear(c.hidden, 1)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)  # start close to uniform choices
        nn.init.zeros_(self.policy.bias)

    def initial_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.hidden, device=device)

    def encode(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-step encoders on a flat batch (N, ...) -> (N, hidden)."""
        vit = torch.clamp(x["vitals"] / self.vitals_scale, -5.0, 5.0)
        parts = [
            self.rays(x["ray_dist"], x["ray_hit"]),
            self.entities(x["ent_feat"], x["ent_type"], x["ent_mask"]),
            self.vitals(torch.cat([vit, x["prev_action"]], dim=-1)),
        ]
        if self.cfg.use_mindmap:
            parts.append(self.mindmap(x["map"]))
        return self.fuse(torch.cat(parts, dim=-1))

    def forward(self, x: dict[str, torch.Tensor], state: torch.Tensor, starts: torch.Tensor | None = None):
        """Sequence forward.

        x: features with leading dims (T, B); state: (B, hidden) memory before step 0;
        starts: (T, B) bool, True where a new episode begins (memory is wiped there).
        Returns logits (T, B, NUM_LOGITS), values (T, B), final state (B, hidden).
        """
        t, b = x["ray_dist"].shape[:2]
        flat = {k: v.reshape(t * b, *v.shape[2:]) for k, v in x.items()
                if k not in MEMORY_KEYS or self.cfg.uses_memory}
        emb = self.encode(flat).reshape(t, b, -1)
        if self.cfg.use_notebook:
            nk, nv, nvalid = self.notebook.keys_values(flat["notes"], flat["note_mask"])
            nk, nv, nvalid = (z.reshape(t, b, *z.shape[1:]) for z in (nk, nv, nvalid))
        outs = []
        for i in range(t):
            if starts is not None:
                state = state * (~starts[i]).unsqueeze(-1).to(state.dtype)
            inp = emb[i]
            if self.cfg.use_notebook:
                inp = torch.cat([inp, self.notebook.read(state, nk[i], nv[i], nvalid[i])], dim=-1)
            state = self.gru(inp, state)
            outs.append(state)
        h = torch.stack(outs)
        return self.policy(h), self.value(h).squeeze(-1), state


# ----------------------------------------------------------------------------- action distribution
def split_logits(logits: torch.Tensor) -> list[torch.Tensor]:
    return list(torch.split(logits, GROUP_SIZES, dim=-1))


def sample_actions(logits: torch.Tensor, generator: torch.Generator | None = None, greedy: bool = False):
    """logits (..., NUM_LOGITS) -> choices (..., 6) int64."""
    picks = []
    for lg in split_logits(logits):
        if greedy:
            picks.append(lg.argmax(-1))
        else:
            probs = torch.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
            picks.append(torch.multinomial(probs, 1, generator=generator).reshape(lg.shape[:-1]))
    return torch.stack(picks, dim=-1)


def log_prob_and_entropy(logits: torch.Tensor, choices: torch.Tensor):
    """Sum over the 6 groups of log-probability of `choices` and of entropy."""
    logp = torch.zeros(logits.shape[:-1], device=logits.device)
    ent = torch.zeros(logits.shape[:-1], device=logits.device)
    for g, lg in enumerate(split_logits(logits)):
        lsm = torch.log_softmax(lg, dim=-1)
        logp = logp + lsm.gather(-1, choices[..., g : g + 1]).squeeze(-1)
        ent = ent - (lsm.exp() * lsm).sum(-1)
    return logp, ent


# ----------------------------------------------------------------------------- checkpoints
def checkpoint_dict(net: SmritiNet, extra: dict | None = None) -> dict:
    """What a checkpoint file contains. Saving lives in training/ (the agent never writes files)."""
    return {"format": CHECKPOINT_FORMAT, "config": asdict(net.cfg), "state_dict": net.state_dict(),
            "extra": extra or {}}


def load_checkpoint(path, device="cpu") -> tuple[SmritiNet, dict]:
    """Load a checkpoint. Raises (never guesses) on an unknown format, unknown config keys or weights
    that do not match the config; DoomAgent catches that and falls back to the random policy."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {ckpt.get('format') if isinstance(ckpt, dict) else None!r}")
    net = SmritiNet(BrainConfig(**ckpt["config"])).to(device)
    net.load_state_dict(ckpt["state_dict"])  # strict: missing or extra memory heads raise
    net.eval()
    return net, ckpt.get("extra", {})
