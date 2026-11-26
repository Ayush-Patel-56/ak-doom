"""BrainPolicy: runs SmritiNet one step at a time inside the agent (keeps memory between steps)."""

from __future__ import annotations

import numpy as np
import torch

from ..brain import load_checkpoint, sample_actions
from ..features import empty, extract
from .actions import groups_to_buttons


class BrainPolicy:
    def __init__(self, weights_path, device: str = "cpu", seed: int = 0, greedy: bool = False):
        self.device = torch.device(device)
        self.net, self.meta = load_checkpoint(weights_path, self.device)
        self.seed = seed
        self.greedy = greedy
        self.generator = torch.Generator(device=self.device)
        self._warm_up()
        self.reset()

    @torch.inference_mode()
    def _warm_up(self, steps: int = 3) -> None:
        """PyTorch's first forward pass is slow (seconds). Pay it here, inside __init__ (60 s limit),
        instead of in the first act() of a map (10 s limit)."""
        x = {k: torch.as_tensor(v, device=self.device)[None, None] for k, v in empty(()).items()}
        state = self.net.initial_state(1, self.device)
        for _ in range(steps):
            logits, _, state = self.net(x, state)
            sample_actions(logits[0, 0], self.generator, self.greedy)

    def reset(self) -> None:
        self.state = self.net.initial_state(1, self.device)
        self.prev_action = np.zeros(9, np.float32)
        self.generator.manual_seed(self.seed)  # same map -> same actions (reproducible evaluation)

    @torch.inference_mode()
    def __call__(self, obs: dict) -> np.ndarray:
        feats = extract(obs, self.prev_action)
        x = {k: torch.as_tensor(v, device=self.device)[None, None] for k, v in feats.items()}  # (T=1, B=1)
        logits, _, self.state = self.net(x, self.state)
        choices = sample_actions(logits[0, 0], self.generator, self.greedy).cpu().numpy()
        buttons = groups_to_buttons(choices)
        self.prev_action = buttons.astype(np.float32)
        return buttons
