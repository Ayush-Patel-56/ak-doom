"""BrainPolicy: runs SmritiNet one step at a time inside the agent (keeps memory between steps).

When the network uses Phase 3 memory, the owner (DoomAgent) updates an EpisodeMemory every decision
and sets `policy.memory` to it; BrainPolicy only reads its features. Without memory (or when it is not
attached) the memory inputs are zeros.
"""

from __future__ import annotations

import numpy as np
import torch

from ..brain import load_checkpoint
from ..hierarchy import GoalState, policy_step
from ..features import empty, extract
from .actions import groups_to_buttons


class BrainPolicy:
    def __init__(self, weights_path, device: str = "cpu", seed: int = 0, greedy: bool = False):
        self.device = torch.device(device)
        self.net, self.meta = load_checkpoint(weights_path, self.device)
        self.seed = seed
        self.greedy = greedy
        self.generator = torch.Generator(device=self.device)
        self.memory = None  # EpisodeMemory, attached by DoomAgent when self.uses_memory
        self._warm_up()
        self.reset()

    @torch.inference_mode()
    def _warm_up(self, steps: int = 3) -> None:
        """PyTorch's first forward pass is slow (seconds). Pay it here, inside __init__ (60 s limit),
        instead of in the first act() of a map (10 s limit)."""
        x = {k: torch.as_tensor(v, device=self.device)[None] for k, v in empty(()).items()}
        state = self.net.initial_state(1, self.device)
        goals = GoalState.initial(1, self.device)
        for _ in range(steps):
            _, _, state, goals, _ = policy_step(self.net, x, state,
                torch.zeros(1, dtype=torch.bool, device=self.device), goals, self.generator, self.greedy)

    @property
    def uses_memory(self) -> bool:
        return self.net.cfg.uses_memory

    def reset(self) -> None:
        self.state = self.net.initial_state(1, self.device)
        self.goals = GoalState.initial(1, self.device)
        self.active_goal = None
        self.exit_probability = None
        self.prev_action = np.zeros(9, np.float32)
        self.generator.manual_seed(self.seed)  # same map -> same actions (reproducible evaluation)

    @torch.inference_mode()
    def __call__(self, obs: dict) -> np.ndarray:
        mem = self.memory.features() if (self.memory is not None and self.uses_memory) else None
        feats = extract(obs, self.prev_action, mem)
        x = {k: torch.as_tensor(v, device=self.device)[None] for k, v in feats.items()}
        choices, _, self.state, self.goals, aux = policy_step(self.net, x, self.state,
            torch.zeros(1, dtype=torch.bool, device=self.device), self.goals, self.generator, self.greedy)
        self.active_goal = int(aux['goal'][0]) if self.net.cfg.use_manager else None
        self.exit_probability = float(aux['exit_logits'][0].sigmoid()) if self.net.cfg.use_rgb else None
        choices = choices[0].cpu().numpy()
        buttons = groups_to_buttons(choices)
        self.prev_action = buttons.astype(np.float32)
        return buttons

    def record_executed_action(self, action):
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
