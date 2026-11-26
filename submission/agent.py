"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6. Every decision goes through the SafetyGuard, so a bug in the policy
can never make the player stand still.

Policy: the trained SmritiNet from weights/champion.pt when it exists; otherwise the random baseline
(so the repository stays submittable before the first champion is exported).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
CHAMPION = WEIGHTS_DIR / "champion.pt"

# The brain is tiny and runs one observation at a time: on one CPU thread it is faster than paying GPU
# launch/transfer costs on a GPU shared with other teams. Revisit when image encoders arrive.
INFERENCE_DEVICE = "cpu"

# The scorer and validate.py already put submission/ on sys.path; this keeps our own scripts working too.
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from smriti import SafetyGuard  # noqa: E402
from smriti.policy import RandomPolicy  # noqa: E402


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str, weights_path: str | Path | None = None):
        """Called once. Loads weights (default: weights/champion.pt). `device` is "cuda" or "cpu"."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.guard = SafetyGuard(observation_space, action_space)
        path = Path(weights_path) if weights_path is not None else CHAMPION
        self.policy = None
        if path.is_file():
            # A bad champion (un-fetched Git LFS pointer, truncated file, old checkpoint format, torch
            # problem) must never crash __init__: that scores 0 on EVERY map. Fall back to random instead.
            try:
                import torch

                torch.set_num_threads(1)  # evaluation gives us one CPU thread
                from smriti.policy.brain_policy import BrainPolicy

                self.policy = BrainPolicy(path, device=INFERENCE_DEVICE, seed=0)
                self.policy_name = f"brain:{path.name}"
            except Exception as exc:  # noqa: BLE001
                print(f"[DoomAgent] could not load {path.name} ({type(exc).__name__}: {exc}); "
                      "using the random policy", file=sys.stderr)
                self.policy = None
        if self.policy is None:
            self.policy = RandomPolicy(int(np.prod(action_space.shape)), seed=0)
            self.policy_name = "random"

    def reset(self) -> None:
        """Called before every map. Clears all per-map state."""
        self.policy.reset()
        self.guard.reset()

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int8 array of shape (9,) with values in {0, 1}."""
        return self.guard.act(observation, self.policy)
