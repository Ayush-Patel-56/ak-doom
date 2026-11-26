"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6. Phase 1: the policy is still random, but every decision already goes
through the SafetyGuard, so a bug in a future policy can never make the player stand still.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"

# The scorer and validate.py already put submission/ on sys.path; this keeps our own scripts working too.
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from smriti import SafetyGuard  # noqa: E402
from smriti.policy import RandomPolicy  # noqa: E402


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load weights here (from WEIGHTS_DIR) onto `device` ("cuda" or "cpu")."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.guard = SafetyGuard(observation_space, action_space)
        self.policy = RandomPolicy(int(np.prod(action_space.shape)), seed=0)

    def reset(self) -> None:
        """Called before every map. Clears all per-map state."""
        self.policy.reset()
        self.guard.reset()

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int8 array of shape (9,) with values in {0, 1}."""
        return self.guard.act(observation, self.policy)
