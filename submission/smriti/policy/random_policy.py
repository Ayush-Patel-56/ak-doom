"""Random baseline (same behaviour as the template's agent).

It is the Phase 1 placeholder: it lets us test the whole pipeline (safety guard, validator,
benchmark, Mini-League) and gives the score every trained agent must beat.
"""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    def __init__(self, action_size: int, seed: int = 0):
        self.action_size = action_size
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        # Re-seed per map so the same map always gives the same actions (reproducible evaluation).
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, observation: dict) -> np.ndarray:
        return self.rng.integers(0, 2, size=self.action_size, dtype=np.int8)
