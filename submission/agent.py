"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6. Every decision goes through the SafetyGuard, so a bug in the policy
can never make the player stand still.

Policy: the trained SmritiNet from weights/champion.pt when it exists; otherwise the random baseline
(so the repository stays submittable before the first champion is exported).

Memory (Phase 3): an EpisodeMemory (odometer + mind map + landmark notebook) is updated on every
decision from the raw observation and the action we actually executed, and the brain reads it when its
checkpoint was trained with memory. Phase 2 checkpoints (no memory heads) keep working unchanged.
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
from smriti.memory import EpisodeMemory, Odometer, Pose  # noqa: E402


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str, weights_path: str | Path | None = None):
        """Called once. Loads weights (default: weights/champion.pt). `device` is "cuda" or "cpu"."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.guard = SafetyGuard(observation_space, action_space)
        # A bad calibration (missing file, ViZDoom/contract mismatch) must never crash
        # __init__: that scores 0 on EVERY map. Disable odometry and keep acting.
        self._previous_executed_action = None
        odometer = None
        init_error = None
        try:
            odometer = Odometer()
        except Exception as exc:  # noqa: BLE001
            print(f"[DoomAgent] could not initialise odometer ({type(exc).__name__}: {exc}); "
                  "odometry disabled", file=sys.stderr)
            init_error = str(exc)
        # Map/notebook drawing is switched on below only if the loaded brain reads them.
        self.memory = EpisodeMemory(odometer, use_mindmap=False, use_notebook=False)
        self._odometer_init_error = init_error
        path = Path(weights_path) if weights_path is not None else CHAMPION
        self.policy = None
        if path.is_file():
            # A bad champion (un-fetched Git LFS pointer, truncated file, old checkpoint format, torch
            # problem) must never crash __init__: that scores 0 on EVERY map. Fall back to random instead.
            try:
                import torch

                torch.set_num_threads(1)  # evaluation gives us one CPU thread
                from smriti.policy.brain_policy import BrainPolicy

                policy = BrainPolicy(path, device=INFERENCE_DEVICE, seed=0)
                cfg = policy.net.cfg
                if cfg.uses_memory:
                    self.memory = EpisodeMemory(odometer, use_mindmap=cfg.use_mindmap,
                                                use_notebook=cfg.use_notebook)
                    policy.memory = self.memory
                self.policy = policy
                self.policy_name = f"brain:{path.name}" + (" +memory" if cfg.uses_memory else "")
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
        self._previous_executed_action = None
        self.memory.reset()

    # Odometry status (read-only; kept for memory/dashboard consumers and tests).
    @property
    def odometer(self) -> Odometer | None:
        return self.memory.odometer

    @property
    def odometry_valid(self) -> bool:
        return self.memory.valid

    @property
    def odometry_error(self) -> str | None:
        return self._odometer_init_error if self.memory.odometer is None else self.memory.error

    @property
    def pose(self) -> Pose:
        """Read-only spawn-relative pose for memory/dashboard consumers; check odometry_valid."""
        return self.memory.pose

    @property
    def active_goal(self):
        """Learned goal index, or None for checkpoints without a manager."""
        return getattr(self.policy, 'active_goal', None)

    @property
    def exit_probability(self):
        """RGB head's learned score; None for checkpoints without RGB."""
        return getattr(self.policy, 'exit_probability', None)

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int8 array of shape (9,) with values in {0, 1}."""
        was_valid = self.memory.valid
        try:
            # Odometry uses raw motion data (SafetyGuard's zero-filled replacement is not a measurement);
            # the map and notebook draw the cleaned observation.
            clean = self.guard.clean(observation) if (self.memory.use_mindmap or self.memory.use_notebook) else None
            self.memory.observe(self._previous_executed_action, observation, clean)
        except Exception as exc:  # noqa: BLE001  (memory must never stop the agent from acting)
            self.memory.valid = False
            self.memory.error = f"{type(exc).__name__}: {exc}"
        if was_valid and not self.memory.valid:
            print(f"[DoomAgent] odometry invalid until reset: {self.memory.error}", file=sys.stderr)
        action = self.guard.act(observation, self.policy)
        # Includes cleaned policy outputs and fallback actions. Never use the intended policy action.
        self._previous_executed_action = action.copy()
        if hasattr(self.policy, 'record_executed_action'):
            self.policy.record_executed_action(action)
        return action
