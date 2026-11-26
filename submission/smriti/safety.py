"""Safety guard: the agent must never stop.

If the agent raises, returns an invalid action or times out, the scorer makes the player stand still
for the rest of the map (docs/EVALUATION.md). The guard wraps every decision:

1. clean the observation (missing keys or wrong shapes -> zeros, NaN/inf -> 0),
2. run the policy,
3. turn its output into a valid action (shape (9,), values 0/1),
4. on any error, return SAFE_ACTION instead and log the error once.

It also tracks our own compute time per map, so later phases can switch off optional encoders when
we run close to the 40 ms-per-step budget.
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Any, Callable

import numpy as np

# Buttons: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, TURN_LEFT, TURN_RIGHT, ATTACK, USE,
# SELECT_NEXT_WEAPON. Fallback only: walk forward while slowly turning, so the player keeps moving.
SAFE_ACTION = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)

Policy = Callable[[dict], Any]


def zeros_like_space(space) -> Any:
    """A zero observation with the exact structure of a gymnasium space (Dict, Box, Multi*)."""
    sub = getattr(space, "spaces", None)
    if isinstance(sub, dict):
        return {k: zeros_like_space(v) for k, v in sub.items()}
    return np.zeros(space.shape, dtype=space.dtype)


class SafetyGuard:
    def __init__(self, observation_space, action_space, *, log=sys.stderr, max_logged_errors: int = 3):
        self.template = zeros_like_space(observation_space)
        self.action_size = int(np.prod(action_space.shape))
        self.safe_action = SAFE_ACTION.copy() if self.action_size == SAFE_ACTION.size else np.zeros(
            self.action_size, dtype=np.int8
        )
        self.log = log
        self.max_logged_errors = max_logged_errors
        self.errors_logged = 0
        self.reset()

    # ----------------------------------------------------------------------- per map
    def reset(self) -> None:
        self.steps = 0
        self.fallbacks = 0
        self.compute_s = 0.0
        self.last_ms = 0.0

    @property
    def mean_ms(self) -> float:
        return 1000.0 * self.compute_s / self.steps if self.steps else 0.0

    # ----------------------------------------------------------------------- per step
    def act(self, observation: Any, policy: Policy) -> np.ndarray:
        t0 = time.perf_counter()
        try:
            action = self.to_action(policy(self.clean(observation)))
        except Exception as exc:  # noqa: BLE001 - any failure must fall back, never propagate
            self._log_error(exc)
            self.fallbacks += 1
            action = self.safe_action.copy()
        dt = time.perf_counter() - t0
        self.steps += 1
        self.compute_s += dt
        self.last_ms = 1000.0 * dt
        return action

    def clean(self, observation: Any) -> dict:
        """Return an observation with the official structure; bad parts are replaced by zeros."""
        return self._clean(observation, self.template)

    def _clean(self, value: Any, template: Any) -> Any:
        if isinstance(template, dict):
            src = value if isinstance(value, dict) else {}
            return {k: self._clean(src.get(k), t) for k, t in template.items()}
        try:
            arr = np.asarray(value)
            if arr.shape != template.shape:
                return template.copy()
            if arr.dtype.kind == "f":
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.dtype.kind in 'fiu' and template.dtype.kind in 'fiu':
                bounds = np.finfo(template.dtype) if template.dtype.kind == 'f' else np.iinfo(template.dtype)
                # Replace unrepresentable values before casting; never create inf or integer wraparound.
                upper = np.nextafter(float(bounds.max), 0.) if arr.dtype.kind == 'f' and template.dtype.kind in 'iu' and template.dtype.itemsize == 8 else bounds.max
                arr = np.where((arr < bounds.min) | (arr > upper), 0, arr)
            return arr.astype(template.dtype, copy=False)
        except Exception:  # noqa: BLE001
            return template.copy()

    def to_action(self, raw: Any) -> np.ndarray:
        """Any array-like with action_size elements -> int8 array of 0/1. Raises if impossible."""
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size != self.action_size:
            raise ValueError(f"policy returned {arr.size} values, expected {self.action_size}")
        arr = np.nan_to_num(arr, nan=0.0)
        return (arr > 0.5).astype(np.int8)

    def _log_error(self, exc: BaseException) -> None:
        if self.errors_logged >= self.max_logged_errors:
            return
        self.errors_logged += 1
        last = traceback.format_exception_only(type(exc), exc)[-1].strip()
        print(f"[smriti] step {self.steps}: fallback to SAFE_ACTION after {last}", file=self.log, flush=True)
