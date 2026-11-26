"""Calibrate the agent's odometer: how turning and moving really work.

The observation has no position or heading. The agent only sees its own velocity (player frame,
units per tic) and knows which buttons it pressed. This script uses `privileged_state()`
(training-only ground truth) to measure:

1. Turning: degrees per step for a held TURN_LEFT/TURN_RIGHT (Doom turns slowly at first).
2. Moving: how to turn velocity into distance. Doom applies friction every tic, so the distance of
   one 4-tic step is best predicted from BOTH the current and the previous velocity:
       step_displacement = a * v_now + b * v_prev
3. Drift: how far a dead-reckoning odometer (velocity + button-based heading, i.e. only what the
   real agent knows) ends up from the true path.

    uv run python scripts/measure_odometry.py
    uv run python scripts/measure_odometry.py --map path/to/map.wad:MAP01 --steps 600 --out configs/odometry.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from doom_env import make_env
from doom_env import spaces as S

REPO = Path(__file__).resolve().parent.parent
VI = S.VITALS_INDEX
B = {name: i for i, name in enumerate(S.BUTTONS)}
WARMUP_STEPS = 5  # the first step after reset() is not applied by the engine; start measuring later


def parse_map(spec: str) -> tuple[str, str]:
    """Split "path/to.wad:MAP05" into (wad, map). Safe with Windows drive letters like "F:\\..."."""
    wad, sep, name = spec.rpartition(":")
    if sep and re.fullmatch(r"(MAP\d\d|E\dM\d)", name, re.IGNORECASE):
        return wad, name.upper()
    return spec, "MAP01"


def press(*names: str) -> np.ndarray:
    a = np.zeros(len(S.BUTTONS), dtype=np.int8)
    for n in names:
        a[B[n]] = 1
    return a


def angle_delta(a_new: float, a_old: float) -> float:
    """Signed smallest difference in degrees (positive = counter-clockwise = left)."""
    return (a_new - a_old + 180.0) % 360.0 - 180.0


def to_spawn_frame(v_fwd: float, v_right: float, heading_deg: float) -> np.ndarray:
    """Player-frame vector -> spawn-frame vector. Right is 90 degrees clockwise from forward."""
    h = math.radians(heading_deg)
    return np.array([v_fwd * math.cos(h) + v_right * math.sin(h), v_fwd * math.sin(h) - v_right * math.cos(h)])


# --------------------------------------------------------------------------- turning


def measure_turns(env, seed: int) -> dict:
    out: dict = {}
    env.reset(seed=seed)
    before = env.privileged_state()["angle"]
    env.step(press("TURN_LEFT"))
    out["first_step_after_reset_deg"] = round(angle_delta(env.privileged_state()["angle"], before), 3)

    for button in ("TURN_LEFT", "TURN_RIGHT"):
        env.reset(seed=seed)
        for _ in range(WARMUP_STEPS):
            env.step(press())
        held = []
        prev = env.privileged_state()["angle"]
        for _ in range(6):
            env.step(press(button))
            cur = env.privileged_state()["angle"]
            held.append(round(angle_delta(cur, prev), 3))
            prev = cur
        env.step(press())  # one released step resets Doom's slow-turn counter
        before = env.privileged_state()["angle"]
        env.step(press(button))
        after_release = round(angle_delta(env.privileged_state()["angle"], before), 3)
        out[button] = {"held_steps_deg": held, "after_one_release_step_deg": after_release}

    seq = out["TURN_LEFT"]["held_steps_deg"]
    out["model"] = {
        "held_sequence_deg": seq[:3],  # 1st, 2nd, 3rd+ consecutive held step
        "steady_deg": seq[-1],
        "release_resets": out["TURN_LEFT"]["after_one_release_step_deg"] == seq[0],
        "left_is_positive": seq[0] > 0,
    }
    return out


class TurnTracker:
    """Predicts the heading change from our own turn buttons (what the real agent can do)."""

    def __init__(self, model: dict):
        self.seq = model["held_sequence_deg"]
        self.held = 0

    def step(self, action: np.ndarray) -> float:
        left, right = int(action[B["TURN_LEFT"]]), int(action[B["TURN_RIGHT"]])
        # Doom counts how long ANY turn button is held (switching left<->right does not reset it).
        self.held = self.held + 1 if (left or right) else 0
        d = left - right
        return d * self.seq[min(self.held, len(self.seq)) - 1] if d else 0.0


# --------------------------------------------------------------------------- moving


def wander(env, seed: int, steps: int):
    """Scripted test pattern (forward, strafes, backward, turns); yields per-step data."""
    rng = np.random.default_rng(seed)
    patterns = [
        press("MOVE_FORWARD"),
        press("MOVE_FORWARD", "TURN_LEFT"),
        press("MOVE_FORWARD", "TURN_RIGHT"),
        press("MOVE_RIGHT"),
        press("MOVE_LEFT"),
        press("MOVE_BACKWARD", "TURN_LEFT"),
        press("TURN_RIGHT"),
        press(),
    ]
    env.reset(seed=seed)
    for _ in range(WARMUP_STEPS):
        env.step(press())
    action = patterns[0]
    for t in range(steps):
        if t % 6 == 0:
            action = patterns[int(rng.integers(0, len(patterns)))]
        s0 = env.privileged_state()
        obs, _, terminated, truncated, _ = env.step(action)
        s1 = env.privileged_state()
        yield action, obs, s0, s1
        if terminated or truncated:
            return


def fit_motion(env, seed: int, steps: int) -> dict:
    """Least-squares fit of displacement = a * v_now + b * v_prev (spawn frame, true heading)."""
    rows_v, rows_p, rows_t, near = [], [], [], []
    v_prev = np.zeros(2)
    for _, obs, s0, s1 in wander(env, seed, steps):
        v = to_spawn_frame(float(obs["vitals"][VI["vel_forward"]]), float(obs["vitals"][VI["vel_right"]]), s1["angle"])
        rows_v.append(v)
        rows_p.append(v_prev)
        rows_t.append([s1["x"] - s0["x"], s1["y"] - s0["y"]])
        near.append(float(np.min(obs["rays"]["distance"])) * S.RAY_RANGE < 40)
        v_prev = v
    V, P, T, near = map(np.array, (rows_v, rows_p, rows_t, near))
    A = np.concatenate([np.stack([V[~near, 0], P[~near, 0]], 1), np.stack([V[~near, 1], P[~near, 1]], 1)])
    y = np.concatenate([T[~near, 0], T[~near, 1]])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    err = np.linalg.norm(a * V + b * P - T, axis=1)
    naive = np.linalg.norm(S.FRAME_SKIP * V - T, axis=1)
    return {
        "a_now": round(float(a), 4),
        "b_prev": round(float(b), 4),
        "mean_step_error_open_units": round(float(err[~near].mean()), 3),
        "mean_step_error_near_wall_units": round(float(err[near].mean()), 3) if near.any() else None,
        "naive_4v_mean_step_error_units": round(float(naive.mean()), 3),
        "mean_true_step_length_units": round(float(np.linalg.norm(T, axis=1).mean()), 2),
        "fit_steps": int(len(T)),
    }


def measure_drift(env, seed: int, steps: int, motion: dict, turn_model: dict) -> dict:
    """Dead reckoning using only velocity + our own buttons, compared with the true path."""
    a, b = motion["a_now"], motion["b_prev"]
    tracker = TurnTracker(turn_model)
    pos, heading, v_prev = np.zeros(2), 0.0, np.zeros(2)
    first = None
    errors, heading_errors, travelled = [], [], 0.0
    for action, obs, s0, s1 in wander(env, seed + 1, steps):  # different seed than the fit
        if first is None:
            first = s0
        heading += tracker.step(action)
        v = to_spawn_frame(float(obs["vitals"][VI["vel_forward"]]), float(obs["vitals"][VI["vel_right"]]), heading)
        pos += a * v + b * v_prev
        v_prev = v
        dx, dy = s1["x"] - first["x"], s1["y"] - first["y"]
        c, s = math.cos(-math.radians(first["angle"])), math.sin(-math.radians(first["angle"]))
        true_pos = np.array([dx * c - dy * s, dx * s + dy * c])
        travelled += math.hypot(s1["x"] - s0["x"], s1["y"] - s0["y"])
        errors.append(float(np.linalg.norm(pos - true_pos)))
        heading_errors.append(abs(angle_delta(heading, s1["angle"] - first["angle"])))
    return {
        "steps": len(errors),
        "distance_travelled_units": round(travelled, 1),
        "final_position_error_units": round(errors[-1], 1),
        "max_position_error_units": round(max(errors), 1),
        "final_error_pct_of_distance": round(100 * errors[-1] / max(travelled, 1e-6), 2),
        "max_heading_error_deg": round(max(heading_errors), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=str(REPO / "maps" / "two_rooms.wad") + ":MAP01", help="wad[:MAPxx]")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="optional JSON file for the calibration")
    args = ap.parse_args()

    wad, map_name = parse_map(args.map)
    env = make_env(wad, map_name, render_buffers=False, episode_ticks=max(2100, (args.steps + 20) * 4))
    try:
        turns = measure_turns(env, args.seed)
        motion = fit_motion(env, args.seed, args.steps)
        drift = measure_drift(env, args.seed, args.steps, motion, turns["model"])
    finally:
        env.close()

    result = {
        "map": f"{Path(wad).name}:{map_name}",
        "frame_skip": S.FRAME_SKIP,
        "turns": turns,
        "motion": motion,
        "drift_with_agent_knowledge_only": drift,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
