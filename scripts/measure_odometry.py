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
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

from doom_env import CONTRACT_VERSION, make_env
from doom_env import spaces as S

from map_spec import parse_map

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'submission'))
from smriti.memory.odometer import Calibration, Odometer  # noqa: E402

VI = S.VITALS_INDEX
B = {name: i for i, name in enumerate(S.BUTTONS)}
FINE_ANGLES_PER_CIRCLE = 8192  # Doom's fine-angle grid; 360° / 8192
# First step after reset() is a no-op. The steady turn/move ramp only appears after
# at least 2 no-op steps (held turn from the first action: 0°, 3.515625°, 14.0625°;
# after exactly one no-op, the next left is 1.7578125°; movement follows the same
# pattern). Mid-episode turns are normal. We wait 5 steps so calibration starts
# on the clean ramp. The production odometer models these startup transitions without imposing no-op actions.
WARMUP_STEPS = 5
STARTUP_NOOP_STEPS = 2


def snap_deg(deg: float) -> float:
    """Snap a measured heading change onto Doom's fine-angle grid."""
    fine = int(round(deg * FINE_ANGLES_PER_CIRCLE / 360.0))
    return fine * 360.0 / FINE_ANGLES_PER_CIRCLE


def deg_to_fine(deg: float) -> int:
    return int(round(deg * FINE_ANGLES_PER_CIRCLE / 360.0))


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
    out["first_step_after_reset_deg"] = snap_deg(angle_delta(env.privileged_state()["angle"], before))

    for button in ("TURN_LEFT", "TURN_RIGHT"):
        env.reset(seed=seed)
        for _ in range(WARMUP_STEPS):
            env.step(press())
        held = []
        prev = env.privileged_state()["angle"]
        for _ in range(6):
            env.step(press(button))
            cur = env.privileged_state()["angle"]
            held.append(snap_deg(angle_delta(cur, prev)))
            prev = cur
        env.step(press())  # one released step resets Doom's slow-turn counter
        before = env.privileged_state()["angle"]
        env.step(press(button))
        after_release = snap_deg(angle_delta(env.privileged_state()["angle"], before))
        out[button] = {
            "held_steps_deg": held,
            "held_steps_fine": [deg_to_fine(d) for d in held],
            "after_one_release_step_deg": after_release,
            "after_one_release_step_fine": deg_to_fine(after_release),
        }

    seq = out["TURN_LEFT"]["held_steps_deg"]
    seq_fine = out["TURN_LEFT"]["held_steps_fine"]
    out["model"] = {
        "held_sequence_deg": seq[:3],  # 1st, 2nd, 3rd+ consecutive held step
        "held_sequence_fine": seq_fine[:3],
        "steady_deg": seq[-1],
        "steady_fine": seq_fine[-1],
        "fine_angles_per_circle": FINE_ANGLES_PER_CIRCLE,
        "release_resets": out["TURN_LEFT"]["after_one_release_step_deg"] == seq[0],
        "left_is_positive": seq[0] > 0,
    }
    return out


def measure_startup(env, seed: int) -> dict:
    """Measure un-warmed turning, including a previously held opposing pair."""
    out = {}
    for button, suffix in (("TURN_LEFT", "left"), ("TURN_RIGHT", "right")):
        env.reset(seed=seed)
        initial = env.privileged_state()["angle"]
        env.step(press(button))
        out[f"first_{suffix}"] = snap_deg(angle_delta(env.privileged_state()["angle"], initial))
        second = []
        for previous in (press(), press("TURN_LEFT", "TURN_RIGHT")):
            env.reset(seed=seed)
            env.step(previous)
            before = env.privileged_state()["angle"]
            env.step(press(button))
            second.append(snap_deg(angle_delta(env.privileged_state()["angle"], before)))
        out[f"second_{suffix}"] = second
    return out


def calibration_data(turns: dict, startup: dict, motion: dict, *, seed: int, steps: int, map_spec: str) -> dict:
    return {
        "schema_version": 1,
        "engine_version": version("vizdoom"),
        "contract_version": CONTRACT_VERSION,
        "frame_skip": S.FRAME_SKIP,
        "buttons": list(S.BUTTONS),
        "vitals_fields": list(S.VITALS_FIELDS),
        "turns": {
            "left": turns["TURN_LEFT"]["held_steps_deg"],
            "right": turns["TURN_RIGHT"]["held_steps_deg"],
            **startup,
        },
        "motion": {"a_now": motion["a_now"], "b_prev": motion["b_prev"]},
        "provenance": {"map": map_spec, "seed": seed, "requested_steps": steps,
                       "fit_steps": motion["fit_steps"], "tool": "scripts/measure_odometry.py"},
    }


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
        if terminated or truncated:
            return
        s1 = env.privileged_state()
        yield action, obs, s0, s1


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
    if len(T) < 2 or not np.isfinite(A).all() or np.linalg.matrix_rank(A) < 2:
        raise ValueError("insufficient independent open-space motion samples for calibration")
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


def measure_drift(env, seed: int, steps: int, calibration: Calibration) -> dict:
    """Validate the PRODUCTION estimator on a held-out real episode, with no warmup.

    Ground truth establishes the comparison frame only. Odometer receives actions
    and legal observations, never the privileged states used by this reporter.
    """
    odometer = Odometer(calibration)
    rng = np.random.default_rng(seed)
    patterns = [press("MOVE_FORWARD"), press("MOVE_BACKWARD"), press("MOVE_LEFT"), press("MOVE_RIGHT"),
                press("MOVE_FORWARD", "TURN_LEFT"), press("MOVE_FORWARD", "TURN_RIGHT"),
                press("TURN_LEFT"), press("TURN_RIGHT"), press("TURN_LEFT", "TURN_RIGHT"), press()]
    env.reset(seed=seed)
    first = previous = env.privileged_state()
    errors, heading_errors, travelled = [], [], 0.0
    for t in range(steps):
        if t % 6 == 0:
            action = patterns[int(rng.integers(len(patterns)))]
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break  # terminal observations are stale in the official wrapper
        state = env.privileged_state()
        pose = odometer.update(action, obs)
        dx, dy = state["x"] - first["x"], state["y"] - first["y"]
        c, s = math.cos(math.radians(first["angle"])), math.sin(math.radians(first["angle"]))
        true_pos = np.array([dx * c + dy * s, -dx * s + dy * c])
        travelled += math.hypot(state["x"] - previous["x"], state["y"] - previous["y"])
        previous = state
        errors.append(float(np.linalg.norm(np.array([pose.x, pose.y]) - true_pos)))
        heading_errors.append(abs(angle_delta(pose.heading_deg, state["angle"] - first["angle"])))
    if not errors:
        raise ValueError("validation episode produced no nonterminal transitions")
    return {
        "seed": seed,
        "steps": len(errors),
        "distance_travelled_units": round(travelled, 1),
        "final_position_error_units": round(errors[-1], 1),
        "max_position_error_units": round(max(errors), 1),
        "final_error_pct_of_distance": round(100 * errors[-1] / max(travelled, 1e-6), 2),
        "max_heading_error_deg": round(max(heading_errors), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=str(REPO / "maps" / "two_rooms.wad") + ":MAP01", help="wad[:MAPxx]")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="optional JSON file for the calibration")
    ap.add_argument("--calibration-out", type=Path, help="write submission-ready measured calibration")
    args = ap.parse_args()
    if args.steps < 2:
        ap.error("--steps must be at least 2")

    wad, map_name = parse_map(args.map)
    env = make_env(wad, map_name, render_buffers=False, episode_ticks=max(2100, (args.steps + 20) * 4))
    try:
        turns = measure_turns(env, args.seed)
        startup = measure_startup(env, args.seed)
        motion = fit_motion(env, args.seed, args.steps)
        data = calibration_data(turns, startup, motion, seed=args.seed, steps=args.steps,
                                map_spec=f"{Path(wad).name}:{map_name}")
        calibration = Calibration.from_dict(data)
        drift = measure_drift(env, args.seed + 1, args.steps, calibration)
    finally:
        env.close()

    result = {
        "map": f"{Path(wad).name}:{map_name}",
        "frame_skip": S.FRAME_SKIP,
        "startup": {
            "first_step_after_reset_is_noop": turns["first_step_after_reset_deg"] == 0.0,
            "startup_noop_steps": STARTUP_NOOP_STEPS,
            "measurement_warmup_steps": WARMUP_STEPS,
            "note": (
                "First step after reset is a no-op (0 turn, 0 move; tic still advances). "
                "The steady turn/move ramp only appears after at least 2 no-op steps. "
                "Held turn from the first action: 0, 3.515625, 14.0625 deg (skips the 12.3046875 deg step). "
                "After exactly one no-op, the next left is 1.7578125, then 12.3046875, then 14.0625 deg. "
                "Movement follows the same pattern. Mid-episode turns are normal. "
                "The odometer models startup turns without requiring no-op actions."
            ),
        },
        "turns": turns,
        "startup_turn_measurements": startup,
        "motion": motion,
        "drift_with_agent_knowledge_only": drift,
    }
    if args.calibration_out:
        args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
