"""Speed check for our agent under evaluation-like limits (one thread, official timeouts).

Measures load time, reset time, first act() (warm-up) and every other act(), then compares the
per-map compute time with the official budget (40 ms x decision steps). Our own target is stricter:
20 ms per decision on average, to leave room for slower hardware.

    uv run python scripts/benchmark_latency.py
    uv run python scripts/benchmark_latency.py --map maps/two_rooms.wad:MAP01 --episodes 3 --target-ms 20

Exit code 1 if any official limit or our target is broken. Only the agent's time is measured, not
the game engine's (same as the scorer).
"""

from __future__ import annotations

import agent_loader  # noqa: I001  (sets one-thread env vars before numpy loads)

agent_loader.apply_eval_thread_limits()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from doom_env import action_space, limits, make_env, observation_space  # noqa: E402
from map_spec import parse_map  # noqa: E402

REPO = agent_loader.REPO


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs), q)) if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", action="append", help="wad[:MAPxx], repeatable (default: template test map)")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per map")
    ap.add_argument("--ticks", type=int, default=2100, help="episode tick cap (2100-4200 at evaluation)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--weights", type=Path, help="brain checkpoint to test (default: submission/weights/champion.pt)")
    ap.add_argument("--target-ms", type=float, default=20.0, help="our own average-ms-per-decision target")
    ap.add_argument("--json", type=Path, help="optional path for a JSON report")
    args = ap.parse_args()

    pinned = agent_loader.pin_one_cpu()
    maps = args.map or [str(REPO / "maps" / "two_rooms.wad") + ":MAP01"]

    t0 = time.perf_counter()
    agent = agent_loader.load_agent_class()(observation_space(), action_space(), args.device, weights_path=args.weights)
    init_s = time.perf_counter() - t0

    report: dict = {"device": args.device, "pinned_one_cpu": pinned, "init_s": round(init_s, 3), "runs": []}
    failures: list[str] = []
    if init_s > limits.INIT_TIMEOUT_S:
        failures.append(f"init {init_s:.1f}s > {limits.INIT_TIMEOUT_S}s")

    for spec in maps:
        wad, map_name = parse_map(spec)
        env = make_env(wad, map_name, episode_ticks=args.ticks)
        budget_s = limits.agent_budget_s(args.ticks)
        try:
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=ep)
                t0 = time.perf_counter()
                agent.reset()
                reset_s = time.perf_counter() - t0
                times: list[float] = []
                done = False
                while not done:
                    t0 = time.perf_counter()
                    action = agent.act(obs)
                    times.append(time.perf_counter() - t0)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                ms = [1000 * t for t in times]
                run = {
                    "map": f"{Path(wad).name}:{map_name}",
                    "episode": ep,
                    "steps": len(times),
                    "reset_s": round(reset_s, 4),
                    "first_act_ms": round(ms[0], 2),
                    "mean_ms": round(float(np.mean(ms[1:] or ms)), 3),
                    "p50_ms": round(percentile(ms[1:], 50), 3),
                    "p99_ms": round(percentile(ms[1:], 99), 3),
                    "max_ms": round(max(ms[1:] or ms), 2),
                    "agent_time_s": round(sum(times), 3),
                    "budget_s": round(budget_s, 1),
                }
                report["runs"].append(run)
                label = f"{run['map']} ep{ep}"
                if reset_s > limits.RESET_TIMEOUT_S:
                    failures.append(f"{label}: reset {reset_s:.1f}s > {limits.RESET_TIMEOUT_S}s")
                if times[0] > limits.FIRST_ACT_TIMEOUT_S:
                    failures.append(f"{label}: first act {times[0]:.1f}s > {limits.FIRST_ACT_TIMEOUT_S}s")
                if max(times[1:] or [0]) > limits.ACT_TIMEOUT_S:
                    failures.append(f"{label}: an act() took > {limits.ACT_TIMEOUT_S}s")
                if sum(times) > budget_s:
                    failures.append(f"{label}: agent time {sum(times):.1f}s > budget {budget_s:.1f}s")
                if run["mean_ms"] > args.target_ms:
                    failures.append(f"{label}: mean {run['mean_ms']:.1f} ms > our target {args.target_ms} ms")
        finally:
            env.close()

    print(f"Policy {agent.policy_name}, init {init_s:.2f}s on {args.device}" + ("" if pinned else "  (not pinned to 1 CPU: Linux only)"))
    print(f"{'map':<28}{'ep':>3}{'steps':>7}{'reset s':>9}{'first ms':>10}{'mean ms':>9}{'p99 ms':>8}{'max ms':>8}{'agent/budget s':>17}")
    for r in report["runs"]:
        print(
            f"{r['map']:<28}{r['episode']:>3}{r['steps']:>7}{r['reset_s']:>9.3f}{r['first_act_ms']:>10.2f}"
            f"{r['mean_ms']:>9.3f}{r['p99_ms']:>8.3f}{r['max_ms']:>8.2f}{r['agent_time_s']:>9.2f} / {r['budget_s']:<6}"
        )
    report["failures"] = failures
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nFAILED:\n  " + "\n  ".join(failures) if failures else f"\nPASSED (target {args.target_ms} ms)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
