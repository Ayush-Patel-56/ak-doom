"""Promote a trained checkpoint to submission/weights/champion.pt (what the agent loads).

Safety checks before copying: the checkpoint loads, the agent built from it gives valid actions, and
its speed is inside our 20 ms target on one thread. Pass the Mini-League result so it's recorded with
the weights (docs/04-TRAINING-PLAN.md §4.8: a checkpoint replaces the champion only if it scores
higher, has 0 crashes and is fast enough).

    uv run python scripts/export_champion.py runs/stage12-v1/latest.pt --eval runs/eval/stage12-v1.json

Then commit submission/weights/champion.pt (Git LFS) in a pull request.
"""

from __future__ import annotations

import agent_loader  # noqa: I001

agent_loader.apply_eval_thread_limits()

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from doom_env import action_space, observation_space  # noqa: E402

REPO = agent_loader.REPO
CHAMPION = REPO / "submission" / "weights" / "champion.pt"
CHAMPION_INFO = REPO / "submission" / "weights" / "champion.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--eval", type=Path, help="Mini-League JSON (scripts/evaluate.py output) for this checkpoint")
    ap.add_argument("--target-ms", type=float, default=20.0)
    ap.add_argument("--force", action="store_true", help="replace even if the eval score is lower than the champion's")
    args = ap.parse_args()

    agent = agent_loader.load_agent_class()(observation_space(), action_space(), "cpu", weights_path=args.checkpoint)
    if not agent.policy_name.startswith("brain"):
        print(f"{args.checkpoint} did not load as a brain")
        return 1
    obs_space = observation_space()
    obs_space.seed(0)
    agent.reset()
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        a = agent.act(obs_space.sample())
        times.append(time.perf_counter() - t0)
        if a.shape != (9,) or not np.isin(a, (0, 1)).all():
            print("invalid action")
            return 1
    mean_ms = 1000 * float(np.mean(times[10:]))
    if agent.guard.fallbacks:
        print(f"{agent.guard.fallbacks} fallbacks to SAFE_ACTION: the brain raised errors")
        return 1
    if mean_ms > args.target_ms:
        print(f"too slow: {mean_ms:.1f} ms > {args.target_ms} ms")
        return 1

    src = args.checkpoint.resolve()
    info = {"source": (src.relative_to(REPO) if src.is_relative_to(REPO) else src).as_posix(),
            "mean_ms": round(mean_ms, 2), "meta": agent.policy.meta}
    if args.eval:
        ev = json.loads(args.eval.read_text(encoding="utf-8"))
        crashes = sum(1 for r in ev["runs"] if r.get("stopped"))
        info["mini_league"] = {"file": args.eval.name, "total": ev["total"], "by_tier": ev["by_tier"], "crashes": crashes}
        if crashes:
            print(f"refusing: {crashes} stopped runs in the Mini-League")
            return 1
        if CHAMPION_INFO.is_file() and not args.force:
            old = json.loads(CHAMPION_INFO.read_text(encoding="utf-8")).get("mini_league", {}).get("total")
            if old is not None and ev["total"] <= old:
                print(f"refusing: Mini-League {ev['total']} is not better than the champion's {old} (use --force)")
                return 1
    CHAMPION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.checkpoint, CHAMPION)
    CHAMPION_INFO.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"champion <- {args.checkpoint} ({mean_ms:.2f} ms/step)\n{json.dumps(info, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
