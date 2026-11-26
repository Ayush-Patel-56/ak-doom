"""Mini-League: play our agent on held-out maps and score it like the official scorer would.

Follows the official rules (docs/EVALUATION.md):
- skill 3, freedoom2 IWAD, fixed seeds, per-map tick cap (from maps/catalog.json, 2,100-4,200)
- the agent's compute time per map is limited to 40 ms x decision steps; act() calls over 5 s, an
  exception, or an invalid action make the agent STOP: the player stands still for the rest of the map
  (points earned so far are kept)
- reset() over 10 s gives 0 points for that map; __init__ over 60 s is flagged with a warning
- partial progress counts; harder tiers weigh more (see training/scorer.py for our guessed weights)

    uv run python scripts/evaluate.py                       # validation maps (maps/lists/val.txt)
    uv run python scripts/evaluate.py --split test          # final, honest score only
    uv run python scripts/evaluate.py --map maps/two_rooms.wad:MAP01 --name smoke

Results: a table on screen and a JSON file in runs/eval/<name>.json (git-ignored).
"""

from __future__ import annotations

import agent_loader  # noqa: I001  (sets one-thread env vars before numpy loads)

agent_loader.apply_eval_thread_limits()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from doom_env import action_space, limits, make_env, observation_space  # noqa: E402
from league_record import file_sha256, fingerprint, map_set_of  # noqa: E402
from map_spec import REPO, parse_map  # noqa: E402

sys.path.insert(0, str(REPO))
from training.scorer import TIER_WEIGHT, ScoreTracker  # noqa: E402

STAND_STILL = np.zeros(9, dtype=np.int8)


def load_catalog() -> dict[str, dict]:
    path = REPO / "maps" / "catalog.json"
    if not path.is_file():
        return {}
    return {m["id"]: m for m in json.loads(path.read_text(encoding="utf-8"))["maps"]}


def valid_action(a) -> bool:
    arr = np.asarray(a)
    return arr.shape == (9,) and arr.dtype.kind in "iub" and bool(np.isin(arr, (0, 1)).all())


def play(agent, spec: str, meta: dict, seed: int, render_buffers: bool) -> dict:
    wad, map_name = parse_map(spec)
    ticks = int(meta.get("episode_ticks", 2100))
    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=render_buffers)
    tracker = ScoreTracker()
    budget_s = limits.agent_budget_s(ticks)
    stopped, zeroed, agent_s, steps, slowest = None, None, 0.0, 0, 0.0
    try:
        obs, _ = env.reset(seed=seed)
        tracker.reset(env.privileged_state())
        t0 = time.perf_counter()
        agent.reset()
        if time.perf_counter() - t0 > limits.RESET_TIMEOUT_S:
            stopped = zeroed = "reset timeout"  # official rule: 0 points for this map
        done = False
        while not done:
            action = STAND_STILL
            if stopped is None:
                t0 = time.perf_counter()
                try:
                    proposed = agent.act(obs)
                except Exception as exc:  # noqa: BLE001
                    proposed, stopped = None, f"exception: {type(exc).__name__}"
                dt = time.perf_counter() - t0
                agent_s += dt
                slowest = max(slowest, dt)
                limit = limits.FIRST_ACT_TIMEOUT_S if steps == 0 else limits.ACT_TIMEOUT_S
                if stopped is None and dt > limit:
                    stopped = f"act() took {dt:.1f}s"
                elif stopped is None and agent_s > budget_s:
                    stopped = "compute budget used up"
                elif stopped is None and not valid_action(proposed):
                    stopped = "invalid action"
                if stopped is None:
                    action = np.asarray(proposed, dtype=np.int8)
            obs, _, terminated, truncated, _ = env.step(action)
            tracker.update(env.privileged_state())
            steps += 1
            done = terminated or truncated
    finally:
        env.close()
    res = tracker.result()
    if zeroed:
        res.buckets = {k: 0.0 for k in res.buckets}
        res.score = 0.0
    tier = int(meta.get("tier", 1))
    return {
        "map": spec,
        "tier": tier,
        "seed": seed,
        "steps": steps,
        "stopped": stopped,
        "agent_mean_ms": round(1000 * agent_s / max(steps, 1), 3),
        "agent_max_ms": round(1000 * slowest, 2),
        "weighted_score": round(TIER_WEIGHT[tier] * res.score, 2),
        **res.to_dict(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--map", action="append", help="wad[:MAPxx] instead of a split, repeatable")
    ap.add_argument("--seeds", type=int, default=1, help="episodes per map (seeds 0..N-1)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--weights", type=Path, help="brain checkpoint to test (default: submission/weights/champion.pt)")
    ap.add_argument("--no-buffers", action="store_true", help="skip rgb/depth (only if the agent ignores them)")
    ap.add_argument("--name", default=None, help="result name (default: <split>-<timestamp>)")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--allow-repeat-test", action="store_true")
    args = ap.parse_args(argv)
    if args.seeds < 1: ap.error("seeds must be positive")

    catalog = load_catalog()
    if args.map:
        specs = args.map
    else:
        lst = REPO / "maps" / "lists" / f"{args.split}.txt"
        if not lst.is_file():
            print(f"{lst} not found: run scripts/build_map_catalog.py first")
            return 2
        specs = [ln.strip() for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip()]

    from scripts.test_use import reserve_test_use, finish_test_use, catalog_metadata
    resolved_metadata = catalog_metadata(specs)
    weights = args.weights or (REPO / 'submission/weights/champion.pt')
    token = reserve_test_use(specs, weights, final_test=args.final_test, allow_repeat=args.allow_repeat_test)
    rows = []
    try:
        t0 = time.perf_counter()
        from scripts.phase4_ablation import strict_agent
        agent, _, _ = strict_agent(weights,device=args.device)
        init_s = agent.scored_init_s
        if hasattr(agent.policy, 'net') and agent.policy.net.cfg.uses_pixels and args.no_buffers:
            raise ValueError('--no-buffers is incompatible with this pixel checkpoint')
        print(f"Policy: {agent.policy_name}")
        if init_s > limits.INIT_TIMEOUT_S:  # official rule: 0 points on every map
            print(f"WARNING: __init__ took {init_s:.1f}s > {limits.INIT_TIMEOUT_S:.0f}s: the scorer would give 0 points")
        rows = []
        for spec in specs:
            for seed in range(args.seeds):
                row = play(agent, spec, resolved_metadata.get(spec, catalog.get(spec, {})), seed, not args.no_buffers)
                rows.append(row)
                print(
                    f"T{row['tier']} {row['map']:<44} s{seed} score {row['score']:7.2f} (x{TIER_WEIGHT[row['tier']]:.0f} = "
                    f"{row['weighted_score']:7.2f})  exit={int(row['exit'])} keys={row['keys']} doors={row['doors']}"
                    f"+{row['locked_doors']}L areas={row['areas']} kills={row['kills']} dmg={row['damage']:.0f}"
                    f" dead={int(row['dead'])} {row['agent_mean_ms']:.2f}ms" + (f"  STOPPED: {row['stopped']}" if row["stopped"] else ""),
                    flush=True,
                )
    finally:
        finish_test_use(token, status='completed' if len(rows) == len(specs)*args.seeds and not any(r.get('stopped') for r in rows) else 'partial')

    total = round(sum(r["weighted_score"] for r in rows) / max(args.seeds, 1), 2)
    by_tier = {t: round(sum(r["weighted_score"] for r in rows if r["tier"] == t) / max(args.seeds, 1), 2) for t in (1, 2, 3, 4)}
    buckets = {k: round(sum(r["buckets"][k] for r in rows) / max(args.seeds, 1), 2) for k in rows[0]["buckets"]} if rows else {}
    print(f"\nWeighted total: {total}   by tier: {by_tier}   buckets (unweighted): {buckets}")
    print(f"Crashes/stops: {sum(1 for r in rows if r['stopped'])} of {len(rows)} runs")

    name = args.name or f"{'custom' if args.map else args.split}-{datetime.now():%Y%m%d-%H%M%S}"
    out = REPO / "runs" / "eval" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # What this result covers, so totals are only ever compared on the same maps (scripts/league_record.py).
    weights = args.weights or (REPO / "submission" / "weights" / "champion.pt")
    entries = map_set_of(rows)
    record = {"total": total, "by_tier": by_tier, "buckets": buckets, "policy": agent.policy_name,
              "weights_sha256": file_sha256(weights) if agent.policy_name.startswith("brain") else None,
              "map_set_fingerprint": fingerprint(entries), "map_set": entries, "runs": rows}
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out.relative_to(REPO)}  (map set {record['map_set_fingerprint']}: {len(entries)} map x seed runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
