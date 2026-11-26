"""Phase 3 ablation: memory (mind map + landmark notebook) ON vs OFF, everything else identical.

    # train both arms (in parallel) and evaluate them on held-out Tier 3 maps
    uv run python scripts/ablation.py run --steps 2000000 --name abl1
    # evaluate existing checkpoints only
    uv run python scripts/ablation.py eval --arm mem=runs/abl1-mem/latest.pt --arm nomem=runs/abl1-nomem/latest.pt

Both arms share: stage (3), seed, PPO settings, number of games and steps. The only difference is
BrainConfig.use_mindmap / use_notebook (training/train.py --memory on|off). The GRU stays in both.

Evaluation uses the Mini-League rules (scripts/evaluate.py play(): skill 3, fixed seeds, compute
budget, catalog tick caps) on the held-out Tier 3 maps of the chosen split: maps/curated/<split>/t3-*
(key & door puzzles generated from seeds that training never uses). Reported per arm: exit rate,
keys picked up, locked doors opened, doors opened, explored areas, score.
Results: runs/ablation/<name>.json and .md (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def heldout_maps(split: str) -> list[str]:
    folder = REPO / "maps" / "curated" / split
    return [f"maps/curated/{split}/{p.name}:MAP01" for p in sorted(folder.glob("t3-*.wad"))]


def evaluate_arm(weights: Path, split: str, seeds: int) -> dict:
    """Runs in its own process (one CPU thread, like the scorer)."""
    sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
    import agent_loader

    agent_loader.apply_eval_thread_limits()
    from doom_env import action_space, observation_space

    import evaluate as league

    catalog = league.load_catalog()
    agent = agent_loader.load_agent_class()(observation_space(), action_space(), "cpu", weights_path=weights)
    rows = []
    for spec in heldout_maps(split):
        meta = catalog.get(spec, {"tier": 3, "episode_ticks": 2100})
        for seed in range(seeds):
            rows.append(league.play(agent, spec, meta, seed, render_buffers=False))
    n = max(len(rows), 1)
    summary = {
        "policy": agent.policy_name,
        "episodes": len(rows),
        "exit_rate": round(sum(r["exit"] for r in rows) / n, 3),
        "key_rate": round(sum(r["keys"] > 0 for r in rows) / n, 3),
        "keys_per_ep": round(sum(r["keys"] for r in rows) / n, 3),
        "locked_door_rate": round(sum(r["locked_doors"] > 0 for r in rows) / n, 3),
        "locked_doors_per_ep": round(sum(r["locked_doors"] for r in rows) / n, 3),
        "doors_per_ep": round(sum(r["doors"] for r in rows) / n, 3),
        "areas_per_ep": round(sum(r["areas"] for r in rows) / n, 2),
        "death_rate": round(sum(r["dead"] for r in rows) / n, 3),
        "mean_score": round(sum(r["score"] for r in rows) / n, 2),
        "mean_act_ms": round(sum(r["agent_mean_ms"] for r in rows) / n, 3),
        "stopped": sorted({r["stopped"] for r in rows if r["stopped"]}),
    }
    return {"summary": summary, "rows": rows}


def table(results: dict) -> str:
    cols = ["episodes", "exit_rate", "key_rate", "locked_door_rate", "keys_per_ep", "locked_doors_per_ep",
            "doors_per_ep", "areas_per_ep", "death_rate", "mean_score", "mean_act_ms"]
    lines = ["| arm | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for arm, res in results.items():
        s = res["summary"]
        lines.append(f"| {arm} | " + " | ".join(str(s[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_eval_subprocess(arm: str, weights: Path, split: str, seeds: int, out: Path) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "_eval_one", "--weights", str(weights),
           "--split", split, "--seeds", str(seeds), "--out", str(out)]
    subprocess.run(cmd, check=True, cwd=REPO)
    return json.loads(out.read_text(encoding="utf-8"))


def write_report(name: str, results: dict, meta: dict) -> Path:
    folder = REPO / "runs" / "ablation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    md = f"# Ablation {name}\n\n```\n{json.dumps(meta, indent=2)}\n```\n\n{table(results)}\n"
    (folder / f"{name}.md").write_text(md, encoding="utf-8")
    print(md)
    return folder / f"{name}.md"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train both arms, then evaluate them")
    r.add_argument("--name", default=time.strftime("abl-%Y%m%d-%H%M%S"))
    r.add_argument("--steps", type=int, default=2_000_000)
    r.add_argument("--workers", type=int, default=3, help="game workers PER ARM (arms train in parallel)")
    r.add_argument("--games", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--split", default="val", choices=["val", "test"])
    r.add_argument("--seeds", type=int, default=3, help="episodes per held-out map")
    r.add_argument("--train-args", default="", help="extra args passed to training/train.py for BOTH arms")
    e = sub.add_parser("eval", help="evaluate existing checkpoints")
    e.add_argument("--arm", action="append", required=True, help="name=path/to/checkpoint.pt (repeatable)")
    e.add_argument("--name", default=time.strftime("abl-eval-%Y%m%d-%H%M%S"))
    e.add_argument("--split", default="val", choices=["val", "test"])
    e.add_argument("--seeds", type=int, default=3)
    one = sub.add_parser("_eval_one")
    one.add_argument("--weights", type=Path, required=True)
    one.add_argument("--split", default="val")
    one.add_argument("--seeds", type=int, default=3)
    one.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "_eval_one":
        a.out.write_text(json.dumps(evaluate_arm(a.weights, a.split, a.seeds)), encoding="utf-8")
        return 0

    tmp = REPO / "runs" / "ablation" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    if a.cmd == "eval":
        arms = dict(x.split("=", 1) for x in a.arm)
        results = {k: run_eval_subprocess(k, Path(v), a.split, a.seeds, tmp / f"{a.name}-{k}.json")
                   for k, v in arms.items()}
        write_report(a.name, results, {"split": a.split, "seeds": a.seeds, "arms": arms})
        return 0

    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    procs, t0 = {}, time.time()
    for arm, mode in (("mem", "on"), ("nomem", "off")):
        cmd = [sys.executable, "-m", "training.train", "--stage", "3", "--memory", mode,
               "--name", f"{a.name}-{arm}", "--steps", str(a.steps), "--workers", str(a.workers),
               "--games", str(a.games), "--seed", str(a.seed), *a.train_args.split()]
        log = (REPO / "runs" / f"{a.name}-{arm}.log").open("w", encoding="utf-8")
        print("$", " ".join(cmd), f"> runs/{a.name}-{arm}.log", flush=True)
        procs[arm] = (subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log)
    failed = []
    for arm, (p, log) in procs.items():
        if p.wait() != 0:
            failed.append(arm)
        log.close()
    if failed:
        print(f"training failed for {failed}; see runs/{a.name}-<arm>.log")
        return 1
    train_s = round(time.time() - t0)
    results = {arm: run_eval_subprocess(arm, REPO / "runs" / f"{a.name}-{arm}" / "latest.pt", a.split, a.seeds,
                                        tmp / f"{a.name}-{arm}.json") for arm in procs}
    meta = {"steps_per_arm": a.steps, "games_per_arm": a.workers * a.games, "seed": a.seed, "split": a.split,
            "seeds_per_map": a.seeds, "maps": heldout_maps(a.split), "train_seconds": train_s,
            "train_args": a.train_args}
    write_report(a.name, results, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
