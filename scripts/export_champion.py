"""Promote a trained checkpoint to submission/weights/champion.pt (what the agent loads).

Safety checks before copying: the checkpoint loads, the agent built from it gives valid actions, and its
speed is inside our 20 ms target on one thread. With --eval, the Mini-League result must:
- come from THIS checkpoint (weights SHA-256 recorded by scripts/evaluate.py),
- have 0 crashes,
- beat the champion's score ON THE SAME MAP SET (same maps and seeds). Totals from different map sets
  (e.g. the 20-map val vs the 24-map val after the keynear maps were added) are never compared.

champion.json keeps the champion's score per map set ("results", keyed by fingerprint). If the champion
has no score on the candidate's map set, evaluate the champion on the same maps first and either pass
that file with --champion-eval, or store it with --rebaseline.

    # score the current champion on today's validation set (stored in champion.json)
    uv run python scripts/evaluate.py --split val --seeds 2 --no-buffers --name champion-val
    uv run python scripts/export_champion.py --rebaseline runs/eval/champion-val.json

    # promote a candidate evaluated on the same maps
    uv run python scripts/evaluate.py --split val --seeds 2 --no-buffers --weights runs/X/latest.pt --name X-val
    uv run python scripts/export_champion.py runs/X/latest.pt --eval runs/eval/X-val.json

Then commit submission/weights/champion.pt (Git LFS) and champion.json in a pull request.
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
from league_record import describe_result, file_sha256, load_result  # noqa: E402

REPO = agent_loader.REPO
CHAMPION = REPO / "submission" / "weights" / "champion.pt"
CHAMPION_INFO = REPO / "submission" / "weights" / "champion.json"


class Refused(Exception):
    pass


def summary(result: dict, path: Path) -> dict:
    crashes = sum(1 for r in result["runs"] if r.get("stopped"))
    return {"file": path.name, "total": result["total"], "by_tier": result["by_tier"], "crashes": crashes,
            **describe_result(result)}


def check_weights(result: dict, weights: Path, what: str) -> None:
    recorded = result.get("weights_sha256")
    if recorded is None:
        raise Refused(f"{what}: this Mini-League file does not record which weights it tested; "
                      "re-run scripts/evaluate.py (it now records them)")
    if recorded != file_sha256(weights):
        raise Refused(f"{what}: the Mini-League file was made with different weights than {weights}")


def champion_results(info: dict) -> dict[str, dict]:
    """Champion scores keyed by map-set fingerprint (old champion.json files have none: no fingerprint)."""
    return dict(info.get("results", {}))


def champion_score_on(fp: str, info: dict, champion_eval: Path | None) -> dict:
    if champion_eval is not None:
        ev = load_result(champion_eval)
        check_weights(ev, CHAMPION, "--champion-eval")
        s = summary(ev, champion_eval)
        if s["map_set_fingerprint"] != fp:
            raise Refused(f"--champion-eval covers map set {s['map_set_fingerprint']}, the candidate {fp}")
        return s
    known = champion_results(info)
    if fp not in known:
        have = ", ".join(f"{k} ({v['maps']} maps, {v['total']})" for k, v in known.items()) or "none"
        raise Refused(f"the champion has no score on map set {fp} (it has: {have}). Evaluate the champion on the "
                      "same maps and pass --champion-eval (or store it with --rebaseline)")
    return known[fp]


def rebaseline(path: Path) -> int:
    ev = load_result(path)
    check_weights(ev, CHAMPION, "--rebaseline")
    s = summary(ev, path)
    if s["crashes"]:
        raise Refused(f"{s['crashes']} stopped runs: not a usable baseline")
    info = json.loads(CHAMPION_INFO.read_text(encoding="utf-8")) if CHAMPION_INFO.is_file() else {}
    results = champion_results(info)
    results[s["map_set_fingerprint"]] = s
    info["results"] = results
    info["weights_sha256"] = file_sha256(CHAMPION)
    CHAMPION_INFO.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"champion.json: stored {s['total']} on map set {s['map_set_fingerprint']} ({s['maps']} maps, {s['runs']} runs)")
    return 0


def speed_and_validity(checkpoint: Path, target_ms: float):
    agent = agent_loader.load_agent_class()(observation_space(), action_space(), "cpu", weights_path=checkpoint)
    if not agent.policy_name.startswith("brain"):
        raise Refused(f"{checkpoint} did not load as a brain")
    obs_space = observation_space()
    obs_space.seed(0)
    agent.reset()
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        a = agent.act(obs_space.sample())
        times.append(time.perf_counter() - t0)
        if a.shape != (9,) or not np.isin(a, (0, 1)).all():
            raise Refused("invalid action")
    if agent.guard.fallbacks:
        raise Refused(f"{agent.guard.fallbacks} fallbacks to SAFE_ACTION: the brain raised errors")
    mean_ms = 1000 * float(np.mean(times[10:]))
    if mean_ms > target_ms:
        raise Refused(f"too slow: {mean_ms:.1f} ms > {target_ms} ms")
    return agent, mean_ms


def export(args) -> int:
    agent, mean_ms = speed_and_validity(args.checkpoint, args.target_ms)
    src = args.checkpoint.resolve()
    info = {"source": (src.relative_to(REPO) if src.is_relative_to(REPO) else src).as_posix(),
            "mean_ms": round(mean_ms, 2), "meta": agent.policy.meta, "weights_sha256": file_sha256(args.checkpoint)}
    if args.eval:
        ev = load_result(args.eval)
        check_weights(ev, args.checkpoint, "--eval")
        cand = summary(ev, args.eval)
        if cand["crashes"]:
            raise Refused(f"{cand['crashes']} stopped runs in the Mini-League")
        fp = cand["map_set_fingerprint"]
        if CHAMPION.is_file() and not args.force:
            old_info = json.loads(CHAMPION_INFO.read_text(encoding="utf-8")) if CHAMPION_INFO.is_file() else {}
            champ = champion_score_on(fp, old_info, args.champion_eval)
            if cand["total"] <= champ["total"]:
                raise Refused(f"Mini-League {cand['total']} is not better than the champion's {champ['total']} "
                              f"on the same {cand['maps']} maps (use --force to override)")
            print(f"candidate {cand['total']} > champion {champ['total']} on map set {fp}")
        info["results"] = {fp: cand}
    elif CHAMPION.is_file() and not args.force:
        raise Refused("pass --eval (a Mini-League result for this checkpoint) or --force")
    CHAMPION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.checkpoint, CHAMPION)
    CHAMPION_INFO.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"champion <- {args.checkpoint} ({mean_ms:.2f} ms/step)\n{json.dumps(info, indent=2)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, nargs="?", help="checkpoint to promote")
    ap.add_argument("--eval", type=Path, help="Mini-League JSON (scripts/evaluate.py output) for this checkpoint")
    ap.add_argument("--champion-eval", type=Path, help="Mini-League JSON of the CURRENT champion on the same maps")
    ap.add_argument("--rebaseline", type=Path, help="store this Mini-League result of the current champion in champion.json")
    ap.add_argument("--target-ms", type=float, default=20.0)
    ap.add_argument("--force", action="store_true", help="replace even without a better score on the same maps")
    args = ap.parse_args(argv)
    try:
        if args.rebaseline:
            return rebaseline(args.rebaseline)
        if args.checkpoint is None:
            ap.error("give a checkpoint to promote, or --rebaseline")
        return export(args)
    except Refused as exc:
        print(f"refusing: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
