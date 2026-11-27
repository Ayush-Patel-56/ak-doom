"""Record what the agent did in a run, step by step, and flag behaviour a replay reviewer would dislike.

The organisers watch the top teams' replays (docs/08-DEMO-AND-BRIEF.md §8.3): spinning, wall-hugging and
getting stuck are visible there even when the score looks fine. This tool plays a run exactly like
scripts/evaluate.py (same agent, map, seed, tick cap and buffers, so the same outcome) and records a
compact trace from the scorer's ground truth (true position and angle, kills, damage, keys, doors, exit,
death). That is evaluation-only data, never an agent input.

Flags are plain numeric rules (no LLM), so they cost nothing and are reproducible:
    stuck            barely moved for a while although not fighting
    spinning         turned more than 1.5 full circles almost on the spot
    wall_hugging     kept pressing forward with a wall right in front
    oscillating      bounced between the same few places
    passive_fire     took damage with enemies in view but never attacked
    no_new_area      explored no new area for a long time while alive

    uv run python scripts/episode_trace.py --map maps/curated/val/t3-keydoor-100003.wad:MAP01 --seed 1
    uv run python scripts/episode_trace.py --split val --seeds 2 --jobs 8 --name val-flags   # all val runs

Test maps are refused: their one-time use is recorded in the final-test ledger.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_spec import REPO, parse_map  # noqa: E402

sys.path.insert(0, str(REPO))
from doom_env import make_env, spaces  # noqa: E402

OUT_DIR = REPO / "runs" / "llm" / "traces"
BUTTON_SHORT = ("fwd", "back", "left", "right", "turnL", "turnR", "attack", "use", "weapon")
ENEMY_TYPES = {1, 2, 3}  # doom_env.spaces.ENTITY_TYPES: enemy_melee, enemy_hitscan, enemy_projectile
AREA = 128.0  # same unique-area cell as the scorer

# flag thresholds (one decision = 4 tics ~ 0.11 s)
STUCK_WINDOW, STUCK_DIST = 24, 32.0
SPIN_WINDOW, SPIN_DEG, SPIN_DIST = 16, 540.0, 64.0
HUG_RUN, HUG_DIST = 16, 40.0
OSC_WINDOW, OSC_CELL, OSC_CHANGES, OSC_UNIQUE = 40, 64.0, 8, 3
PASSIVE_WINDOW = 24
NO_AREA_STEPS = 150
WEAPON_SHARE = 0.5


def angle_diff(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


def trace_run(agent, spec: str, meta: dict, seed: int, render_buffers: bool = True) -> dict:
    """Play one run like evaluate.play() and record every decision."""
    from training.scorer import TIER_WEIGHT, ScoreTracker

    wad, map_name = parse_map(spec)
    ticks = int(meta.get("episode_ticks", 2100))
    env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=render_buffers)
    tracker = ScoreTracker()
    steps: list[dict] = []
    try:
        obs, _ = env.reset(seed=seed)
        tracker.reset(env.privileged_state())
        agent.reset()
        prev = env.privileged_state()
        prev_opened = 0
        done = False
        while not done:
            action = np.asarray(agent.act(obs), dtype=np.int8)
            ents = obs["visible_entities"]
            mask = np.asarray(ents["mask"]).astype(bool)
            types = np.asarray(ents["entity_type"]).astype(int)
            enemies = int(np.sum(mask & np.isin(types, list(ENEMY_TYPES))))
            front = float(np.asarray(obs["rays"]["distance"])[0]) * spaces.RAY_RANGE
            goal = getattr(agent, "active_goal", None)
            nb = getattr(agent.memory, "notebook", None)
            obs, _, terminated, truncated, _ = env.step(action)
            s = env.privileged_state()
            tracker.update(s)
            events = []
            if int(np.sum(np.asarray(s["keys"]) > 0)) > int(np.sum(np.asarray(prev["keys"]) > 0)):
                events.append("key")
            if int(s["kills"]) > int(prev["kills"]):
                events.append(f"kill x{int(s['kills']) - int(prev['kills'])}")
            dmg = float(s["damage_taken"]) - float(prev["damage_taken"])
            if dmg > 0:
                events.append(f"hit -{dmg:.0f}hp")
            if len(tracker.opened) > prev_opened:
                events.append("door opened")
                prev_opened = len(tracker.opened)
            if bool(s["exit_reached"]):
                events.append("EXIT")
            if bool(s["dead"]):
                events.append("DIED")
            steps.append(dict(
                t=len(steps), x=round(float(s["x"]), 1), y=round(float(s["y"]), 1), a=round(float(s["angle"]), 1),
                b=[BUTTON_SHORT[i] for i in np.flatnonzero(action)], hp=round(float(s["health"])),
                enemies=enemies, front=round(front), dmg=round(max(dmg, 0.0), 1), ev=events,
                goal=goal, notes=int(np.sum(nb.kind > 0)) if nb is not None else None))
            prev = s
            done = terminated or truncated
    finally:
        env.close()
    res = tracker.result()
    tier = int(meta.get("tier", 1))
    return dict(map=spec, seed=seed, tier=tier, kind=meta.get("kind") or kind_of(spec), episode_ticks=ticks,
                policy=getattr(agent, "policy_name", "?"), steps=steps,
                result=dict(res.to_dict(), weighted_score=round(TIER_WEIGHT[tier] * res.score, 2)))


def kind_of(spec: str) -> str:
    for k in ("arena", "maze", "rooms", "keynear", "keydoor"):
        if f"-{k}-" in spec:
            return k
    return "full level" if "freedoom" in spec.lower() else "map"


# ----------------------------------------------------------------------------- flags
def _segments(mask: list[bool], min_len: int = 1) -> list[tuple[int, int]]:
    out, start = [], None
    for i, m in enumerate(mask + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def behaviour_flags(steps: list[dict]) -> list[dict]:
    """Numeric rules over the trace -> [{kind, start, end, detail}] (merged, step indices inclusive)."""
    n = len(steps)
    if n == 0:
        return []
    xy = np.array([[s["x"], s["y"]] for s in steps], np.float64)
    ang = np.array([s["a"] for s in steps], np.float64)
    turn = np.abs(np.array([0.0] + [angle_diff(ang[i - 1], ang[i]) for i in range(1, n)]))
    fighting = np.array([s["enemies"] > 0 for s in steps])  # shooting at nothing is not fighting
    flags: list[dict] = []

    def add(kind: str, mask: list[bool], detail) -> None:
        for a, b in _segments(mask):
            flags.append(dict(kind=kind, start=a, end=b, detail=detail(a, b)))

    stuck = [False] * n
    spin = [False] * n
    for i in range(n):
        if i + STUCK_WINDOW <= n:
            w = slice(i, i + STUCK_WINDOW)
            if np.linalg.norm(xy[i + STUCK_WINDOW - 1] - xy[i]) < STUCK_DIST and np.ptp(xy[w], axis=0).max() < 2 * STUCK_DIST \
                    and fighting[w].sum() < STUCK_WINDOW // 4 and not any(s["ev"] for s in steps[w]):
                stuck[i:i + STUCK_WINDOW] = [True] * STUCK_WINDOW
        if i + SPIN_WINDOW <= n:
            w = slice(i, i + SPIN_WINDOW)
            if turn[w].sum() > SPIN_DEG and np.ptp(xy[w], axis=0).max() < SPIN_DIST:
                spin[i:i + SPIN_WINDOW] = [True] * SPIN_WINDOW
    add("stuck", stuck, lambda a, b: f"moved {np.linalg.norm(xy[b] - xy[a]):.0f} units in {b - a + 1} steps, no enemy or event")
    add("spinning", spin, lambda a, b: f"turned {turn[a:b + 1].sum():.0f} deg while staying within {np.ptp(xy[a:b + 1], axis=0).max():.0f} units")

    hug = [("fwd" in s["b"] and s["front"] < HUG_DIST) for s in steps]
    hug_runs = [(a, b) for a, b in _segments(hug) if b - a + 1 >= HUG_RUN]
    for a, b in hug_runs:
        flags.append(dict(kind="wall_hugging", start=a, end=b,
                          detail=f"pressed forward for {b - a + 1} steps with a wall < {HUG_DIST:.0f} units ahead"))

    cells = [(int(x // OSC_CELL), int(y // OSC_CELL)) for x, y in xy]
    osc = [False] * n
    for i in range(0, max(n - OSC_WINDOW, 0) + 1):
        w = cells[i:i + OSC_WINDOW]
        changes = sum(1 for j in range(1, len(w)) if w[j] != w[j - 1])
        if changes >= OSC_CHANGES and len(set(w)) <= OSC_UNIQUE:
            osc[i:i + OSC_WINDOW] = [True] * len(w)
    add("oscillating", osc, lambda a, b: f"switched between {len(set(cells[a:b + 1]))} small places for {b - a + 1} steps")

    passive = [False] * n
    for i in range(0, max(n - PASSIVE_WINDOW, 0) + 1):
        w = steps[i:i + PASSIVE_WINDOW]
        if sum(s["dmg"] for s in w) > 0 and any(s["enemies"] for s in w) and not any("attack" in s["b"] for s in w):
            passive[i:i + PASSIVE_WINDOW] = [True] * len(w)
    add("passive_fire", passive, lambda a, b: f"lost {sum(s['dmg'] for s in steps[a:b + 1]):.0f} hp with enemies in view, never attacked")

    weapon_share = sum("weapon" in s["b"] for s in steps) / n
    if n >= 20 and weapon_share >= WEAPON_SHARE:
        flags.append(dict(kind="weapon_cycling", start=0, end=n - 1,
                          detail=f"pressed next-weapon on {100 * weapon_share:.0f}% of decisions (keeps switching weapons)"))

    seen: set[tuple[int, int]] = set()
    last_new, gaps = 0, []
    for i, (x, y) in enumerate(xy):
        c = (int(x // AREA), int(y // AREA))
        if c not in seen:
            seen.add(c)
            if i - last_new >= NO_AREA_STEPS:
                gaps.append((last_new + 1, i - 1))
            last_new = i
    terminal = any(e in ("EXIT", "DIED") for e in steps[-1]["ev"])
    if not terminal and n - 1 - last_new >= NO_AREA_STEPS:
        gaps.append((last_new + 1, n - 1))
    for a, b in gaps:
        flags.append(dict(kind="no_new_area", start=a, end=b, detail=f"{b - a + 1} steps without reaching a new area"))
    return sorted(flags, key=lambda f: (f["start"], f["kind"]))


# ----------------------------------------------------------------------------- text summary
def summarize(trace: dict, flags: list[dict], segment: int = 25) -> str:
    """Compact, model-friendly text of one run: outcome, per-segment timeline, events and flags."""
    r, steps = trace["result"], trace["steps"]
    outcome = "reached the EXIT" if r["exit"] else ("DIED" if r["dead"] else "ran out of time")
    lines = [f"Map: {trace['map']} (tier {trace['tier']}, {trace['kind']}), seed {trace['seed']}, "
             f"tick cap {trace['episode_ticks']} ({trace['episode_ticks'] // 4} decisions).",
             f"Outcome: {outcome} after {len(steps)} decisions. keys {r['keys']}, doors {r['doors']}, locked doors "
             f"{r['locked_doors']}, unique areas {r['areas']}, kills {r['kills']}, damage taken {r['damage']:.0f}, "
             f"final health {r['health']:.0f}; map score {r['score']} (weighted {r['weighted_score']}).",
             "Button use over the whole run: " + ", ".join(
                 f"{b} {100 * sum(b in s['b'] for s in steps) // max(len(steps), 1)}%" for b in BUTTON_SHORT) + ".",
             "", f"Timeline in blocks of {segment} decisions (distance in map units; 32 units ~ 1 m):"]
    seen: set = set()
    for a in range(0, len(steps), segment):
        blk = steps[a:a + segment]
        b = a + len(blk) - 1
        path = sum(math.dist((blk[i - 1]["x"], blk[i - 1]["y"]), (blk[i]["x"], blk[i]["y"])) for i in range(1, len(blk)))
        net = math.dist((blk[0]["x"], blk[0]["y"]), (blk[-1]["x"], blk[-1]["y"]))
        turned = sum(abs(angle_diff(blk[i - 1]["a"], blk[i]["a"])) for i in range(1, len(blk)))
        new = 0
        for s in blk:
            c = (int(s["x"] // AREA), int(s["y"] // AREA))
            if c not in seen:
                seen.add(c)
                new += 1
        counts: dict[str, int] = {}
        for s in blk:
            for btn in s["b"]:
                counts[btn] = counts.get(btn, 0) + 1
        top = ", ".join(f"{k} {100 * v // len(blk)}%" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:4])
        ev = [f"{e}@{s['t']}" for s in blk for e in s["ev"]]
        enemy_steps = sum(1 for s in blk if s["enemies"])
        goal = {s["goal"] for s in blk if s["goal"] is not None}
        lines.append(f"- {a}-{b}: path {path:.0f}, net {net:.0f}, turned {turned:.0f} deg, new areas {new}, "
                     f"enemies in view {enemy_steps}/{len(blk)}, hp {blk[-1]['hp']}, buttons: {top or 'none'}"
                     + (f", manager goals {sorted(goal)}" if goal else "")
                     + (f", notebook {blk[-1]['notes']} landmarks" if blk[-1]["notes"] is not None else "")
                     + (f"; events: {', '.join(ev)}" if ev else ""))
    lines += ["", "Automatic behaviour flags (numeric rules):"]
    lines += [f"- {f['kind']} steps {f['start']}-{f['end']}: {f['detail']}" for f in flags] or ["- none"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- batch over a split
_WORKER: dict = {}


def _init_worker(weights: str) -> None:
    from scripts.phase4_ablation import strict_agent
    _WORKER["agent"] = strict_agent(Path(weights), device="cpu")[0]


def _job(job):
    spec, meta, seed = job
    trace = trace_run(_WORKER["agent"], spec, meta, seed)
    return trace, behaviour_flags(trace["steps"])


def resolve_jobs(split: str | None, maps: list[str], seeds: int) -> list[tuple[str, dict, int]]:
    from scripts.test_use import catalog_metadata
    if split:
        lst = REPO / "maps" / "lists" / f"{split}.txt"
        maps = [ln.strip() for ln in lst.read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta = catalog_metadata(maps)
    for spec in maps:
        m = meta.get(spec, {})
        if m.get("split") == "test" or "/curated/test/" in spec.replace("\\", "/"):
            raise SystemExit(f"refusing test map {spec}: test maps are used once, through evaluate.py --final-test")
    catalog = {}
    cat_path = REPO / "maps" / "catalog.json"
    if cat_path.is_file():
        catalog = {m["id"]: m for m in json.loads(cat_path.read_text(encoding="utf-8"))["maps"]}
    return [(spec, meta.get(spec, catalog.get(spec, {})), seed) for spec in maps for seed in range(seeds)]


def run_batch(jobs: list, weights: Path, n_jobs: int):
    if n_jobs <= 1:
        _init_worker(str(weights))
        for job in jobs:
            yield _job(job)
        return
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(processes=n_jobs, initializer=_init_worker, initargs=(str(weights),)) as pool:
        yield from pool.imap(_job, jobs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=REPO / "submission" / "weights" / "champion.pt")
    ap.add_argument("--split", choices=["train", "val"])
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--seed", type=int, help="single seed (default: seeds 0..--seeds-1)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--name", default="trace")
    a = ap.parse_args(argv)
    if not a.split and not a.map:
        ap.error("give --split or --map")
    jobs = resolve_jobs(a.split, a.map, 1 if a.seed is not None else a.seeds)
    if a.seed is not None:
        jobs = [(spec, meta, a.seed) for spec, meta, _ in jobs]
    out = OUT_DIR / a.name
    out.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}
    for trace, flags in run_batch(jobs, a.weights, min(a.jobs, len(jobs))):
        stem = f"{Path(parse_map(trace['map'])[0]).stem}-{parse_map(trace['map'])[1]}-s{trace['seed']}"
        (out / f"{stem}.json").write_text(json.dumps(dict(trace, flags=flags)), encoding="utf-8")
        (out / f"{stem}.txt").write_text(summarize(trace, flags), encoding="utf-8")
        for f in flags:
            totals[f["kind"]] = totals.get(f["kind"], 0) + 1
        r = trace["result"]
        print(f"{stem:42s} exit={int(r['exit'])} dead={int(r['dead'])} steps={len(trace['steps']):4d}  flags: "
              + (", ".join(f"{f['kind']}@{f['start']}-{f['end']}" for f in flags) or "none"), flush=True)
    print(f"flag totals: {totals or 'none'}  -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
