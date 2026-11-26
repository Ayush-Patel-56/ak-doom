"""Phase 3 training pieces: Stage 3 curriculum, rewards (scorer deltas), PPO smoke run, ablation maps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


def test_stage3_sampler_uses_training_seeds_and_moves_from_near_to_far(tmp_path, monkeypatch):
    from training import curriculum

    monkeypatch.setattr(curriculum, "GENERATED", tmp_path)
    s = curriculum.MapSampler(3, seed=0, seeds_per_kind=5)
    early = [s.pick_kind() for _ in range(400)]
    s.progress = 1.0
    late = [s.pick_kind() for _ in range(400)]
    assert early.count("keynear") > late.count("keynear") > 0
    assert late.count("keydoor") > early.count("keydoor")
    assert sum(k in ("keynear", "keydoor") for k in early) / 400 > 0.6  # mostly puzzles
    wad, m, kind = next(s)
    assert Path(wad).parent == tmp_path and int(Path(wad).stem.split("-")[-1]) < 100_000 and m == "MAP01"
    with pytest.raises(ValueError):
        curriculum.map_path("keydoor", 100_000)  # val seeds are held out


def test_keynear_puts_the_key_near_the_start_and_needs_it():
    from training.mapgen import generate, solvable
    from training.mapgen.layouts import bfs, Canvas

    for seed in range(20):
        g = generate("keynear", seed)
        with_keys, without = solvable(g)
        assert with_keys and not without
        cv = Canvas(g.height, g.width)
        cv.g = [list(r) for r in g.rows]
        start = next(iter(g.cells("P")))
        dist = bfs(cv, start)
        key = next(iter(g.cells("rby")))
        assert dist[key] <= 20, f"seed {seed}: key {dist[key]} cells away"


def state(**kw):
    base = {"tic": 0, "episode_ticks": 2100, "x": 0.0, "y": 0.0, "z": 0.0, "angle": 0.0, "health": 100.0,
            "kills": 0, "items": 0, "damage_taken": 0.0, "keys": np.zeros(6), "dead": False, "timeout": False,
            "exit_reached": False, "finished": False, "sector_floor": np.zeros(2), "sector_ceiling": np.zeros(2),
            "door_lock": np.array([0, 1])}
    base.update(kw)
    return base


def test_rewards_follow_scorer_events():
    from training.rewards import WEIGHTS, RewardShaper

    r = RewardShaper()
    r.reset(state())
    noop = np.zeros(9, np.int8)
    assert r(state(tic=4), noop) == pytest.approx(WEIGHTS["step"])
    key = r(state(tic=8, keys=np.array([1, 0, 0, 0, 0, 0])), noop)
    assert key == pytest.approx(WEIGHTS["step"] + WEIGHTS["key"])
    locked = r(state(tic=12, keys=np.array([1, 0, 0, 0, 0, 0]), sector_ceiling=np.array([0.0, 64.0])), noop)
    assert locked == pytest.approx(WEIGHTS["step"] + WEIGHTS["locked_door"])
    moved = r(state(tic=16, x=300.0, keys=np.array([1, 0, 0, 0, 0, 0]), sector_ceiling=np.array([0.0, 64.0])), noop)
    assert moved == pytest.approx(WEIGHTS["step"] + WEIGHTS["area"])
    done = r(state(tic=1050, x=300.0, keys=np.array([1, 0, 0, 0, 0, 0]), sector_ceiling=np.array([0.0, 64.0]),
                   exit_reached=True, finished=True), noop)
    assert done > WEIGHTS["exit"]
    assert r.stats()["keys"] == 1 and r.stats()["locked_doors"] == 1 and r.stats()["exit"] == 1


def test_use_spam_and_death_are_penalised():
    from training.rewards import WEIGHTS, RewardShaper

    r = RewardShaper()
    r.reset(state())
    use = np.zeros(9, np.int8)
    use[7] = 1
    r(state(tic=4), use)
    assert r(state(tic=8), use) == pytest.approx(WEIGHTS["step"] + WEIGHTS["use_spam"])
    dead = r(state(tic=12, dead=True, finished=True, health=0.0), np.zeros(9, np.int8))
    assert dead < WEIGHTS["death"] * 0.5


@pytest.mark.parametrize("memory", ["on", "off"])
def test_train_smoke_runs_and_checkpoint_plays(tmp_path, memory, monkeypatch, spaces):
    """A real (tiny) PPO run on Stage 3 maps, then the checkpoint loads in DoomAgent."""
    from training import train

    name = f"pytest-smoke-{memory}"
    rc = train.main(["--stage", "3", "--memory", memory, "--workers", "1", "--games", "2", "--steps", "64",
                     "--T", "32", "--name", name, "--threads", "1", "--kinds", "keynear"])
    assert rc == 0
    run = REPO / "runs" / name
    try:
        rows = [json.loads(ln) for ln in (run / "metrics.jsonl").read_text().splitlines()]
        assert rows and all(np.isfinite(v) for v in (rows[-1]["pg"], rows[-1]["vf"]))
        from conftest import load_agent_class

        agent = load_agent_class()(*spaces, "cpu", weights_path=run / "latest.pt")
        assert agent.policy_name.startswith("brain") and agent.policy_name.endswith("+memory") == (memory == "on")
    finally:
        import shutil

        shutil.rmtree(run, ignore_errors=True)


def test_ablation_heldout_maps_are_tier3_and_never_training_seeds():
    sys.path.insert(0, str(REPO / "scripts"))
    import ablation

    for split, lo in (("val", 100_000), ("test", 200_000)):
        maps = ablation.heldout_maps(split)
        assert len(maps) >= 8
        for spec in maps:
            name = Path(spec.split(":")[0]).stem
            assert name.startswith("t3-") and lo <= int(name.split("-")[-1]) < lo + 100_000
