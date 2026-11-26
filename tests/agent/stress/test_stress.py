"""Stress & chaos tests (docs/05-TESTING-PLAN.md §5.7): the agent must never stand still.

If the agent raises, returns an invalid action or is too slow, the scorer freezes the player for the
rest of the map. Chaos observations, a crowded view, and act-before-reset run against all four brain
types we ship or may ship: random fallback, Phase 2 champion (no memory), Phase 3 memory brain and a
full Phase 4 brain (memory + manager + RGB + depth). The back-to-back maps test covers memory and phase4.
Untrained weights are fine here: robustness, not skill, is tested.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from conftest import REPO, load_agent_class
from smriti.brain import BrainConfig, SmritiNet
from training.checkpoints import save_checkpoint
from training.mapgen import GridMap, write_map

BRAINS = ["random", "champion", "memory", "phase4"]


@pytest.fixture(scope="module")
def brain_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("brains")
    torch.manual_seed(0)
    memory = save_checkpoint(d / "memory.pt", SmritiNet(BrainConfig(use_mindmap=True, use_notebook=True)))
    phase4 = save_checkpoint(d / "phase4.pt", SmritiNet(BrainConfig(
        use_mindmap=True, use_notebook=True, use_manager=True, use_rgb=True, use_depth=True)))
    return {"random": d / "missing.pt", "champion": REPO / "submission" / "weights" / "champion.pt",
            "memory": memory, "phase4": phase4}


@pytest.fixture(params=BRAINS)
def agent(request, brain_paths, spaces):
    if request.param == "champion" and not brain_paths["champion"].is_file():
        pytest.skip("no champion.pt in this checkout")
    a = load_agent_class()(*spaces, "cpu", weights_path=brain_paths[request.param])
    expected = "random" if request.param == "random" else "brain"
    assert a.policy_name.startswith(expected), a.policy_name
    a.reset()
    return a


def valid(action) -> bool:
    arr = np.asarray(action)
    return arr.shape == (9,) and arr.dtype.kind in "iub" and bool(np.isin(arr, (0, 1)).all())


def chaos_observation(rng: np.random.Generator, obs_space) -> object:
    """A random mix of good, corrupted and nonsense observations."""
    kind = rng.integers(0, 8)
    good = obs_space.sample()
    if kind == 0:
        return good
    if kind == 1:  # NaN / inf everywhere floats live
        good["vitals"] = np.full(17, rng.choice([np.nan, np.inf, -np.inf]), np.float32)
        good["rays"]["distance"] = np.full(64, np.nan, np.float32)
        good["visible_entities"]["features"] = np.full((32, 5), np.inf, np.float32)
        return good
    if kind == 2:  # huge values (vitals[14:17] are velocity, not an odometer teleport)
        good["vitals"][14:17] = rng.choice([1e9, -1e9])
        good["rgb"] = np.full((120, 160, 3), 255, np.uint8)
        return good
    if kind == 3:  # wrong shapes
        good["rays"]["distance"] = np.ones(7, np.float32)
        good["depth"] = np.zeros((3, 3), np.uint8)
        return good
    if kind == 4:  # missing keys
        good.pop(rng.choice(list(good.keys())))
        return good
    if kind == 5:  # wrong types
        good["visible_entities"] = "not a dict"
        good["vitals"] = None
        return good
    if kind == 6:  # out-of-range ids
        good["rays"]["hit_type"] = np.full(64, 99, np.int64)
        good["visible_entities"]["entity_type"] = np.full(32, -5, np.int64)
        return good
    nonsense = [None, {}, "garbage", 42, [1, 2, 3]]
    return nonsense[int(rng.integers(0, len(nonsense)))]


def test_chaos_observations_never_break_the_agent(agent, spaces):
    obs_space = spaces[0]
    obs_space.seed(1)
    rng = np.random.default_rng(1)
    for i in range(600):
        action = agent.act(chaos_observation(rng, obs_space))
        assert valid(action), f"step {i}: invalid action {action!r}"
        if i % 150 == 0:
            agent.reset()


def test_crowded_view_is_valid_and_fast(agent, spaces):
    """32 visible enemies and every ray hitting an enemy."""
    obs = spaces[0].sample()
    obs["visible_entities"]["mask"] = np.ones(32, np.int8)
    obs["visible_entities"]["entity_type"] = np.full(32, 3, np.int64)
    obs["rays"]["hit_type"] = np.full(64, 4, np.int64)
    agent.act(obs)  # warm-up
    t0 = time.perf_counter()
    for _ in range(50):
        assert valid(agent.act(obs))
    assert (time.perf_counter() - t0) / 50 * 1000 < 40


def test_act_before_reset_and_double_reset(brain_paths, spaces):
    for name in BRAINS:
        if not brain_paths[name].exists() and name != "random":
            continue
        a = load_agent_class()(*spaces, "cpu", weights_path=brain_paths[name])
        assert valid(a.act(spaces[0].sample())), name  # the scorer always resets first, but be safe
        a.reset()
        a.reset()
        assert valid(a.act(spaces[0].sample())), name


def _play(agent, wad, ticks: int, seed: int = 0) -> tuple[int, float]:
    from doom_env import make_env

    env = make_env(str(wad), "MAP01", episode_ticks=ticks)
    try:
        obs, _ = env.reset(seed=seed)
        agent.reset()
        steps, spent = 0, 0.0
        done = False
        while not done:
            t0 = time.perf_counter()
            action = agent.act(obs)
            spent += time.perf_counter() - t0
            assert valid(action)
            obs, _, term, trunc, _ = env.step(action)
            steps += 1
            done = term or trunc
        return steps, 1000 * spent / steps
    finally:
        env.close()


@pytest.mark.parametrize("brain", ["memory", "phase4"])
def test_many_maps_in_a_row_stay_fast(brain, brain_paths, spaces, tmp_path):
    """Several full episodes on different maps: no slowdown, memory is reset between maps."""
    maps = {
        "closet": "#####\n#P..#\n#####",  # spawn in a tiny closet, no exit
        "empty": "##########\n#P.......#\n#........#\n##########",  # no enemies, no keys, no exit
        "crowd": "############\n#P.zzzz.sss#\n#..iiii.ddd#\n############",  # 14 monsters at once
    }
    agent = load_agent_class()(*spaces, "cpu", weights_path=brain_paths[brain])
    timings = []
    for name, grid in maps.items():
        wad = write_map(GridMap.parse(grid), tmp_path / f"{name}.wad")
        steps, mean_ms = _play(agent, wad, ticks=1200)
        assert steps > 0
        timings.append(mean_ms)
        assert agent.guard.fallbacks == 0, f"{name}: the brain raised and fell back to SAFE_ACTION"
    assert max(timings) < 40, timings
    assert timings[-1] < 3 * timings[0] + 5, f"slowdown across maps: {timings}"
