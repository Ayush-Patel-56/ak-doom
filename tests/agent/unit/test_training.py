"""Unit tests for the training factory: rewards, GAE, curriculum sampling, env pool and a PPO update."""

from __future__ import annotations

import random

import numpy as np
import pytest

from training.curriculum import STAGES, TRAIN_SEED_LIMIT, MapSampler
from training.ppo import PPOConfig, PPOTrainer, compute_gae
from training.rewards import RewardCalculator, RewardConfig


def state(**kw) -> dict:
    s = {"tic": 0, "episode_ticks": 2100, "x": 32.0, "y": 32.0, "health": 100.0, "kills": 0, "damage_taken": 0.0,
         "keys": np.zeros(6, np.float32), "dead": False, "exit_reached": False,
         "sector_floor": np.zeros(2), "sector_ceiling": np.array([128.0, 0.0]), "door_lock": np.array([-1, 0])}
    s.update(kw)
    return s


def obs(enemy=False, door_ahead=False) -> dict:
    hit = np.ones(64, np.int64)
    dist = np.ones(64, np.float32)
    if door_ahead:
        hit[0], dist[0] = 2, 0.05
    mask = np.zeros(32, np.int8)
    etype = np.zeros(32, np.int64)
    if enemy:
        mask[0], etype[0] = 1, 2
    return {"rays": {"hit_type": hit, "distance": dist},
            "visible_entities": {"mask": mask, "entity_type": etype, "features": np.zeros((32, 5), np.float32)}}


NOOP = np.zeros(9, np.int8)


def test_rewards_follow_the_scoring_events():
    c = RewardConfig()
    rc = RewardCalculator(c)
    rc.reset(state())
    r_idle = rc.step(state(tic=4), NOOP, obs())
    assert r_idle == pytest.approx(c.step + c.novelty / np.sqrt(2))  # same cell again: novelty shrinks, no new area
    r_move = rc.step(state(tic=8, x=200.0), NOOP, obs())
    assert r_move == pytest.approx(c.step + c.new_area + c.novelty)
    keys = np.array([0, 0, 1, 0, 0, 0], np.float32)
    r_key = rc.step(state(tic=12, x=200.0, keys=keys, kills=2, sector_ceiling=np.array([128.0, 72.0])), NOOP, obs())
    assert rc.totals["key"] == c.key and rc.totals["kill"] == 2 * c.kill and rc.totals["door"] == c.door
    assert r_key > 4
    r_exit = rc.step(state(tic=1050, x=200.0, keys=keys, kills=2, exit_reached=True,
                           sector_ceiling=np.array([128.0, 72.0])), NOOP, obs())
    assert r_exit > c.exit


def test_penalties_death_damage_stuck_and_button_spam():
    c = RewardConfig()
    rc = RewardCalculator(c)
    rc.reset(state())
    attack = NOOP.copy()
    attack[6] = 1
    rc.step(state(tic=4), attack, obs(enemy=False))
    assert rc.totals["shoot_nothing"] == c.shoot_nothing
    rc.step(state(tic=8), attack, obs(enemy=True))
    assert rc.totals["shoot_nothing"] == c.shoot_nothing  # enemy visible: no penalty
    use = NOOP.copy()
    use[7] = 1
    rc.step(state(tic=12), use, obs(door_ahead=True))
    assert "use_nothing" not in rc.totals
    for t in range(c.stuck_steps + 2):
        rc.step(state(tic=16 + 4 * t), NOOP, obs())
    assert rc.totals["stuck"] < 0
    rc.step(state(tic=200, damage_taken=30.0, dead=True), NOOP, obs())
    assert rc.totals["damage"] == pytest.approx(30 * c.damage_per_hp) and rc.totals["death"] == c.death


def test_gae_matches_hand_computation():
    r = np.array([[1.0], [0.0], [2.0]], np.float32)
    v = np.array([[0.5], [0.4], [0.3]], np.float32)
    d = np.array([[False], [True], [False]])
    adv, ret = compute_gae(r, v, d, np.array([1.0], np.float32), gamma=0.9, lam=0.5)
    d2 = 2.0 + 0.9 * 1.0 - 0.3
    d1 = 0.0 - 0.4  # episode ended after step 1: no bootstrap
    d0 = 1.0 + 0.9 * 0.4 - 0.5
    expect = np.array([d0 + 0.45 * d1, d1, d2])
    assert np.allclose(adv[:, 0], expect) and np.allclose(ret, adv + v)


def test_map_sampler_uses_training_seeds_and_is_deterministic(tmp_path):
    a = MapSampler(STAGES[3], random.Random(5), tmp_path / "a")
    b = MapSampler(STAGES[3], random.Random(5), tmp_path / "b")
    picks = [a.next() for _ in range(6)]
    assert [p[3] for p in picks] == [b.next()[3] for _ in range(6)]
    for wad, map_name, ticks, label in picks:
        assert 2100 <= ticks <= 4200 and map_name.startswith("MAP")
        if label in ("arena", "maze", "rooms", "keydoor"):
            assert int(wad.rsplit("-", 1)[1].split(".")[0]) < TRAIN_SEED_LIMIT


def test_env_pool_and_one_ppo_update(tmp_path):
    from training.envpool import EnvPool

    pool = EnvPool(2, stage=2, seed=0, episodes_per_map=1)
    try:
        cfg = PPOConfig(num_envs=2, rollout=32, seq_len=16, epochs=1, minibatch_seqs=2, total_steps=64, torch_threads=1)
        trainer = PPOTrainer(pool, cfg, tmp_path / "run", log=lambda m: None)
        batch = trainer.collect()
        assert batch["feats"]["ray_dist"].shape == (32, 2, 64) and batch["chunk_states"].shape == (2, 2, 256)
        assert batch["starts"][0].all()  # the first observation of each env starts an episode
        before = [p.detach().clone() for p in trainer.net.parameters()]
        stats = trainer.learn(batch)
        assert all(np.isfinite(v) for v in stats.values())
        assert any(not np.allclose(a, b.detach()) for a, b in zip(before, trainer.net.parameters()))
        path = trainer.save()
        from smriti.brain import load_checkpoint

        net, extra = load_checkpoint(path)
        assert extra["steps"] == 64
    finally:
        pool.close()
