"""Curiosity uses only real, public engine observation features."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def observations():
    from doom_env import make_env
    from smriti.features import extract

    env = make_env(str(Path(__file__).resolve().parents[3] / 'maps/two_rooms.wad'),
                   render_buffers=False, episode_ticks=1000)
    try:
        obs, _ = env.reset(seed=3)
        rows = []
        for _ in range(8):
            x = extract(obs)
            rows.append(np.concatenate([x['ray_dist'], x['vitals']]))
            obs, _, terminated, truncated, _ = env.step(np.zeros(9, np.int8))
            assert not (terminated or truncated)
        return np.stack(rows)
    finally:
        env.close()


def test_predictor_learns_stationary_engine_observations(observations):
    from training.rnd import RND

    rnd = RND(observations.shape[1], seed=41)
    before = rnd.bonus(observations, update_stats=False)
    target = deepcopy(rnd.target.state_dict())
    losses = [rnd.update(observations) for _ in range(80)]
    after = rnd.bonus(observations, update_stats=False)
    assert losses[-1] < losses[0] * .1
    assert after.mean() < before.mean() * .1
    assert all(not p.requires_grad for p in rnd.target.parameters())
    for key, value in target.items():
        torch.testing.assert_close(value, rnd.target.state_dict()[key], rtol=0, atol=0)


def test_checkpoint_restores_bonus_optimizer_and_past_moments(observations):
    from training.rnd import RND

    rnd = RND(observations.shape[1], seed=2, clip=.7)
    rnd.bonus(observations)
    for _ in range(3):
        rnd.update(observations)
    saved = deepcopy(rnd.state_dict())
    restored = RND(observations.shape[1], seed=99)
    restored.load_state_dict(saved)
    expected = rnd.bonus(observations, update_stats=False)
    np.testing.assert_array_equal(expected, restored.bonus(observations, update_stats=False))
    assert np.all((expected >= 0) & (expected <= .7))
    assert rnd.update(observations) == restored.update(observations)
    for a, b in zip(rnd.predictor.parameters(), restored.predictor.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_bonus_uses_past_statistics_and_eval_never_mutates(observations):
    from training.rnd import RND

    torch.manual_seed(14)
    rng = torch.random.get_rng_state().clone()
    rnd = RND(observations.shape[1], seed=2)
    assert torch.equal(rng, torch.random.get_rng_state())
    expected = rnd.bonus(observations, update_stats=False)
    np.testing.assert_array_equal(expected, rnd.bonus(observations, update_stats=True))
    saved = deepcopy(rnd.state_dict())
    for _ in range(3):
        rnd.bonus(observations, update_stats=False)
    actual = rnd.state_dict()
    for name in ('observation_moments', 'reward_moments'):
        assert actual[name]['count'] == saved[name]['count']
        np.testing.assert_array_equal(actual[name]['mean'], saved[name]['mean'])
        np.testing.assert_array_equal(actual[name]['m2'], saved[name]['m2'])
