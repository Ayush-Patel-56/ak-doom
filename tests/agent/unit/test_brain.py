"""Unit tests for the Phase 2 brain: action groups, features, network, checkpoints and BrainPolicy."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from smriti.brain import (
    BrainConfig,
    SmritiNet,
    load_checkpoint,
    log_prob_and_entropy,
    sample_actions,
)
from smriti.features import FEATURE_SPEC, empty, extract, stack
from smriti.policy.actions import GROUP_SIZES, NUM_LOGITS, buttons_to_groups, groups_to_buttons
from smriti.policy.brain_policy import BrainPolicy
from training.checkpoints import save_checkpoint

from conftest import load_agent_class


# ----------------------------------------------------------------------------- action groups
def test_every_group_choice_maps_to_buttons_and_back():
    import itertools

    for choice in itertools.product(*[range(n) for n in GROUP_SIZES]):
        buttons = groups_to_buttons(np.array(choice))
        assert buttons.shape == (9,) and set(np.unique(buttons)) <= {0, 1}
        assert not (buttons[0] and buttons[1]) and not (buttons[2] and buttons[3]) and not (buttons[4] and buttons[5])
        assert tuple(buttons_to_groups(buttons)) == choice


def test_contradictory_buttons_become_none():
    both = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0])
    assert tuple(buttons_to_groups(both)[:3]) == (0, 0, 0)


def test_groups_to_buttons_works_on_batches():
    choices = np.zeros((4, 7, 6), np.int64)
    choices[..., 0] = 1
    out = groups_to_buttons(choices)
    assert out.shape == (4, 7, 9) and out[..., 0].all() and not out[..., 1:].any()


# ----------------------------------------------------------------------------- features + network
def batch_from_space(obs_space, n: int, seed: int = 0) -> dict[str, np.ndarray]:
    obs_space.seed(seed)
    return stack([extract(obs_space.sample()) for _ in range(n)])


def to_torch(x: dict, t: int, b: int) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v.reshape(t, b, *v.shape[1:])) for k, v in x.items()}


def test_feature_shapes_match_spec(spaces):
    x = batch_from_space(spaces[0], 3)
    for k, (shape, dtype) in FEATURE_SPEC.items():
        assert x[k].shape == (3,) + shape and x[k].dtype == dtype
    e = empty((5, 2))
    assert e["ray_dist"].shape == (5, 2, 64)


def test_network_shapes_and_memory_reset(spaces):
    torch.manual_seed(0)
    net = SmritiNet()
    t, b = 6, 4
    x = to_torch(batch_from_space(spaces[0], t * b), t, b)
    logits, values, state = net(x, net.initial_state(b))
    assert logits.shape == (t, b, NUM_LOGITS) and values.shape == (t, b) and state.shape == (b, 256)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()

    # Wiping memory at step 3 must make steps 3+ independent of steps 0-2.
    starts = torch.zeros(t, b, dtype=torch.bool)
    starts[3] = True
    noisy = net.initial_state(b) + 5.0
    a, _, _ = net(x, noisy, starts)
    later = {k: v[3:] for k, v in x.items()}
    c, _, _ = net(later, net.initial_state(b))
    assert torch.allclose(a[3:], c, atol=1e-5)


def test_entity_encoder_handles_empty_view_and_ignores_slot_order(spaces):
    torch.manual_seed(0)
    net = SmritiNet().eval()
    x = batch_from_space(spaces[0], 1)
    x["ent_mask"][:] = False
    out = net.entities(torch.as_tensor(x["ent_feat"]), torch.as_tensor(x["ent_type"]), torch.as_tensor(x["ent_mask"]))
    assert torch.isfinite(out).all()

    x["ent_mask"][0, :5] = True
    perm = np.r_[4, 2, 0, 3, 1, 5:32]
    y = {k: (v[:, perm] if k.startswith("ent_") else v) for k, v in x.items()}
    enc = lambda d: net.entities(*(torch.as_tensor(d[k]) for k in ("ent_feat", "ent_type", "ent_mask")))  # noqa: E731
    assert torch.allclose(enc(x), enc(y), atol=1e-5)


def test_sampling_and_log_probs():
    torch.manual_seed(0)
    logits = torch.randn(10, NUM_LOGITS)
    choices = sample_actions(logits, torch.Generator().manual_seed(1))
    assert choices.shape == (10, 6)
    for g, n in enumerate(GROUP_SIZES):
        assert (choices[:, g] >= 0).all() and (choices[:, g] < n).all()
    logp, ent = log_prob_and_entropy(logits, choices)
    assert (logp <= 0).all() and (ent > 0).all()
    uniform_ent = log_prob_and_entropy(torch.zeros(1, NUM_LOGITS), choices[:1])[1]
    assert torch.allclose(uniform_ent, torch.tensor([sum(np.log(GROUP_SIZES))], dtype=torch.float32))


def test_new_network_starts_close_to_uniform_choices(spaces):
    net = SmritiNet()
    x = to_torch(batch_from_space(spaces[0], 8), 8, 1)
    logits, _, _ = net(x, net.initial_state(1))
    assert logits.abs().max() < 0.5


# ----------------------------------------------------------------------------- checkpoints + policy
@pytest.fixture()
def weights(tmp_path):
    torch.manual_seed(0)
    return save_checkpoint(tmp_path / "w.pt", SmritiNet(BrainConfig()), {"note": "test"})


def test_checkpoint_round_trip(weights, spaces):
    net, extra = load_checkpoint(weights)
    assert extra == {"note": "test"}
    again, _ = load_checkpoint(weights)
    x = to_torch(batch_from_space(spaces[0], 4), 4, 1)
    assert torch.equal(net(x, net.initial_state(1))[0], again(x, again.initial_state(1))[0])


def test_brain_policy_is_valid_and_reproducible(weights, spaces):
    obs_space = spaces[0]
    obs_space.seed(3)
    observations = [obs_space.sample() for _ in range(30)]
    pol = BrainPolicy(weights)
    first = [pol(o) for o in observations]
    pol.reset()
    second = [pol(o) for o in observations]
    for a in first:
        assert a.shape == (9,) and a.dtype == np.int8 and set(np.unique(a)) <= {0, 1}
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_agent_uses_brain_when_weights_exist(weights, spaces):
    obs_space, act_space = spaces
    agent = load_agent_class()(obs_space, act_space, "cpu", weights_path=weights)
    assert agent.policy_name.startswith("brain")
    agent.reset()
    for _ in range(20):
        a = agent.act(obs_space.sample())
        assert a.shape == (9,)
    assert agent.guard.fallbacks == 0


def test_agent_falls_back_to_random_without_weights(tmp_path, spaces):
    agent = load_agent_class()(*spaces, "cpu", weights_path=tmp_path / "missing.pt")
    assert agent.policy_name == "random"


def test_brain_step_is_fast_on_one_thread(weights, spaces):
    torch.set_num_threads(1)
    pol = BrainPolicy(weights)
    obs = spaces[0].sample()
    for _ in range(20):
        pol(obs)  # warm-up
    t0 = time.perf_counter()
    for _ in range(200):
        pol(obs)
    mean_ms = (time.perf_counter() - t0) / 200 * 1000
    assert mean_ms < 20, f"{mean_ms:.2f} ms per step"
