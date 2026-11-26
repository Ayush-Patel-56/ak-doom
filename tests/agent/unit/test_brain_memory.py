"""Phase 3: memory heads in SmritiNet, checkpoint compatibility and the agent's memory wiring."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from conftest import load_agent_class
from smriti.brain import BrainConfig, SmritiNet, checkpoint_dict, load_checkpoint
from smriti.features import FEATURE_SPEC, empty
from training.checkpoints import save_checkpoint


def batch(t=2, b=3, seed=0):
    rng = np.random.default_rng(seed)
    x = empty((t, b))
    x["ray_dist"][:] = rng.random(x["ray_dist"].shape)
    x["map"][:] = rng.integers(0, 256, x["map"].shape)
    x["notes"][:] = rng.normal(size=x["notes"].shape)
    x["note_mask"][..., :5] = True
    return {k: torch.as_tensor(v) for k, v in x.items()}


@pytest.mark.parametrize("use_map,use_notes", [(True, True), (True, False), (False, True), (False, False)])
def test_shapes_for_every_memory_combination(use_map, use_notes):
    net = SmritiNet(BrainConfig(use_mindmap=use_map, use_notebook=use_notes)).eval()
    logits, values, state = net(batch(), net.initial_state(3), torch.zeros(2, 3, dtype=torch.bool))
    assert logits.shape == (2, 3, 15) and values.shape == (2, 3) and state.shape == (3, 256)
    assert torch.isfinite(logits).all()


def test_memory_off_ignores_memory_inputs_and_memory_on_uses_them():
    torch.manual_seed(0)
    off = SmritiNet(BrainConfig()).eval()
    on = SmritiNet(BrainConfig(use_mindmap=True, use_notebook=True)).eval()
    x1, x2 = batch(seed=1), batch(seed=1)
    x2["map"] = torch.randint(0, 256, x2["map"].shape, dtype=torch.uint8)
    x2["notes"] = torch.randn_like(x2["notes"])
    h = torch.zeros(3, 256)
    with torch.no_grad():
        assert torch.equal(off(x1, h)[0], off(x2, h)[0])
        assert not torch.allclose(on(x1, h)[1], on(x2, h)[1])


def test_empty_notebook_is_well_defined_and_reader_depends_on_state():
    net = SmritiNet(BrainConfig(use_notebook=True)).eval()
    x = batch(t=1, b=2)
    x["note_mask"][:] = False
    with torch.no_grad():
        logits, _, _ = net(x, net.initial_state(2))
    assert torch.isfinite(logits).all()
    r = net.notebook
    k, v, valid = r.keys_values(torch.randn(2, 16, 9), torch.ones(2, 16, dtype=torch.bool))
    with torch.no_grad():
        a = r.read(torch.zeros(2, 256), k, v, valid)
        b = r.read(torch.randn(2, 256) * 3, k, v, valid)
    assert not torch.allclose(a, b)


def test_phase2_checkpoint_without_memory_keys_still_loads(tmp_path):
    net = SmritiNet(BrainConfig())
    ckpt = checkpoint_dict(net)
    for key in ("use_mindmap", "use_notebook", "map_out", "note_dim"):
        ckpt["config"].pop(key)  # exactly what a Phase 2 (PR #5) checkpoint stores
    path = tmp_path / "phase2.pt"
    torch.save(ckpt, path)
    loaded, _ = load_checkpoint(path)
    assert not loaded.cfg.uses_memory
    for a, b in zip(net.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


@pytest.fixture()
def memory_weights(tmp_path):
    torch.manual_seed(0)
    return save_checkpoint(tmp_path / "mem.pt", SmritiNet(BrainConfig(use_mindmap=True, use_notebook=True)))


def test_agent_with_memory_checkpoint_acts_validly_and_fast(memory_weights, spaces):
    from doom_env import make_env
    from pathlib import Path

    torch.set_num_threads(1)
    agent = load_agent_class()(*spaces, "cpu", weights_path=memory_weights)
    assert agent.policy_name.endswith("+memory")
    assert agent.memory.mindmap is not None and agent.memory.notebook is not None
    repo = Path(__file__).resolve().parents[3]
    env = make_env(str(repo / "maps" / "curated" / "val" / "t3-keynear-100000.wad"), render_buffers=False,
                   episode_ticks=1000)
    try:
        obs, _ = env.reset(seed=0)
        agent.reset()
        times = []
        for _ in range(120):
            t0 = time.perf_counter()
            a = agent.act(obs)
            times.append(time.perf_counter() - t0)
            assert a.shape == (9,) and a.dtype == np.int8 and set(np.unique(a)) <= {0, 1}
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        assert agent.guard.fallbacks == 0
        assert agent.memory.mindmap.cells_visited >= 1
        assert np.mean(times[5:]) * 1000 < 20, f"{np.mean(times[5:]) * 1000:.2f} ms per step"
        agent.reset()
        assert agent.memory.mindmap.grid.sum() == 0 and agent.memory.notebook.kind.sum() == 0
    finally:
        env.close()


def test_agent_without_memory_checkpoint_does_not_draw(tmp_path, spaces):
    path = save_checkpoint(tmp_path / "nomem.pt", SmritiNet(BrainConfig()))
    agent = load_agent_class()(*spaces, "cpu", weights_path=path)
    agent.reset()
    assert agent.policy_name == "brain:nomem.pt" and agent.memory.mindmap is None
    assert agent.act(spaces[0].sample()).shape == (9,)


@pytest.mark.parametrize("breakage", ["missing_heads", "unknown_key", "not_a_dict"])
def test_incompatible_checkpoints_fall_back_to_random(tmp_path, spaces, breakage):
    ckpt = checkpoint_dict(SmritiNet(BrainConfig()))
    if breakage == "missing_heads":  # config says memory, weights are Phase 2
        ckpt["config"]["use_mindmap"] = True
    elif breakage == "unknown_key":  # a checkpoint from a FUTURE design
        ckpt["config"]["use_manager"] = True
    else:
        ckpt = [1, 2, 3]
    path = tmp_path / "bad.pt"
    torch.save(ckpt, path)
    agent = load_agent_class()(*spaces, "cpu", weights_path=path)
    assert agent.policy_name == "random"
    agent.reset()
    assert agent.act(spaces[0].sample()).shape == (9,)


def test_memory_brain_survives_missing_calibration(memory_weights, spaces, monkeypatch):
    from smriti.memory import odometer as odmod

    def broken(cls, path=None):
        raise ValueError("calibration missing")

    monkeypatch.setattr(odmod.Calibration, "load", classmethod(broken))
    agent = load_agent_class()(*spaces, "cpu", weights_path=memory_weights)
    agent.reset()
    assert agent.policy_name.endswith("+memory") and agent.odometer is None
    for _ in range(5):
        assert agent.act(spaces[0].sample()).shape == (9,)
    assert agent.guard.fallbacks == 0


def test_feature_spec_has_memory_keys():
    assert FEATURE_SPEC["map"][0] == (8, 32, 32) and FEATURE_SPEC["map"][1] is np.uint8
    assert FEATURE_SPEC["notes"][0] == (16, 9)
