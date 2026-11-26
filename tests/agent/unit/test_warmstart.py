"""Warm start: a memory brain built from a Phase 2 brain must start out playing EXACTLY like it."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from smriti.brain import BrainConfig, SmritiNet
from smriti.features import FEATURE_SPEC, empty
from training.warmstart import warm_start_from


def random_features(t: int, b: int, seed: int = 0) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = empty((t, b))
    for k, (shape, dtype) in FEATURE_SPEC.items():
        if dtype == np.bool_:
            x[k] = rng.random(x[k].shape) < 0.5
        elif np.issubdtype(dtype, np.integer):
            x[k] = rng.integers(0, 5, x[k].shape)
        else:
            x[k] = rng.normal(size=x[k].shape).astype(np.float32)
    if "note_mask" in x:
        x["note_mask"][..., 0] = True  # at least one remembered landmark, so the notebook read matters
    return {k: torch.as_tensor(v) for k, v in x.items()}


@pytest.mark.parametrize("use_map,use_notes", [(True, True), (True, False), (False, True)])
def test_warm_started_memory_brain_plays_exactly_like_the_source(use_map, use_notes):
    torch.manual_seed(0)
    src = SmritiNet(BrainConfig()).eval()
    for p in src.parameters():  # make the source clearly "trained" (non-trivial weights everywhere)
        p.data.normal_(0, 0.1)
    tgt = SmritiNet(BrainConfig(use_mindmap=use_map, use_notebook=use_notes)).eval()
    report = warm_start_from(tgt, src)
    assert report["widened"] == (["fuse.0.weight"] if use_map else []) + (["gru.weight_ih"] if use_notes else [])
    x = random_features(6, 3)
    starts = torch.zeros(6, 3, dtype=torch.bool)
    starts[0] = True
    with torch.no_grad():
        a = src(x, src.initial_state(3), starts)
        b = tgt(x, tgt.initial_state(3), starts)
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)


def test_memory_can_still_learn_after_warm_start():
    torch.manual_seed(1)
    src = SmritiNet(BrainConfig())
    tgt = SmritiNet(BrainConfig(use_mindmap=True, use_notebook=True))
    warm_start_from(tgt, src)
    x = random_features(4, 2, seed=1)
    logits, values, _ = tgt(x, tgt.initial_state(2))
    (logits.sum() + values.sum()).backward()
    hidden = tgt.cfg.hidden
    assert tgt.fuse[0].weight.grad[:, -tgt.cfg.map_out:].abs().sum() > 0  # mind-map columns get gradient
    assert tgt.gru.weight_ih.grad[:, hidden:].abs().sum() > 0  # notebook columns get gradient


def test_incompatible_brains_are_refused():
    with pytest.raises(ValueError):
        warm_start_from(SmritiNet(BrainConfig(use_mindmap=True)), SmritiNet(BrainConfig(hidden=128)))


def test_rgb_inputs_stay_aligned_when_adding_memory():
    torch.manual_seed(71)
    source = SmritiNet(BrainConfig(use_rgb=True)).eval()
    target = SmritiNet(BrainConfig(use_rgb=True, use_mindmap=True, use_notebook=True)).eval()
    warm_start_from(target, source)
    x = random_features(3, 2, seed=71)
    with torch.no_grad():
        before = source(x, source.initial_state(2))
        after = target(x, target.initial_state(2))
    torch.testing.assert_close(before[0], after[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(before[1], after[1], atol=1e-6, rtol=1e-5)


def test_phase4_cli_preserves_upstream_path_based_warm_start():
    from pathlib import Path
    from training.train import build_args
    args = build_args(['--warm-start', 'source.pt', '--manager', 'on', '--rgb', 'on', '--puzzle-share', '.6'])
    assert args.warm_start == Path('source.pt')
    assert args.manager == args.rgb == 'on' and args.puzzle_share == .6


def test_puzzle_override_reaches_phase4_worker_processes():
    from training.rollout import VecGames
    games = VecGames(1, 1, 3, 1, False, False, episode_ticks=40,
                     switch_every=1, puzzle_share=0., use_rgb=True)
    try:
        completed = []
        for _ in range(35):
            _, _, _, infos = games.step(np.zeros((1, 6), np.int64), 0.)
            completed.extend(info for info in infos if info is not None)
        assert len(completed) >= 3
        assert all(info['kind'] in ('arena', 'maze', 'rooms') for info in completed)
    finally:
        games.close()


def test_champion_warm_start_trains_with_phase4_flags():
    import shutil
    import uuid
    from conftest import REPO
    from training.train import main
    from smriti.brain import load_checkpoint
    champion = REPO / 'submission/weights/champion.pt'
    source, _ = load_checkpoint(champion)
    # RGB is always new here, so fuse is always widened; the GRU input only widens when the
    # champion has no notebook yet (champion #1 was a Phase 2 brain, champion #2 already has memory).
    expected = {'fuse.0.weight'} | (set() if source.cfg.use_notebook else {'gru.weight_ih'})
    name = 'merge-warmstart-' + uuid.uuid4().hex[:8]
    out = REPO / 'runs' / name
    try:
        assert main(['--warm-start', str(champion),
                     '--memory', 'on', '--manager', 'on', '--rgb', 'on', '--puzzle-share', '0',
                     '--workers', '1', '--games', '1', '--T', '8', '--steps', '32',
                     '--epochs', '1', '--minibatches', '1', '--threads', '1', '--ticks', '40',
                     '--name', name]) == 0
        net, metadata = load_checkpoint(out / 'latest.pt')
        assert net.cfg.use_manager and net.cfg.use_rgb and net.cfg.uses_memory
        assert set(metadata['warm_start']['widened']) == expected
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_puzzle_share_override_changes_the_mix():
    from training.curriculum import MapSampler

    default = MapSampler(3, seed=0)
    more_replay = MapSampler(3, seed=0, puzzle_share=0.6)
    assert default.stage.puzzle_share == 0.8 and more_replay.stage.puzzle_share == 0.6
    kinds = [more_replay.pick_kind() for _ in range(2000)]
    puzzles = sum(k in ("keynear", "keydoor") for k in kinds) / len(kinds)
    assert 0.55 < puzzles < 0.65
