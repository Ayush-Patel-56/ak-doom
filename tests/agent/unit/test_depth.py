"""Optional depth behavior verified with real ViZDoom frames and checkpoint transfer."""
import numpy as np
import torch

from conftest import REPO
from doom_env import make_env
from smriti.brain import BrainConfig, SmritiNet, load_checkpoint
from smriti.features import extract
from training.checkpoints import save_checkpoint
from training.warmstart import warm_start_from


def observation():
    env = make_env(str(REPO / 'maps/two_rooms.wad'))
    try:
        return env.reset(seed=53)[0]
    finally:
        env.close()


def test_real_depth_is_extracted_without_inventing_metric_distances():
    obs = observation()
    features = extract(obs)
    assert features['depth'].shape == (1, 60, 80)
    assert features['depth'].dtype == np.uint8
    np.testing.assert_array_equal(features['depth'][0], obs['depth'][::2, ::2])
    assert features['depth'].max() > features['depth'].min()


def test_depth_encoder_receives_gradients_and_warm_start_preserves_old_inputs():
    torch.manual_seed(53)
    x = {k: torch.as_tensor(v)[None, None] for k, v in extract(observation()).items()}
    target = SmritiNet(BrainConfig(use_rgb=True, use_depth=True))
    _, values, _ = target(x, target.initial_state(1))
    values.sum().backward()
    assert sum(p.grad.abs().sum() for p in target.depth.parameters() if p.grad is not None) > 0
    source = SmritiNet(BrainConfig(use_rgb=True)).eval()
    warm_start_from(target, source)
    target.eval()
    with torch.no_grad():
        old = source(x, source.initial_state(1))
        new = target(x, target.initial_state(1))
    torch.testing.assert_close(old[0], new[0])
    torch.testing.assert_close(old[1], new[1])


def test_older_checkpoints_default_depth_off(tmp_path):
    net = SmritiNet(BrainConfig()).eval()
    path = save_checkpoint(tmp_path / 'older.pt', net)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint['config'].pop('use_depth')
    checkpoint['config'].pop('depth_out')
    torch.save(checkpoint, path)
    loaded, _ = load_checkpoint(path)
    assert not loaded.cfg.use_depth


def test_stage4_depth_training_and_resume_use_real_full_levels():
    import json
    import shutil
    import uuid
    from training.train import main
    from conftest import load_agent_class
    from doom_env import observation_space, action_space
    name = 'depth-full-level-' + uuid.uuid4().hex[:8]
    out = REPO / 'runs' / name
    args = ['--stage', '4', '--memory', 'off', '--depth', 'on', '--full-level-share', '1',
            '--workers', '1', '--games', '1', '--T', '8', '--epochs', '1', '--minibatches', '1',
            '--threads', '1', '--name', name]
    try:
        assert main(args + ['--steps', '16']) == 0
        assert main(args + ['--steps', '24', '--resume']) == 0
        manifest = json.loads((out / 'curriculum.json').read_text())
        assert manifest['full_level_share'] == 1
        agent = load_agent_class()(observation_space(), action_space(), 'cpu', weights_path=out / 'latest.pt')
        assert agent.policy.net.cfg.use_depth
        assert agent.act(observation()).shape == (9,)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_legacy_resume_cannot_silently_change_to_stage4():
    import shutil
    import uuid
    import pytest
    from training.train import main
    name = 'legacy-pool-check-' + uuid.uuid4().hex[:8]
    out = REPO / 'runs' / name
    args = ['--memory', 'off', '--workers', '1', '--games', '1', '--T', '8',
            '--epochs', '1', '--minibatches', '1', '--threads', '1', '--ticks', '40', '--name', name]
    try:
        assert main(args + ['--stage', '3', '--kinds', 'arena', '--steps', '8']) == 0
        state = torch.load(out / 'train_state.pt', weights_only=False)
        state.pop('curriculum_config')  # reproduce an actual older training-state schema
        torch.save(state, out / 'train_state.pt')
        with pytest.raises(ValueError, match='provenance'):
            main(args + ['--stage', '4', '--resume', '--steps', '16'])
    finally:
        shutil.rmtree(out, ignore_errors=True)
