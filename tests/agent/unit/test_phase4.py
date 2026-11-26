"""Phase 4 behavior and gradients, using real game observations."""
import numpy as np
import pytest
import torch

from conftest import REPO
from doom_env import make_env
from smriti.brain import BrainConfig, SmritiNet, load_checkpoint
from smriti.features import extract
from smriti.hierarchy import GoalState, policy_step
from training.checkpoints import save_checkpoint


def real_features():
    env = make_env(str(REPO / 'maps/two_rooms.wad'))
    try:
        obs, _ = env.reset(seed=17)
        return extract(obs)
    finally:
        env.close()


def test_goal_is_held_for_eight_decisions_and_resets():
    torch.manual_seed(3)
    net = SmritiNet(BrainConfig(use_manager=True, use_rgb=True)).eval()
    x = {k: torch.as_tensor(v)[None] for k, v in real_features().items()}
    state = net.initial_state(1)
    goals = GoalState.initial(1, 'cpu')
    masks, picked = [], []
    for i in range(19):
        starts = torch.tensor([i in (0, 11)])
        _, _, state, goals, aux = policy_step(net, x, state, starts, goals)
        masks.append(bool(aux['boundary'][0]))
        picked.append(int(aux['goal'][0]))
    assert [i for i, yes in enumerate(masks) if yes] == [0, 8, 11]
    assert len(set(picked[:8])) == len(set(picked[8:11])) == len(set(picked[11:19])) == 1


def test_goal_replay_conditions_worker_and_backpropagates_rgb_and_goals():
    net = SmritiNet(BrainConfig(use_manager=True, use_rgb=True))
    x = {k: torch.as_tensor(v)[None, None].repeat(2, 1, *([1] * v.ndim))
         for k, v in real_features().items()}
    state = net.initial_state(1)
    first = net(x, state, goals=torch.zeros((2, 1), dtype=torch.long), return_aux=True)
    other = net(x, state, goals=torch.ones((2, 1), dtype=torch.long), return_aux=True)
    assert not torch.equal(first[0], other[0])
    loss = first[0].square().sum() + first[3]['exit_logits'].square().sum()
    loss.backward()
    assert sum(p.grad.abs().sum() for p in net.rgb.parameters() if p.grad is not None) > 0
    assert net.goal_embedding.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        net(x, state)  # a hierarchical worker cannot guess its selected goal


def test_legacy_checkpoint_outputs_are_unchanged(tmp_path):
    net = SmritiNet(BrainConfig()).eval()
    x = {k: torch.as_tensor(v)[None, None] for k, v in real_features().items()}
    original = net(x, net.initial_state(1))[0]
    path = save_checkpoint(tmp_path / 'legacy.pt', net)
    checkpoint = torch.load(path, weights_only=True)
    for key in ('use_manager', 'goal_interval', 'goal_dim', 'use_rgb', 'rgb_out'):
        checkpoint['config'].pop(key)
    torch.save(checkpoint, path)  # actual Phase 3 schema has no Phase 4 config fields
    loaded, _ = load_checkpoint(path)
    assert not loaded.cfg.use_manager and not loaded.cfg.use_rgb
    assert torch.equal(original, loaded(x, loaded.initial_state(1))[0])


def test_rollout_replay_matches_actor_with_asynchronous_resets_and_bootstrap():
    from smriti.brain import log_prob_and_entropy
    from training.ppo import PPO, PPOConfig
    net = SmritiNet(BrainConfig(use_manager=True, use_rgb=True)).eval()
    ppo = PPO(net, PPOConfig(), 2, 'cpu')
    one = real_features()
    feats = {k: np.stack([v, v]) for k, v in one.items()}
    state = net.initial_state(2)
    goals, choices, logps, resets = [], [], [], []
    for i in range(11):
        starts = np.array([i == 0, i in (0, 4)], bool)
        a, lp, _, state = ppo.act(feats, state, starts)
        goals.append(ppo.goals.goal.clone())
        choices.append(torch.as_tensor(a))
        logps.append(torch.as_tensor(lp))
        resets.append(torch.as_tensor(starts))
        ppo.intervals.step(np.zeros(2), np.zeros(2, bool))
    sequence = {k: torch.as_tensor(np.stack([v] * 11)) for k, v in feats.items()}
    logits, _, _ = net(sequence, net.initial_state(2), torch.stack(resets), goals=torch.stack(goals))
    replay_lp, _ = log_prob_and_entropy(logits, torch.stack(choices))
    torch.testing.assert_close(replay_lp, torch.stack(logps), atol=2e-6, rtol=1e-6)
    before_remaining = ppo.goals.remaining.clone()
    before_rng = torch.get_rng_state()
    ppo.bootstrap(feats, state, np.zeros(2, bool))
    assert torch.equal(before_remaining, ppo.goals.remaining)
    assert torch.equal(before_rng, torch.get_rng_state())


def test_exit_head_learns_real_positive_and_negative_views(tmp_path):
    from training.mapgen import GridMap, write_map
    from training.exit_visibility import ExitVisibility
    images, labels = [], []
    for angle in (0, 180):
        path = write_map(GridMap.parse('########\n#P..XX.#\n########', player_angle=angle),
                         tmp_path / f'view-{angle}.wad')
        env = make_env(str(path))
        try:
            labeler = ExitVisibility.from_env(env)
            obs, _ = env.reset(seed=51)
            target, known = labeler.label()
            assert known
            images.append(extract(obs)['rgb'])
            labels.append(target)
        finally:
            env.close()
    assert set(labels) == {0., 1.}
    torch.manual_seed(41)
    net = SmritiNet(BrainConfig(use_rgb=True))
    x = torch.as_tensor(np.stack(images)).float() / 255
    y = torch.tensor(labels)
    optimizer = torch.optim.Adam(list(net.rgb.parameters()) + list(net.exit_head.parameters()), lr=.003)
    def loss():
        return torch.nn.functional.binary_cross_entropy_with_logits(net.exit_head(net.rgb(x)).squeeze(-1), y)
    before = float(loss().detach())
    for _ in range(30):
        optimizer.zero_grad()
        error = loss()
        error.backward()
        optimizer.step()
    assert float(loss().detach()) < before * .5


def test_all_features_train_resume_and_checkpoint_plays(spaces):
    import json
    import shutil
    import uuid
    from training.train import main
    from conftest import load_agent_class
    name = 'phase4-test-' + uuid.uuid4().hex[:8]
    out = REPO / 'runs' / name
    args = ['--manager', 'on', '--rgb', 'on', '--rnd', 'on', '--goal-helpers', 'on',
            '--memory', 'off', '--workers', '1', '--games', '2', '--T', '5', '--epochs', '1',
            '--minibatches', '1', '--threads', '1', '--kinds', 'keynear', '--ticks', '100', '--name', name]
    torch.manual_seed(0)
    initial = SmritiNet(BrainConfig(use_manager=True, use_rgb=True)).state_dict()
    try:
        assert main(args + ['--steps', '40']) == 0
        assert main(args + ['--steps', '60', '--resume']) == 0
        rows = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
        assert rows[-1]['steps'] == 60
        assert sum(row['manager_intervals'] for row in rows) > 0
        assert sum(row['exit_labels_known'] for row in rows) > 0
        for row in rows:
            for key in ('pg', 'vf', 'rnd_loss', 'exit_aux_loss', 'reward_extrinsic', 'reward_helper', 'reward_intrinsic'):
                assert np.isfinite(row[key])
        state = torch.load(out / 'train_state.pt', weights_only=False)
        assert state['rnd']['observation_moments']['count'] > 0
        assert state['manager_opt']
        trained, _ = load_checkpoint(out / 'latest.pt')
        for key in ('manager.weight', 'goal_embedding.weight', 'rgb.0.weight', 'exit_head.weight'):
            assert not torch.equal(initial[key], trained.state_dict()[key]), key
        agent = load_agent_class()(*spaces, 'cpu', weights_path=out / 'latest.pt')
        assert agent.policy_name.startswith('brain:')
        env = make_env(str(REPO / 'maps/two_rooms.wad'))
        try:
            obs, _ = env.reset(seed=2)
            agent.reset()
            for _ in range(12):
                action = agent.act(obs)
                assert action.shape == (9,) and set(action) <= {0, 1}
                obs, _, _, _, _ = env.step(action)
            assert agent.policy.active_goal in range(5)
            assert 0 <= agent.policy.exit_probability <= 1
        finally:
            env.close()
    finally:
        shutil.rmtree(out, ignore_errors=True)
