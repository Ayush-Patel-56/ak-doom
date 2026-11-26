"""Analytic interval returns and event-counter helper rewards."""
import numpy as np
import pytest
import torch


def aux(boundary=(True, True), values=(10., 20.)):
    return dict(hidden=torch.tensor([[1., 2.], [3., 4.]], requires_grad=True),
                boundary=torch.tensor(boundary), goal=torch.tensor([1, 3]),
                manager_logp=torch.tensor([-.2, -.3]), manager_value=torch.tensor(values))


def test_intervals_cross_rollout_and_bootstrap_only_at_boundaries():
    from training.manager import ManagerIntervals
    m = ManagerIntervals(2, .5)
    m.begin(aux())
    m.step(np.array([2., 4.]), np.array([False, False]))
    assert m.drain() == []  # pending interval survives a rollout drain
    m.begin(aux((False, False)))
    m.step(np.array([6., 8.]), np.array([False, True]))
    terminal = m.drain()
    assert len(terminal) == 1 and terminal[0]['target'] == 8.  # 4 + .5*8
    assert terminal[0]['duration'] == 2
    m.begin(aux((True, True), (12., 90.)))
    record = m.drain()[0]
    assert record['target'] == 8.  # 2 + .5*6 + .5**2*12
    assert record['duration'] == 2 and record['goal'] == 1
    assert record['old_value'] == 10.
    np.testing.assert_array_equal(record['hidden'], [1., 2.])
    m.step(np.array([7., 9.]), np.array([True, True]))
    assert [r['target'] for r in m.drain()] == [7., 9.]
    m.begin(aux())  # new episode must have no accumulated prior reward
    m.step(np.array([1., 3.]), np.array([True, True]))
    assert [r['target'] for r in m.drain()] == [1., 3.]


def test_manager_validates_shapes():
    from training.manager import ManagerIntervals
    m = ManagerIntervals(2, .9)
    bad = aux()
    bad['goal'] = torch.tensor([1])
    with pytest.raises(ValueError):
        m.begin(bad)
    with pytest.raises(ValueError):
        m.step(np.array([1.]), np.array([False, False]))


def state(**changes):
    s = dict(episode_ticks=1000, tic=0, health=100., dead=False, exit_reached=False,
             x=0., y=0., keys=np.zeros(6), kills=0, items=0, damage_taken=0.,
             door_lock=np.array([0, 1]), sector_floor=np.zeros(2), sector_ceiling=np.zeros(2))
    s.update(changes)
    return s


def test_goal_helpers_count_only_selected_actual_progress():
    from training.goal_rewards import GoalHelpers
    for goal, changed in enumerate([
        dict(x=256.), dict(kills=2), dict(items=1, keys=np.array([1, 0, 0, 0, 0, 0])),
        dict(sector_ceiling=np.array([40., 40.])), dict(exit_reached=True),
    ]):
        h = GoalHelpers(dict(explore=1., fight=1., collect=1., landmark=1., exit=1.))
        h.reset(state())
        expected = [1., 2., 1., 2., 1.][goal]
        assert h(state(**changed), goal) == expected
        assert h(state(**changed), goal) == 0.
        h.reset(state())
        assert h(state(**changed), (goal + 1) % 5) == 0.
        # Progress observed under a different goal cannot be paid retroactively.
        assert h(state(**changed), goal) == 0.


def test_goal_helpers_use_real_engine_exploration():
    from pathlib import Path
    from doom_env import make_env
    from training.goal_rewards import GoalHelpers
    env = make_env(str(Path(__file__).resolve().parents[3] / 'maps/two_rooms.wad'),
                   render_buffers=False, episode_ticks=1000)
    try:
        env.reset(seed=3)
        h = GoalHelpers(dict(explore=1.))
        h.reset(env.privileged_state())
        rewards = []
        for _ in range(25):
            _, _, done, truncated, _ = env.step(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8))
            rewards.append(h(env.privileged_state(), 0))
            if done or truncated:
                break
        assert sum(rewards) > 0
    finally:
        env.close()


def test_collect_pays_real_ammo_pickup_without_itemcount(tmp_path):
    from doom_env import make_env
    from training.mapgen import GridMap, write_map
    from training.goal_rewards import GoalHelpers
    grid = GridMap.parse('##########\n#P..a...E#\n##########')
    env = make_env(str(write_map(grid, tmp_path / 'ammo.wad')), 'MAP01',
                   render_buffers=False, episode_ticks=1000)
    try:
        obs, _ = env.reset(seed=0)
        initial = env.privileged_state()
        h = GoalHelpers(dict(collect=1.))
        h.reset(initial, obs)
        previous_ammo = np.asarray(obs['vitals'])[4:8].copy()
        pickup_seen = False
        for _ in range(60):
            obs, _, done, truncated, _ = env.step(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8))
            current = env.privileged_state()
            reward = h(current, 2, obs)
            ammo = np.asarray(obs['vitals'])[4:8]
            if np.any(ammo > previous_ammo):
                assert current['items'] == initial['items']
                assert reward > 0
                pickup_seen = True
                break
            previous_ammo = ammo.copy()
            if done or truncated:
                break
        assert pickup_seen, 'real corridor clip must increase legal ammo inventory'
    finally:
        env.close()


def test_collect_health_and_ammo_deltas_do_not_pay_for_spending_or_duplicate():
    from training.goal_rewards import GoalHelpers
    # Explicit inventory counterfixtures, not claimed measured engine results.
    def obs(ammo):
        vitals = np.zeros(17)
        vitals[4] = ammo
        return dict(vitals=vitals)
    h = GoalHelpers(dict(collect=1.))
    h.reset(state(health=50.), obs(20))
    assert h(state(health=50.), 2, obs(18)) == 0.
    assert h(state(health=75.), 2, obs(18)) == 1.
    assert h(state(health=75., items=1), 2, obs(28)) == 1.
    assert h(state(health=75., items=1), 2, obs(28)) == 0.
    assert h(state(health=75., items=1), 0, obs(38)) == 0.
    assert h(state(health=75., items=1), 2, obs(38)) == 0.
