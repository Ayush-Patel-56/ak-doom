"""Real-engine odometry regression tests; expected poses come from ViZDoom, not the estimator."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from doom_env import make_env
from doom_env import spaces as S
from conftest import REPO
from smriti.memory.odometer import Calibration, Odometer


def press(*names):
    action = np.zeros(len(S.BUTTONS), np.int8)
    for name in names:
        action[S.BUTTONS.index(name)] = 1
    return action


def true_pose(state, spawn):
    dx, dy = state['x'] - spawn['x'], state['y'] - spawn['y']
    h = math.radians(spawn['angle'])
    return np.array([dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h)])


@pytest.mark.parametrize('prefix', list(itertools.product(range(4), repeat=2)))
def test_heading_tracks_startup_reversal_release_and_wrap(prefix):
    odometer = Odometer()
    env = make_env(str(REPO / 'maps/two_rooms.wad'), render_buffers=False, episode_ticks=2000)
    try:
        for seed in (0, 1):
            env.reset(seed=seed)
            odometer.reset()
            spawn = env.privileged_state()
            codes = [*prefix, *([1] * 30), 2, 3, 2, 0, 2, 1]
            for code in codes:
                action = press()
                action[S.BUTTONS.index('TURN_LEFT')] = code & 1
                action[S.BUTTONS.index('TURN_RIGHT')] = code >> 1
                obs, _, done, truncated, _ = env.step(action)
                assert not (done or truncated)
                pose = odometer.update(action, obs)
                actual = env.privileged_state()['angle'] - spawn['angle']
                error = (pose.heading_deg - actual + 180) % 360 - 180
                assert abs(error) < 0.06
    finally:
        env.close()


@pytest.mark.parametrize('map_name', ['two_rooms.wad', 'curated/val/t2-maze-100000.wad', 'curated/val/t3-keydoor-100000.wad'])
def test_position_on_real_maps_with_strafing_turning_and_wall_contact(map_name):
    env = make_env(str(REPO / 'maps' / map_name), render_buffers=False, episode_ticks=2000)
    odometer = Odometer()
    patterns = [press('MOVE_FORWARD'), press('MOVE_BACKWARD'), press('MOVE_LEFT'), press('MOVE_RIGHT'),
                press('MOVE_FORWARD', 'TURN_LEFT'), press('MOVE_RIGHT', 'TURN_RIGHT'), press(),
                press('TURN_LEFT', 'TURN_RIGHT')]
    rng = np.random.default_rng(733)
    try:
        env.reset(seed=733)
        spawn = env.privileged_state()
        previous = spawn
        distance = 0.
        errors = []
        for i in range(240):
            if i % 6 == 0:
                action = patterns[int(rng.integers(len(patterns)))]
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:  # terminal observation is stale in the official wrapper
                break
            state = env.privileged_state()
            pose = odometer.update(action, obs)
            distance += math.hypot(state['x'] - previous['x'], state['y'] - previous['y'])
            errors.append(np.linalg.norm(np.array([pose.x, pose.y]) - true_pose(state, spawn)))
            previous = state
        assert len(errors) >= 100
        assert distance > 100
        # Dead reckoning is approximate at collisions: bound trajectory-wide drift, not a copied result.
        assert max(errors) < max(32., .05 * distance)
    finally:
        env.close()


def test_bad_transition_is_rejected_without_changing_pose():
    odometer = Odometer()
    original = odometer.pose
    for action, obs in [(np.zeros(8), {}), (press(), {'vitals': np.full(17, np.nan)}),
                        (press(), {'vitals': np.zeros(3)}), (np.full(9, 2), {'vitals': np.zeros(17)})]:
        with pytest.raises(ValueError):
            odometer.update(action, obs)
        assert odometer.pose == original


def test_calibration_rejects_mismatched_contract_and_nonfinite_coefficients():
    import copy
    import json
    from smriti.memory.odometer import DEFAULT_CALIBRATION
    data = json.loads(DEFAULT_CALIBRATION.read_text())
    for key, value in [('frame_skip', 1), ('engine_version', '0.0'), ('schema_version', 999)]:
        bad = copy.deepcopy(data)
        bad[key] = value
        with pytest.raises(ValueError):
            Calibration.from_dict(bad)
    bad = copy.deepcopy(data)
    bad['motion']['a_now'] = float('nan')
    with pytest.raises(ValueError):
        Calibration.from_dict(bad)


def test_agent_tracks_executed_fallback_action_and_resets(spaces):
    from conftest import load_agent_class
    env = make_env(str(REPO / 'maps/two_rooms.wad'), render_buffers=False)
    agent = load_agent_class()(*spaces, 'cpu')
    agent.reset()
    try:
        obs, _ = env.reset(seed=5)
        spawn = env.privileged_state()
        def broken(_):
            raise RuntimeError('exercise real SafetyGuard fallback')
        agent.policy = broken
        agent.act(obs)  # reset observation must not advance the odometer
        assert agent.odometer.steps == 0
        action = press('MOVE_FORWARD', 'TURN_LEFT')  # SafetyGuard fallback, not the failed policy output
        for _ in range(12):
            obs, _, done, truncated, _ = env.step(action)
            assert not (done or truncated)
            action = agent.act(obs)
            actual = env.privileged_state()
            assert abs((agent.pose.heading_deg - (actual['angle'] - spawn['angle']) + 180) % 360 - 180) < .06
            assert np.linalg.norm(np.array([agent.pose.x, agent.pose.y]) - true_pose(actual, spawn)) < 10
        assert agent.odometer.steps == 12
        # Restore the real policy before reset (test callable intentionally has no reset).
        from smriti.policy import RandomPolicy
        agent.policy = RandomPolicy(9)
        agent.reset()
        assert agent.odometer.steps == 0
        assert agent.pose.x == agent.pose.y == agent.pose.heading_deg == 0
        assert agent.odometry_valid
        agent.act(env.reset(seed=6)[0])
        assert agent.odometer.steps == 0
    finally:
        env.close()


def test_agent_marks_missing_motion_invalid_without_breaking_actions(agent, spaces):
    agent.act(spaces[0].sample())
    action = agent.act({'vitals': np.full(17, np.nan)})
    assert action.shape == (9,)
    assert not agent.odometry_valid
    frozen = agent.pose
    agent.act(spaces[0].sample())
    assert agent.pose == frozen  # do not invent movement across a missing transition
    agent.reset()
    assert agent.odometry_valid


@pytest.mark.parametrize('angle', [90, 225])
def test_spawn_frame_is_independent_of_absolute_heading(tmp_path, angle):
    from training.mapgen import GridMap, write_map
    grid = GridMap.parse('########\n#......#\n#......#\n#..P...#\n#......#\n#.....X#\n########', player_angle=angle)
    wad = write_map(grid, tmp_path / 'rotated.wad')
    env = make_env(str(wad), render_buffers=False)
    odometer = Odometer()
    try:
        env.reset(seed=9)
        spawn = env.privileged_state()
        assert abs(spawn['angle'] - angle) < .06
        for action in [press('MOVE_FORWARD')] * 7 + [press('MOVE_RIGHT')] * 7:
            obs, _, done, truncated, _ = env.step(action)
            assert not (done or truncated)
            pose = odometer.update(action, obs)
            assert np.linalg.norm(np.array([pose.x, pose.y]) - true_pose(env.privileged_state(), spawn)) < 10
    finally:
        env.close()


def test_validation_cli_reports_real_drift_and_fails_exceeded_budget(tmp_path):
    import json
    import subprocess
    import sys
    report_path = tmp_path / 'report.json'
    command = [sys.executable, str(REPO / 'scripts/check_odometry.py'), '--steps', '60',
               '--map', str(REPO / 'maps/two_rooms.wad'), '--out', str(report_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report['passed']
    assert report['episodes'][0]['steps'] == 60
    assert report['episodes'][0]['distance_travelled_units'] > 0
    assert report['episodes'][0]['max_heading_error_deg'] < .06
    result = subprocess.run([*command, '--max-position-drift-fraction', '0'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stderr
    assert not json.loads(report_path.read_text())['passed']


def test_calibration_does_not_fit_stale_terminal_observations():
    import sys
    sys.path.insert(0, str(REPO / 'scripts'))
    from measure_odometry import wander
    env = make_env(str(REPO / 'maps/two_rooms.wad'), render_buffers=False, episode_ticks=40)
    try:
        samples = list(wander(env, seed=11, steps=100))
        assert samples
        assert all(not state_after['finished'] for _, _, _, state_after in samples)
    finally:
        env.close()
