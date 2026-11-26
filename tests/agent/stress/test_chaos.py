import numpy as np
from scripts.agent_loader import load_agent_class


def test_malformed_observations_and_repeated_resets(spaces):
    from scripts.stress_agent import chaos_observations, valid_action
    agent=load_agent_class()(*spaces,'cpu')
    for obs in chaos_observations(spaces[0]):
        agent.reset();agent.reset()
        assert agent._previous_executed_action is None
        assert valid_action(agent.act(obs))
    agent.reset()
    assert agent.guard.steps==0


def test_stress_scenarios_are_actual_playable_maps():
    from scripts.stress_agent import scenario_maps
    from training.mapgen import playable
    maps=scenario_maps(7)
    assert {'dense','empty','closet'} <= maps.keys()
    for grid in maps.values(): assert playable(grid,'rooms')
    assert sum(ch in 'zsid' for row in maps['dense'].rows for ch in row)>=15
    assert not any(ch in 'zsidrby' for row in maps['empty'].rows for ch in row)


def test_extreme_cleaning_does_not_overflow_or_warn(spaces):
    import warnings
    from smriti.safety import SafetyGuard
    guard=SafetyGuard(*spaces)
    def extreme(x):
        if isinstance(x,dict): return {k:extreme(v) for k,v in x.items()}
        return np.full(x.shape,1e300)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',RuntimeWarning)
        clean=guard.clean(extreme(guard.template))
    def finite(x):
        if isinstance(x,dict): return all(finite(v) for v in x.values())
        return np.isfinite(x).all()
    assert finite(clean)
    assert not caught


def test_representable_depth_extremes_are_preserved(spaces):
    from smriti.safety import SafetyGuard
    guard=SafetyGuard(*spaces)
    raw=dict(guard.template,depth=np.full(guard.template['depth'].shape,255,np.uint8))
    assert np.all(guard.clean(raw)['depth']==255)
