"""Contract tests: DoomAgent behaves exactly as docs/CONTRACT.md requires, even on bad input."""

from __future__ import annotations

import ast

import numpy as np
import pytest

from conftest import REPO, SUBMISSION, load_agent_class


def assert_valid(action) -> None:
    arr = np.asarray(action)
    assert arr.shape == (9,), f"shape {arr.shape}"
    assert arr.dtype.kind in "iub", f"dtype {arr.dtype}"
    assert set(np.unique(arr)).issubset({0, 1}), f"values {np.unique(arr)}"


def test_interface_builds_with_official_spaces(spaces):
    obs_space, act_space = spaces
    agent = load_agent_class()(obs_space, act_space, "cpu")
    agent.reset()
    assert_valid(agent.act(obs_space.sample()))


def test_many_sampled_observations_give_valid_actions(agent, spaces):
    obs_space, _ = spaces
    obs_space.seed(0)
    for _ in range(300):
        assert_valid(agent.act(obs_space.sample()))
    assert agent.guard.fallbacks == 0


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"rays": None},
        {"vitals": np.full(17, np.nan, np.float32)},
        {"vitals": np.full(17, np.inf, np.float32)},
        {"vitals": np.zeros(5, np.float32)},
        {"rays": {"distance": np.ones(3), "hit_type": "nonsense"}},
        "not an observation",
    ],
    ids=["none", "empty", "none-rays", "nan-vitals", "inf-vitals", "short-vitals", "broken-rays", "string"],
)
def test_bad_observations_never_crash(agent, bad):
    assert_valid(agent.act(bad))


def test_policy_errors_fall_back_to_safe_action(agent, spaces):
    from smriti import SAFE_ACTION

    obs_space, _ = spaces

    def broken(_obs):
        raise RuntimeError("boom")

    agent.policy = broken
    action = agent.act(obs_space.sample())
    assert_valid(action)
    assert np.array_equal(action, SAFE_ACTION)
    assert agent.guard.fallbacks == 1


@pytest.mark.parametrize("raw", [np.zeros(5), np.zeros((3, 3, 3)), "abc", None])
def test_invalid_policy_outputs_fall_back(agent, spaces, raw):
    agent.policy = lambda _obs: raw
    assert_valid(agent.act(spaces[0].sample()))
    assert agent.guard.fallbacks == 1


def test_policy_output_is_converted_to_binary(agent, spaces):
    agent.policy = lambda _obs: np.array([0.9, 0.1, True, False, 2, -1, np.nan, 0.51, 0.49])
    assert np.array_equal(agent.act(spaces[0].sample()), [1, 0, 1, 0, 1, 0, 0, 1, 0])


def test_reset_clears_per_map_state_and_is_reproducible(spaces):
    obs_space, act_space = spaces
    obs_space.seed(1)
    observations = [obs_space.sample() for _ in range(20)]
    agent = load_agent_class()(obs_space, act_space, "cpu")

    agent.reset()
    first = [agent.act(o) for o in observations]
    assert agent.guard.steps == 20
    agent.reset()
    assert agent.guard.steps == 0 and agent.guard.compute_s == 0.0
    second = [agent.act(o) for o in observations]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_reset_twice_in_a_row_is_fine(agent, spaces):
    agent.reset()
    agent.reset()
    assert_valid(agent.act(spaces[0].sample()))


FORBIDDEN_TOP_LEVEL = {"training", "scripts", "map_spec", "agent_loader", "doom_env", "vizdoom"}


def test_submission_never_imports_training_code():
    """submission/ must only use its own code + base-image libraries, never the training factory."""
    offenders = []
    for path in SUBMISSION.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            offenders += [f"{path.relative_to(REPO)}: {n}" for n in names if n in FORBIDDEN_TOP_LEVEL]
    assert not offenders, offenders


def test_submission_never_writes_files():
    """The repo is read-only at evaluation; agent code must not open files for writing."""
    offenders = []
    for path in SUBMISSION.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in (".write_text(", ".write_bytes(", "np.save(", "torch.save("):
            if needle in text:
                offenders.append(f"{path.relative_to(REPO)}: {needle}")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open":
                modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
                modes += [k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
                if any(set(str(m)) & set("wax+") for m in modes):
                    offenders.append(f"{path.relative_to(REPO)}: open(..., {modes})")
    assert not offenders, offenders
