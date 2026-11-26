"""Shared fixtures for our agent tests (the template's own tests live directly in tests/)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUBMISSION = REPO / "submission"

# Same import setup as the scorer: submission/ and the repo root first on sys.path.
for p in (str(REPO), str(SUBMISSION)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_agent_class():
    """Load DoomAgent exactly like the scorer does (by file path, module name 'submission_agent')."""
    spec = importlib.util.spec_from_file_location("submission_agent", SUBMISSION / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DoomAgent


@pytest.fixture(scope="session")
def spaces():
    from doom_env import action_space, observation_space

    return observation_space(), action_space()


@pytest.fixture()
def agent(spaces):
    obs_space, act_space = spaces
    a = load_agent_class()(obs_space, act_space, "cpu")
    a.reset()
    return a
