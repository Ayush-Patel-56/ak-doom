"""Load DoomAgent the same way the official scorer does. Shared by benchmark and evaluation scripts.

Import this module BEFORE numpy/torch when you want the evaluation thread limits: it sets the
one-thread environment variables (doom_env.limits.THREAD_ENV) as early as possible.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Same values as doom_env.limits.THREAD_ENV; set before numerical libraries load (checked below).
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}


def apply_eval_thread_limits() -> None:
    for key, value in _THREAD_ENV.items():
        os.environ.setdefault(key, value)
    from doom_env import limits

    assert limits.THREAD_ENV == _THREAD_ENV, "agent_loader._THREAD_ENV is out of date with doom_env.limits"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def pin_one_cpu() -> bool:
    """Pin this process to one CPU core like the scorer (Linux only). Returns True if pinned."""
    if hasattr(os, "sched_setaffinity"):
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
        return True
    return False


def load_agent_class(repo: Path = REPO):
    """Import submission/agent.py by path with submission/ and the repo root on sys.path."""
    submission = repo / "submission"
    for p in (str(repo), str(submission)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("submission_agent", submission / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DoomAgent
