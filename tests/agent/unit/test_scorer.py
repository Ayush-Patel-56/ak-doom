"""Unit tests for our Mini-League scorer (training/scorer.py) using made-up game states."""

from __future__ import annotations

import numpy as np
import pytest

from training.scorer import POINTS, ScoreTracker


def state(**kw) -> dict:
    s = {
        "tic": 0, "episode_ticks": 2100, "x": 0.0, "y": 0.0, "health": 100.0, "kills": 0,
        "damage_taken": 0.0, "keys": np.zeros(6, np.float32), "dead": False, "exit_reached": False,
        # sectors: 0 = room, 1 = normal door (closed), 2 = locked door (closed)
        "sector_floor": np.zeros(3), "sector_ceiling": np.array([128.0, 0.0, 0.0]),
        "door_lock": np.array([-1, 0, 1]),
    }
    s.update(kw)
    return s


def test_standing_still_scores_one_area_only():
    t = ScoreTracker()
    t.reset(state())
    t.update(state(tic=2100))
    r = t.result()
    assert r.areas == 1 and r.score == POINTS["area"]


def test_spinning_in_place_earns_nothing_extra():
    t = ScoreTracker()
    t.reset(state(x=64.0, y=64.0))  # middle of a cell, so small jitter can't cross a cell border
    for tic in range(4, 400, 4):
        t.update(state(tic=tic, x=69.0, y=61.0))
    assert t.result().areas == 1


def test_exploration_counts_unique_cells():
    t = ScoreTracker()
    t.reset(state())
    for x in (0, 130, 260, 130, 0):  # 3 distinct 128-unit cells, revisits don't count
        t.update(state(x=float(x)))
    assert t.result().areas == 3


def test_doors_keys_kills_and_exit():
    t = ScoreTracker()
    t.reset(state())
    t.update(state(tic=100, sector_ceiling=np.array([128.0, 72.0, 0.0])))  # normal door opens
    t.update(state(tic=200, keys=np.array([1, 0, 0, 0, 0, 0], np.float32), kills=3, damage_taken=20.0))
    t.update(state(tic=300, sector_ceiling=np.array([128.0, 72.0, 72.0]), kills=3, damage_taken=20.0,
                   keys=np.array([1, 0, 0, 0, 0, 0], np.float32)))  # locked door opens
    t.update(state(tic=1050, exit_reached=True, health=80.0, kills=3, damage_taken=20.0,
                   keys=np.array([1, 0, 0, 0, 0, 0], np.float32), sector_ceiling=np.array([128.0, 72.0, 72.0])))
    r = t.result()
    assert (r.exit, r.keys, r.doors, r.locked_doors, r.kills) == (True, 1, 1, 1, 3)
    assert r.buckets["exit"] == pytest.approx(100 + 10 * 0.8 + 10 * 0.5)
    assert r.buckets["progression"] == pytest.approx(15 + 3 + 15)
    assert r.buckets["combat"] == pytest.approx(3 * 2 - 0.05 * 20)


def test_door_counted_once_and_small_wobble_ignored():
    t = ScoreTracker()
    t.reset(state())
    t.update(state(sector_ceiling=np.array([128.0, 10.0, 0.0])))  # rose only 10: not opened
    assert t.result().doors == 0
    for _ in range(3):
        t.update(state(sector_ceiling=np.array([128.0, 72.0, 0.0])))
    assert t.result().doors == 1


def test_dying_early_costs_more_than_dying_late():
    early, late = ScoreTracker(), ScoreTracker()
    early.reset(state())
    late.reset(state())
    early.update(state(tic=210, dead=True))
    late.update(state(tic=1890, dead=True))
    assert early.result().buckets["combat"] < late.result().buckets["combat"] < 0
