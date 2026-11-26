"""Phase 3 memory: mind map, landmark notebook, EpisodeMemory (docs/03-SYSTEM-DESIGN.md §3.3.2-3.3.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from smriti.memory import EpisodeMemory, MindMap, Notebook, Odometer
from smriti.memory import mindmap as mm
from smriti.memory import notebook as nbm


def rays(dist_units=None, hit=None):
    d = np.ones(64, np.float32) if dist_units is None else np.asarray(dist_units, np.float32) / 1024.0
    h = np.zeros(64, np.int64) if hit is None else np.asarray(hit, np.int64)
    return {"distance": d, "hit_type": h}


def obs_with(r, entities=None, keys=None):
    feats = np.zeros((32, 5), np.float32)
    etype = np.zeros(32, np.int64)
    mask = np.zeros(32, np.int8)
    for i, (t, dist, rel_deg) in enumerate(entities or []):
        feats[i, :3] = (dist / 1024.0, math.sin(math.radians(rel_deg)), math.cos(math.radians(rel_deg)))
        etype[i], mask[i] = t, 1
    vit = np.zeros(17, np.float32)
    vit[0] = 100
    if keys:
        vit[8:14] = keys
    return {"rays": r, "visible_entities": {"features": feats, "entity_type": etype, "mask": mask}, "vitals": vit}


# ----------------------------------------------------------------------------- mind map
def test_wall_ahead_is_drawn_ahead_and_free_space_before_it():
    m = MindMap()
    d = np.full(64, 1024.0)
    d[0] = 300.0
    h = np.zeros(64)
    h[0] = 1  # wall straight ahead
    m.write(0.0, 0.0, 0.0, rays(d, h))
    cx, cy = (int(v) for v in m.to_cell(304.0, 0.0))
    assert m.grid[mm.WALL, cy, cx] == 1
    fx, fy = (int(v) for v in m.to_cell(150.0, 0.0))
    assert m.grid[mm.FREE, fy, fx] == 1 and m.grid[mm.WALL, fy, fx] == 0
    assert m.grid[mm.VISITED, mm.GRID // 2, mm.GRID // 2] == 1


@pytest.mark.parametrize("heading", [0.0, 90.0, -135.0])
def test_crop_is_rotated_so_up_is_facing(heading):
    """A wall 200 units ahead of the agent appears above the crop centre whatever the heading."""
    m = MindMap()
    d = np.full(64, 1024.0)
    d[0] = 200.0
    h = np.zeros(64)
    h[0] = 1
    x, y = 100.0, -50.0
    m.write(x, y, heading, rays(d, h))
    crop = m.crop(x, y, heading)
    assert crop.shape == (mm.NUM_CHANNELS, mm.CROP, mm.CROP) and crop.dtype == np.uint8
    rows, cols = np.nonzero(crop[mm.WALL])
    assert rows.size, "wall missing from crop"
    centre = mm.CROP / 2 - 0.5
    assert np.all(rows < centre) and np.all(np.abs(cols - centre) <= 1.5)
    assert crop[mm.VISITED, mm.CROP // 2 - 1: mm.CROP // 2 + 1, mm.CROP // 2 - 1: mm.CROP // 2 + 1].max() > 0


def test_left_wall_appears_on_the_left_of_the_crop():
    m = MindMap()
    d = np.full(64, 1024.0)
    d[16] = 160.0  # ray 16 = 90 deg to the left
    h = np.zeros(64)
    h[16] = 1
    m.write(0.0, 0.0, 30.0, rays(d, h))
    rows, cols = np.nonzero(m.crop(0.0, 0.0, 30.0)[mm.WALL])
    assert cols.size and np.all(cols < mm.CROP / 2 - 0.5)


def test_door_seen_through_is_cleared_and_enemy_mark_fades():
    m = MindMap()
    d = np.full(64, 1024.0)
    h = np.zeros(64)
    d[0], h[0] = 200.0, 2  # closed door ahead
    d[8], h[8] = 300.0, 4  # enemy
    m.write(0.0, 0.0, 0.0, rays(d, h))
    cx, cy = (int(v) for v in m.to_cell(204.0, 0.0))
    assert m.grid[mm.DOOR, cy, cx] == 1
    m.write(0.0, 0.0, 0.0, rays())  # door opened: rays pass through
    assert m.grid[mm.DOOR, cy, cx] == 0
    first = m.crop(0.0, 0.0, 0.0)[mm.ENEMY].max()
    for _ in range(100):
        m.tick()
    assert first > 0 and m.crop(0.0, 0.0, 0.0)[mm.ENEMY].max() < first // 4


def test_positions_outside_the_grid_are_ignored():
    m = MindMap()
    m.write(1e6, -1e6, 0.0, rays())
    m.crop(1e6, -1e6, 0.0)
    assert m.grid.sum() == 0
    m.reset()
    assert m.grid.sum() == 0 and m.t == 0


# ----------------------------------------------------------------------------- notebook
def test_locked_door_is_noted_once_and_marked_open_when_seen_through():
    nb = Notebook()
    d = np.full(64, 1024.0)
    h = np.zeros(64)
    d[[63, 0, 1]], h[[63, 0, 1]] = 400.0, 3  # one locked door across the wrap-around
    for _ in range(3):
        nb.write(0.0, 0.0, 0.0, obs_with(rays(d, h)))
    assert int((nb.kind == nbm.LOCKED_DOOR).sum()) == 1
    s = int(np.nonzero(nb.kind == nbm.LOCKED_DOOR)[0][0])
    assert abs(nb.pos[s, 0] - 404) < 20 and abs(nb.pos[s, 1]) < 20 and nb.done[s] == 0
    nb.write(0.0, 0.0, 0.0, obs_with(rays()))  # rays now pass the door: open
    assert nb.done[s] == 1
    f, mask = nb.features(0.0, 0.0, 90.0)  # facing left: the door is now on our right
    assert mask[s] and f[s, 1] == 1 and f[s, 4] < -0.99  # sin(rel) = -1 -> right


def test_key_sighting_and_pickup():
    nb = Notebook()
    # key 300 units away, 90 deg to the left, while facing +x from (100, 0)
    nb.write(100.0, 0.0, 0.0, obs_with(rays(), entities=[(nbm.ENTITY_KEY, 300.0, 90.0)]))
    s = int(np.nonzero(nb.kind == nbm.KEY)[0][0])
    assert np.allclose(nb.pos[s], (100.0, 300.0), atol=1.0)
    # walk to it and pick it up: blue card flag flips
    nb.write(100.0, 290.0, 90.0, obs_with(rays(), keys=[0, 0, 0, 0, 0, 0]))
    nb.write(100.0, 300.0, 90.0, obs_with(rays(), keys=[1, 0, 0, 0, 0, 0]))
    assert nb.done[s] == 1 and nb.pickups == 1
    assert int((nb.kind == nbm.KEY).sum()) == 1


def test_notebook_replaces_least_recent_slot_when_full():
    nb = Notebook(slots=4)
    for i in range(6):
        nb.write(float(i) * 1000.0, 0.0, 0.0, obs_with(rays(), entities=[(nbm.ENTITY_KEY, 100.0, 0.0)]))
    assert (nb.kind == nbm.KEY).sum() == 4
    xs = sorted(nb.pos[:, 0])
    assert xs[0] >= 2000  # the two oldest were replaced


def test_features_empty_and_finite():
    nb = Notebook()
    f, mask = nb.features(0.0, 0.0, 0.0)
    assert f.shape == (nbm.NUM_SLOTS, nbm.NOTE_FEATURES) and not mask.any()
    nb.write(0.0, 0.0, 0.0, obs_with(rays(), entities=[(nbm.ENTITY_KEY, 0.0, 0.0)]))  # key at our feet
    f, mask = nb.features(0.0, 0.0, 0.0)
    assert np.isfinite(f).all() and mask.sum() == 1


# ----------------------------------------------------------------------------- EpisodeMemory
def test_episode_memory_without_odometer_gives_zero_features():
    mem = EpisodeMemory(None)
    o = obs_with(rays())
    mem.observe(None, o, o)
    f = mem.features()
    assert not mem.valid and f["map"].max() == 0 and not f["note_mask"].any()


def test_episode_memory_freezes_on_bad_transition_and_resets():
    mem = EpisodeMemory(Odometer())
    o = obs_with(rays())
    mem.observe(None, o, o)
    before = mem.mindmap.grid.copy()
    bad = dict(o, vitals=np.full(17, np.nan, np.float32))
    mem.observe(np.zeros(9, np.int8), bad, o)
    assert not mem.valid and np.array_equal(before, mem.mindmap.grid)
    mem.reset()
    assert mem.valid and mem.mindmap.grid.sum() == 0 and mem.notebook.kind.sum() == 0


def test_memory_tracks_true_position_on_a_real_map():
    """Training-only ground truth: our own visited cell follows the engine's position (short, clean run)."""
    from pathlib import Path

    from doom_env import make_env

    repo = Path(__file__).resolve().parents[3]
    env = make_env(str(repo / "maps" / "two_rooms.wad"), render_buffers=False, episode_ticks=1000)
    mem = EpisodeMemory(Odometer())
    try:
        obs, _ = env.reset(seed=3)
        spawn = env.privileged_state()
        mem.observe(None, obs, obs)
        a = np.zeros(9, np.int8)
        a[0] = 1
        errors = []
        for t in range(40):
            a[4] = 1 if t % 10 < 3 else 0
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc or obs["rays"]["distance"][0] * 1024 < 64:
                break  # stop before wall contact: dead reckoning drifts there (System Design §3.3.1)
            mem.observe(a, obs, obs)
            s = env.privileged_state()
            h = math.radians(spawn["angle"])
            dx, dy = s["x"] - spawn["x"], s["y"] - spawn["y"]
            true = (dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h))
            errors.append(math.hypot(mem.pose.x - true[0], mem.pose.y - true[1]))
        assert errors and max(errors) < 2 * mm.CELL
        assert len(errors) >= 8 and mem.mindmap.cells_visited >= 3
        assert mem.mindmap.grid[mm.WALL].sum() > 10  # it saw the room's walls
    finally:
        env.close()
