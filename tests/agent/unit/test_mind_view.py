"""Mind View dashboard and results charts: they render, and they never change what the agent does."""

from __future__ import annotations

import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))

CHAMPION = REPO / "submission" / "weights" / "champion.pt"
KEYNEAR = "maps/curated/val/t3-keynear-100000.wad:MAP01"


def test_render_frame_without_memory_or_manager():
    import mind_view

    snap = mind_view.Snapshot(step=1, max_steps=525, rgb=np.zeros((120, 160, 3), np.uint8),
                              vitals=np.zeros(17, np.float32), buttons=np.zeros(9, np.int8), ms=1.0,
                              policy_name="random")
    img = mind_view.render_frame(snap)
    assert img.size == (mind_view.W, mind_view.H)


def test_view_window_fits_everything_drawn_and_the_agent():
    import mind_view
    from smriti.memory.mindmap import CELL, FREE, GRID, NUM_CHANNELS, WALL

    g = np.zeros((NUM_CHANNELS, GRID, GRID), np.float32)
    g[WALL, 100:110, 120:170] = 1
    g[FREE, 104, 125] = 1
    x, y = 30 * CELL, -20 * CELL  # agent at x-cell 158, y-cell 108
    cells, x0, y0 = mind_view.view_window(g, x, y)
    assert mind_view.MIN_CELLS <= cells <= mind_view.MAX_CELLS and cells % 16 == 0
    assert x0 <= 120 and x0 + cells > 169 and y0 <= 100 and y0 + cells > 109


@pytest.mark.skipif(not CHAMPION.is_file(), reason="needs the trained champion")
def test_recording_matches_the_scored_run_and_reads_memory(tmp_path):
    """The dashboard drives the real agent: same outcome as the Mini-League run (exit in 45 steps)."""
    import mind_view

    summary = mind_view.record(None, KEYNEAR, 0, None, 17.5, 3500, [10, 44], tmp_path, None)
    assert summary["exit"] and summary["keys"] == 1 and summary["locked_doors"] == 1 and summary["steps"] == 45
    assert (tmp_path / "mindview-step0044.png").is_file()


@pytest.mark.skipif(not CHAMPION.is_file() or shutil.which("ffmpeg") is None, reason="needs champion + ffmpeg")
def test_mp4_is_written(tmp_path):
    import mind_view

    out = tmp_path / "clip.mp4"
    mind_view.record(None, KEYNEAR, 0, out, 17.5, 3500, [], None, 12)
    assert out.stat().st_size > 10_000


def test_notebook_bearing_is_left_positive():
    import mind_view
    from smriti.memory.notebook import KEY, Notebook

    class Pose:
        x, y, heading_deg = 0.0, 0.0, 90.0  # facing +y

    class Memory:
        pose = Pose()
        notebook = Notebook()

    Memory.notebook.kind[0] = KEY
    Memory.notebook.pos[0] = (100.0, 100.0)  # ahead-right when facing +y
    note = mind_view.read_notes(Memory)[0]
    assert math.isclose(note["bearing"], -45.0, abs_tol=1e-6)
    assert math.isclose(note["dist_m"], math.hypot(100, 100) / mind_view.UNITS_PER_M)


def test_charts_render_valid_svg(tmp_path, monkeypatch):
    import make_charts

    monkeypatch.setattr(make_charts, "FIG", tmp_path)
    assert make_charts.main([]) == 0
    for name in ("progress-val24.svg", "tiers-val24.svg", "ablation-memory.svg"):
        root = ET.parse(tmp_path / name).getroot()
        assert root.tag.endswith("svg")
    data = json.loads(make_charts.DATA.read_text(encoding="utf-8"))
    assert all(b["total"] is not None for b in data["brains"])


def test_catalog_tick_caps_and_test_map_guard():
    import mind_view

    assert mind_view.default_ticks("maps/curated/val/t3-keynear-100000.wad:MAP01") == 3500
    assert mind_view.default_ticks("two_rooms") == 2100
    with pytest.raises(SystemExit, match="test map"):
        mind_view.refuse_test_map("maps/curated/test/t1-arena-200000.wad:MAP01")


def test_release_check_spots_lfs_pointers_and_wrong_hashes(tmp_path, monkeypatch):
    import release_check

    pointer = tmp_path / "champion.pt"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3018785\n")
    assert release_check.is_lfs_pointer(pointer)
    meta = tmp_path / "champion.json"
    meta.write_text(json.dumps({"weights_sha256": "0" * 64}), encoding="utf-8")
    monkeypatch.setattr(release_check, "CHAMPION", pointer)
    monkeypatch.setattr(release_check, "CHAMPION_JSON", meta)
    ok, msg = release_check.check_hash()
    assert not ok and "!=" in msg
    meta.write_text(json.dumps({"weights_sha256": release_check.sha256(pointer)}), encoding="utf-8")
    assert release_check.check_hash()[0]


@pytest.mark.skipif(not CHAMPION.is_file(), reason="needs the trained champion")
def test_release_check_accepts_the_real_champion():
    import release_check

    if release_check.is_lfs_pointer(CHAMPION):
        pytest.skip("champion.pt is an LFS pointer in this checkout")
    assert release_check.check_hash()[0]
    ok, msg, agent = release_check.check_agent()
    assert ok, msg
    assert release_check.check_plays(agent)[0]
