"""Unit tests for scripts/map_spec.py (map arguments and portable WAD paths)."""

from __future__ import annotations

import sys
from pathlib import Path

from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))
from map_spec import parse_map, resolve_wad  # noqa: E402


def test_windows_drive_letter_is_not_a_map_name():
    wad, name = parse_map(r"F:\ak-doom\maps\two_rooms.wad:MAP03")
    assert wad == r"F:\ak-doom\maps\two_rooms.wad" and name == "MAP03"


def test_default_map_and_episode_names():
    assert parse_map(r"C:\x\y.wad")[1] == "MAP01"
    assert parse_map("maps/two_rooms.wad:e1m2")[1] == "E1M2"


def test_relative_paths_resolve_against_repo():
    assert Path(resolve_wad("maps/two_rooms.wad")).resolve() == (REPO / "maps" / "two_rooms.wad").resolve()


def test_vizdoom_prefix_points_at_bundled_wads():
    path = Path(resolve_wad("@vizdoom/freedoom2.wad"))
    assert path.name == "freedoom2.wad" and path.is_file()


def test_windows_drive_paths_are_kept_on_every_os():
    assert resolve_wad(r"C:\maps\x.wad") == r"C:\maps\x.wad"
    assert resolve_wad("D:/maps/x.wad") == "D:/maps/x.wad"
