"""Unit tests for the procedural map generator (training/mapgen) and its catalog integration."""

from __future__ import annotations

import sys
from collections import deque

import numpy as np
import pytest

from conftest import REPO
from training.mapgen import KINDS, GridMap, generate, to_udmf, write_map
from training.mapgen.udmf import LOCKED, label_regions, validate

sys.path.insert(0, str(REPO / "scripts"))
from build_map_catalog import assign_splits, describe  # noqa: E402

KEY_FOR = {"R": "r", "B": "b", "Y": "y"}


def solvable(grid: GridMap) -> tuple[bool, bool]:
    """(exit reachable with all keys, exit reachable with no keys), walking through doors."""
    def reach(keys: set[str]) -> bool:
        start = next(grid.cells("P"))
        seen, q = {start}, deque([start])
        while q:
            r, c = q.popleft()
            if grid.at(r, c) == "X":
                return True
            for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                ch = grid.at(*n)
                if n in seen or ch == "#" or (ch in LOCKED and KEY_FOR[ch] not in keys):
                    continue
                seen.add(n)
                q.append(n)
        return False

    return reach({"r", "b", "y"}), reach(set())


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_layouts_are_valid_solvable_and_reproducible(kind):
    for seed in range(40):
        grid = generate(kind, seed)
        validate(grid)
        assert len(list(grid.cells("X"))) >= 1
        with_keys, without_keys = solvable(grid)
        assert with_keys, f"{kind} seed {seed}: exit unreachable"
        if kind == "keydoor":
            assert not without_keys, f"{kind} seed {seed}: exit reachable without the key"
            assert any(grid.cells("rby")), "keydoor map has no key"
        assert generate(kind, seed).rows == grid.rows


def test_udmf_doors_and_locks_are_seen_by_the_official_parser(tmp_path):
    from doom_env.mapinfo import load_mapinfo

    grid = GridMap.parse("#############\n#P.D..r..R.X#\n#############")
    info = load_mapinfo(str(write_map(grid, tmp_path / "m.wad")), "MAP01")
    regions = label_regions(grid)
    assert info.num_sectors == max(regions.values()) + 1
    assert sorted(info.door_sectors.values()) == [0, 129]


def test_door_must_sit_in_a_corridor():
    with pytest.raises(ValueError):
        validate(GridMap.parse("#####\n#P.D#\n#...#\n#####"))


def test_udmf_text_is_deterministic():
    g = generate("keydoor", 5)
    assert to_udmf(g) == to_udmf(generate("keydoor", 5))


def _run_corridor(tmp_path, text: str) -> dict:
    """Walk east along a corridor, pressing USE at doors (a test-only script, not the agent)."""
    from doom_env import make_env

    env = make_env(str(write_map(GridMap.parse(text), tmp_path / "c.wad")), "MAP01", render_buffers=False,
                   episode_ticks=2100)
    try:
        obs, _ = env.reset(seed=0)
        for t in range(400):
            ahead, dist = int(obs["rays"]["hit_type"][0]), float(obs["rays"]["distance"][0]) * 1024
            act = np.zeros(9, np.int8)
            if ahead in (2, 3) and dist < 48:
                act[7] = t % 10 == 0  # USE (tapped: USE fires on button-down)
            else:
                act[0] = 1  # forward
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
        return env.privileged_state()
    finally:
        env.close()


def test_generated_map_plays_in_the_engine(tmp_path):
    with_key = _run_corridor(tmp_path, "#################\n#P..D..r..R....X#\n#################")
    assert with_key["exit_reached"] and with_key["keys"][2] == 1  # red keycard picked up
    no_key = _run_corridor(tmp_path, "#################\n#P..D.....R....X#\n#################")
    assert not no_key["exit_reached"]


def test_catalog_reads_tier_from_name_and_pins_curated_splits():
    m = describe("maps/curated/val/t3-keydoor-100000.wad", "curated", "MAP01")
    assert m["tier"] == 3 and m["has_exit"]
    maps = [dict(m), dict(m, id="maps/curated/test/t3-keydoor-200000.wad:MAP01"),
            dict(m, id="maps/generated/t3-keydoor-1.wad:MAP01", source="generated")]
    assign_splits(maps)
    assert [x["split"] for x in maps] == ["val", "test", "train"]
