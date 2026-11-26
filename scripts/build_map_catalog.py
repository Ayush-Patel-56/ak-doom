"""Build the map library: find every playable map, describe it, and split it into train/val/test.

Sources (nothing is downloaded):
- Freedoom 2 (MAP01-MAP32), bundled with vizdoom: real levels with doors, keys and exits
- ViZDoom's single-player scenarios: small combat/navigation drills (mostly no exit; training only)
- maps/*.wad (template test map), maps/curated/**, maps/generated/** (our own maps, when we add them)

For each map we read the WAD itself (no engine needed): sector count, doors and locked doors (via the
official doom_env.mapinfo parser), plus for Doom-format maps the monsters/keys present at skill 3 and
whether there is an exit line. A simple heuristic sorts maps into the 4 evaluation tiers.

Split rules: within each tier, maps are ordered by a stable hash; every 10th goes to test, the next
to val, the rest to train (at least 1 val + 1 test per tier when the tier has >= 3 eligible maps).
Maps without an exit, scenarios and the template test map are always train-only.

    uv run python scripts/build_map_catalog.py            # writes maps/catalog.json + maps/lists/*.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

from doom_env.mapinfo import load_mapinfo
from doom_env.wad import list_maps, map_lumps, read_wad
from map_spec import REPO, VIZDOOM_PREFIX, resolve_wad

MAX_SECTORS = 2048  # ViZDoom limit: rays/line-of-sight break beyond it (docs/CONTRACT.md)
TIER_TICKS = {1: 2100, 2: 2800, 3: 3500, 4: 4200}  # our guess within the official 2,100-4,200 range

SCENARIOS = [  # single-player ViZDoom scenarios worth training on (multiplayer/duplicates left out)
    "basic", "simpler_basic", "rocket_basic", "defend_the_center", "defend_the_line", "deadly_corridor",
    "health_gathering", "health_gathering_supreme", "my_way_home", "predict_position", "take_cover",
    "deathmatch",
]

# Doom editor numbers (Doom 2 / Freedoom 2)
MONSTERS = {3004, 9, 65, 3001, 3002, 58, 3006, 3005, 69, 3003, 68, 71, 66, 67, 64, 16, 7, 84}
KEYS = {5, 6, 13, 38, 39, 40}
SKILL3_FLAG, MULTIPLAYER_ONLY_FLAG = 0x0002, 0x0010
EXIT_SPECIALS = {11, 51, 52, 124, 197, 198}  # normal + secret exits (Doom/Boom)


def sources() -> list[tuple[str, str]]:
    """(portable wad reference, source name)."""
    out = [(f"{VIZDOOM_PREFIX}freedoom2.wad", "freedoom2")]
    out += [(f"{VIZDOOM_PREFIX}scenarios/{name}.wad", "vizdoom-scenario") for name in SCENARIOS]
    for folder, source in (("maps", "template"), ("maps/curated", "curated"), ("maps/generated", "generated")):
        base = REPO / folder
        if base.is_dir():
            pattern = "*.wad" if folder == "maps" else "**/*.wad"
            out += [(p.relative_to(REPO).as_posix(), source) for p in sorted(base.glob(pattern))]
    return out


def doom_format_stats(lumps: dict[str, bytes]) -> dict:
    """Monsters/keys present at skill 3 and exit lines. Only for classic Doom-format maps."""
    things = lumps.get("THINGS", b"")
    monsters = keys = 0
    for i in range(0, len(things) - 9, 10):
        _x, _y, _a, kind, flags = struct.unpack_from("<hhhHH", things, i)
        if not flags & SKILL3_FLAG or flags & MULTIPLAYER_ONLY_FLAG:
            continue
        monsters += kind in MONSTERS
        keys += kind in KEYS
    lines = lumps.get("LINEDEFS", b"")
    exits = sum(
        struct.unpack_from("<HHHHHhh", lines, i)[3] in EXIT_SPECIALS for i in range(0, len(lines) - 13, 14)
    )
    return {"monsters": monsters, "keys": keys, "exit_lines": exits}


def tier_for(m: dict) -> int:
    """Heuristic tier, mirroring docs/EVALUATION.md: 1 combat, 2 multi-room, 3 key puzzle, 4 gauntlet."""
    if m["locked_doors"] > 0 and (m.get("keys") or 0) > 0:
        big = m["sectors"] > 300 or (m.get("monsters") or 0) > 60
        return 4 if big else 3
    if m["doors"] > 0 or m["sectors"] > 40:
        return 2
    return 1


def describe(ref: str, source: str, map_name: str) -> dict:
    path = resolve_wad(ref)
    info = load_mapinfo(path, map_name)
    m = {
        "id": f"{ref}:{map_name}",
        "source": source,
        "sectors": info.num_sectors,
        "format": info.format or "doom",
        "doors": sum(1 for lock in info.door_sectors.values() if lock == 0),
        "locked_doors": sum(1 for lock in info.door_sectors.values() if lock > 0),
    }
    lumps = map_lumps(read_wad(path), map_name)
    if "TEXTMAP" not in lumps and "BEHAVIOR" not in lumps:
        m.update(doom_format_stats(lumps))
    else:
        m.update({"monsters": None, "keys": None, "exit_lines": None})
    m["tier"] = tier_for(m)
    m["episode_ticks"] = TIER_TICKS[m["tier"]]
    m["has_exit"] = bool(m["exit_lines"]) if m["exit_lines"] is not None else None
    m["usable"] = m["sectors"] <= MAX_SECTORS
    return m


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:12], 16)


def assign_splits(maps: list[dict]) -> None:
    by_tier: dict[int, list[dict]] = defaultdict(list)
    for m in maps:
        if not m["usable"]:
            m["split"] = "excluded"
        elif m["source"] in ("vizdoom-scenario", "template") or m["has_exit"] is False:
            m["split"] = "train"  # drills and exit-less maps can't be scored like evaluation maps
        else:
            by_tier[m["tier"]].append(m)
    for tier_maps in by_tier.values():
        tier_maps.sort(key=lambda m: stable_hash(m["id"]))
        for i, m in enumerate(tier_maps):
            m["split"] = "test" if i % 10 == 0 else "val" if i % 10 == 1 else "train"
        if len(tier_maps) >= 3 and not any(m["split"] == "val" for m in tier_maps):
            tier_maps[1]["split"] = "val"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=REPO / "maps")
    args = ap.parse_args()

    maps: list[dict] = []
    for ref, source in sources():
        path = resolve_wad(ref)
        if not Path(path).is_file():
            print(f"skip (missing): {ref}")
            continue
        for map_name in list_maps(read_wad(path)):
            try:
                maps.append(describe(ref, source, map_name))
            except Exception as exc:  # noqa: BLE001 - a broken map should not stop the catalog
                print(f"skip {ref}:{map_name}: {exc}")
    assign_splits(maps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "catalog.json").write_text(json.dumps({"maps": maps}, indent=2) + "\n", encoding="utf-8")
    lists = args.out_dir / "lists"
    lists.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        ids = [m["id"] for m in maps if m["split"] == split]
        (lists / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    print(f"{'split':<9}{'maps':>5}   tiers 1/2/3/4")
    for split in ("train", "val", "test", "excluded"):
        sel = [m for m in maps if m["split"] == split]
        tiers = "/".join(str(sum(m["tier"] == t for m in sel)) for t in (1, 2, 3, 4))
        print(f"{split:<9}{len(sel):>5}   {tiers}")
    print(f"\nWrote {args.out_dir / 'catalog.json'} and {lists}/{{train,val,test}}.txt")


if __name__ == "__main__":
    main()
