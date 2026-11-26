"""Generate procedural maps (training/mapgen) into WAD files.

Fixed evaluation sets live in maps/curated/{val,test}/ (committed, Git LFS) and use seed ranges that
training never touches; training maps go to maps/generated/ (git-ignored, rebuild any time).

    uv run python scripts/generate_maps.py --set val      # 4 kinds x 4 maps, seeds 100000+
    uv run python scripts/generate_maps.py --set test     # 4 kinds x 4 maps, seeds 200000+
    uv run python scripts/generate_maps.py --set train --count 50     # seeds 0..49 per kind
    uv run python scripts/generate_maps.py --set train --check        # also load every map in the engine

File names: t<tier>-<kind>-<seed>.wad, e.g. t3-keydoor-100002.wad (the catalog reads the tier from it).
Afterwards run scripts/build_map_catalog.py to refresh maps/lists/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "submission")]

from training.mapgen import KINDS, generate, write_map  # noqa: E402

SETS = {  # set -> (folder, first seed, default count per kind)
    "val": (REPO / "maps" / "curated" / "val", 100_000, 4),
    "test": (REPO / "maps" / "curated" / "test", 200_000, 4),
    "train": (REPO / "maps" / "generated", 0, 25),
}
TRAIN_SEED_LIMIT = 100_000  # training seeds must stay below the val/test ranges


def check_loads(path: Path) -> None:
    from doom_env import make_env

    env = make_env(str(path), "MAP01", render_buffers=False, episode_ticks=100)
    try:
        env.reset(seed=0)
        env.step(env.action_space.sample() * 0)
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), required=True)
    ap.add_argument("--count", type=int, help="maps per kind")
    ap.add_argument("--kinds", default=",".join(KINDS), help="comma list of: " + ", ".join(KINDS))
    ap.add_argument("--check", action="store_true", help="load every map in the engine once")
    args = ap.parse_args()

    folder, first_seed, default_count = SETS[args.set]
    count = args.count or default_count
    if args.set == "train" and first_seed + count > TRAIN_SEED_LIMIT:
        print(f"train seeds must stay below {TRAIN_SEED_LIMIT}")
        return 2
    written = 0
    for kind in args.kinds.split(","):
        for seed in range(first_seed, first_seed + count):
            grid = generate(kind, seed)
            path = write_map(grid, folder / f"t{KINDS[kind]}-{kind}-{seed}.wad")
            if args.check:
                check_loads(path)
            written += 1
    print(f"Wrote {written} maps to {folder.relative_to(REPO)}" + (" (all load in the engine)" if args.check else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
