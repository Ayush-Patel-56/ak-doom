"""Curriculum stages -> training map pools (docs/04-TRAINING-PLAN.md §4.4).

Only Stage 3 (key & door puzzles) is defined in detail here (Phase 3). Earlier stages are listed so a
Stage 3 run keeps replaying them (~20% of games) and does not forget navigation.

Maps are generated on demand into maps/generated/ (git-ignored) from TRAINING seeds (< 100,000), so
they never overlap the held-out val (100,000+) and test (200,000+) maps in maps/curated/.

Stage 3 schedule: "key placed close at first, then farther away". The share of `keynear` maps falls
linearly from `near_start` to `near_end` over the run; the rest of the puzzle share is full `keydoor`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GENERATED = REPO / "maps" / "generated"
TRAIN_SEED_LIMIT = 100_000


@dataclass
class Stage:
    name: str
    replay: dict  # kind -> weight of older-stage maps
    puzzle_share: float = 0.0  # fraction of games on key & door puzzles
    near_start: float = 1.0
    near_end: float = 0.3


STAGES = {
    1: Stage("combat", {"arena": 1.0}),
    2: Stage("navigation", {"maze": 0.5, "rooms": 0.5}),
    3: Stage("key & door puzzles", {"rooms": 0.6, "maze": 0.2, "arena": 0.2}, puzzle_share=0.8),
}


def map_path(kind: str, seed: int) -> Path:
    """Generate (once) and return the WAD for a training map."""
    if not 0 <= seed < TRAIN_SEED_LIMIT:
        raise ValueError(f"training seeds must be in [0, {TRAIN_SEED_LIMIT}) (val/test are held out)")
    from training.mapgen import KINDS, generate, write_map

    path = GENERATED / f"t{KINDS[kind]}-{kind}-{seed}.wad"
    if not path.is_file():
        tmp = path.with_suffix(f".{random.getrandbits(32):08x}.tmp")  # several workers may race
        write_map(generate(kind, seed), tmp)
        tmp.replace(path)
    return path


class MapSampler:
    """Picks (wad, map, kind) for the next game of a stage. progress() in [0, 1] moves the schedule."""

    def __init__(self, stage: int, seed: int, seeds_per_kind: int = 200, kinds: list[str] | None = None):
        self.stage = STAGES[stage]
        self.rng = random.Random(seed)
        self.seeds_per_kind = seeds_per_kind
        self.kinds = kinds  # fixed kind list overrides the stage (used by tests / smoke runs)
        self.progress = 0.0

    def near_share(self) -> float:
        s = self.stage
        return s.near_start + (s.near_end - s.near_start) * min(max(self.progress, 0.0), 1.0)

    def pick_kind(self) -> str:
        if self.kinds:
            return self.rng.choice(self.kinds)
        s = self.stage
        if self.rng.random() < s.puzzle_share:
            return "keynear" if self.rng.random() < self.near_share() else "keydoor"
        kinds, weights = zip(*s.replay.items())
        return self.rng.choices(kinds, weights)[0]

    def __next__(self) -> tuple[str, str, str]:
        kind = self.pick_kind()
        seed = self.rng.randrange(self.seeds_per_kind)
        return str(map_path(kind, seed)), "MAP01", kind
