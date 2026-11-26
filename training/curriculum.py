"""Curriculum stages (docs/04-TRAINING-PLAN.md §4.4): which maps each stage trains on.

A source is either "gen:<kind>" (a fresh procedural map from training seeds 0-99,999, generated on the
fly inside each worker) or a fixed map spec ("@vizdoom/scenarios/x.wad:MAP01", "maps/...:MAP01").
Every later stage keeps some of the earlier maps, so old skills aren't forgotten.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

TRAIN_SEED_LIMIT = 100_000  # val uses 100,000+, test 200,000+ (scripts/generate_maps.py)
GEN_TICKS = {"arena": 2100, "maze": 2800, "rooms": 2800, "keydoor": 3500}
FIXED_TICKS = 2100


@dataclass
class Stage:
    name: str
    mix: dict[str, float]
    notes: str = ""
    ticks: dict[str, int] = field(default_factory=dict)


STAGES: dict[int, Stage] = {
    1: Stage(
        "combat",
        {
            "gen:arena": 0.8,
            "@vizdoom/scenarios/defend_the_center.wad:MAP01": 0.1,
            "@vizdoom/scenarios/deadly_corridor.wad:MAP01": 0.1,
        },
        "survive, aim and kill in open rooms",
    ),
    2: Stage(
        "navigation",
        {
            "gen:maze": 0.3,
            "gen:rooms": 0.35,
            "gen:arena": 0.25,
            "@vizdoom/scenarios/my_way_home.wad:MAP01": 0.1,
        },
        "explore, open doors, find the exit (+25% combat replay)",
    ),
    3: Stage(
        "keys",
        {"gen:keydoor": 0.55, "gen:rooms": 0.15, "gen:maze": 0.1, "gen:arena": 0.2},
        "keycard -> locked door -> exit (+ replay)",
    ),
}


class MapSampler:
    """Picks the next training map for one worker. Deterministic for a given rng seed."""

    def __init__(self, stage: Stage, rng: random.Random, tmp_dir: Path):
        self.stage, self.rng, self.tmp_dir = stage, rng, Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.dropped: set[str] = set()
        self.last_source: str | None = None

    def set_stage(self, stage: Stage) -> None:
        self.stage = stage

    def drop(self, source: str) -> None:
        """Stop using a fixed map that failed to load (generated maps just get a new seed next time)."""
        if not source.startswith("gen:"):
            self.dropped.add(source)

    def next(self) -> tuple[str, str, int, str]:
        """-> (wad path, map name, episode tick cap, label for logs)."""
        mix = {s: w for s, w in self.stage.mix.items() if s not in self.dropped} or self.stage.mix
        sources, weights = zip(*mix.items())
        source = self.rng.choices(sources, weights=weights)[0]
        self.last_source = source
        if source.startswith("gen:"):
            from training.mapgen import generate, write_map

            kind = source[4:]
            seed = self.rng.randrange(TRAIN_SEED_LIMIT)
            path = self.tmp_dir / f"{kind}-{seed}.wad"
            if not path.exists():
                write_map(generate(kind, seed), path)
            return str(path), "MAP01", self.stage.ticks.get(source, GEN_TICKS[kind]), kind
        from map_spec import parse_map

        wad, map_name = parse_map(source)
        label = Path(wad).stem
        return wad, map_name, self.stage.ticks.get(source, FIXED_TICKS), label
