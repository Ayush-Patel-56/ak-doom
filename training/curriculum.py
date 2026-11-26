"""Curriculum stages -> training map pools (docs/04-TRAINING-PLAN.md §4.4).

Stage 4 samples catalog training full levels with earlier generated-map replay. Stage 3
replays earlier stages (~20% of games) to retain navigation skills.

Maps are generated on demand into maps/generated/ (git-ignored) from TRAINING seeds (< 100,000), so
they never overlap the held-out val (100,000+) and test (200,000+) maps in maps/curated/.

Stage 3 schedule: "key placed close at first, then farther away". The share of `keynear` maps falls
linearly from `near_start` to `near_end` over the run; the rest of the puzzle share is full `keydoor`.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from pathlib import Path

from training.map_pool import DEFAULT_CATALOG, load_catalog_pool, map_identity

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
    5: Stage("mixed randomised", {"arena": .2, "maze": .2, "rooms": .2, "keydoor": .2, "keynear": .2}),
    4: Stage("full levels", {"keydoor": 0.4, "rooms": 0.3, "maze": 0.15, "arena": 0.15}),
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

    def __init__(self, stage: int, seed: int, seeds_per_kind: int | None = None, kinds: list[str] | None = None,
                 puzzle_share: float | None = None, full_level_share: float | None = None,
                 catalog_path=DEFAULT_CATALOG):
        from training.mapgen import KINDS

        seeds_per_kind = (TRAIN_SEED_LIMIT if stage == 5 else 200) if seeds_per_kind is None else seeds_per_kind
        full_level_share = (.5 if stage == 5 else .8) if full_level_share is None else full_level_share
        self._generated_metadata = {}
        if stage not in STAGES:
            raise ValueError(f"unknown curriculum stage: {stage}")
        if not 0 <= full_level_share <= 1:
            raise ValueError("full_level_share must be in [0, 1]")
        if not 0 < seeds_per_kind <= TRAIN_SEED_LIMIT:
            raise ValueError("seeds_per_kind must use the training seed range")
        if kinds is not None and (not kinds or any(k not in KINDS and k != 'full' for k in kinds)):
            raise ValueError("invalid curriculum kinds")
        if kinds and 'full' in kinds and stage not in (4,5):
            raise ValueError("full maps require stage 4 or 5")
        self.stage_number = stage
        self.full_level_share = full_level_share
        self.catalog_path = Path(catalog_path)
        self.full_pool = []
        if stage in (4,5) and (kinds is None or 'full' in kinds):
            self.full_pool = load_catalog_pool('train', [1,2,3,4] if stage == 5 else [4],
                                               source=None if stage == 5 else 'freedoom2', catalog_path=catalog_path)
            if not self.full_pool:
                raise ValueError("empty full-level training pool")
        self._caps = {map_identity(r['id']): r['episode_ticks'] for r in self.full_pool}
        self.stage = STAGES[stage]
        if puzzle_share is not None:  # e.g. more replay when warm-starting from a brain that must not forget
            self.stage = replace(self.stage, puzzle_share=puzzle_share)
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
        if self.stage_number in (4,5) and self.rng.random() < self.full_level_share:
            return "full"
        if self.rng.random() < s.puzzle_share:
            return "keynear" if self.rng.random() < self.near_share() else "keydoor"
        kinds, weights = zip(*s.replay.items())
        return self.rng.choices(kinds, weights)[0]

    def __next__(self) -> tuple[str, str, str]:
        kind = self.pick_kind()
        if kind == 'full':
            row = self.rng.choice(self.full_pool)
            return row['resolved_wad'], row['map_name'], kind
        seed = self.rng.randrange(self.seeds_per_kind)
        if self.stage_number == 5:
            from training.mapgen.randomize import generate_randomized, PROFILE_HASH
            from training.mapgen import write_map
            grid = generate_randomized(kind, seed)
            path = GENERATED / f"s5-{PROFILE_HASH}-{kind}-{seed}.wad"
            if not path.is_file():
                tmp = path.with_suffix(f".{random.getrandbits(32):08x}.tmp")
                write_map(grid,tmp)
                tmp.replace(path)
            self._generated_metadata[map_identity(f"{path}:MAP01")] = dict(grid.notes)
            return str(path), "MAP01", kind
        return str(map_path(kind, seed)), "MAP01", kind

    def episode_ticks(self, wad: str, map_name: str) -> int:
        """Catalog cap for full levels, legacy 2100 ticks for generated replay."""
        identity = map_identity(f"{wad}:{map_name}")
        return self._caps.get(identity, self._generated_metadata.get(identity, {}).get('randomisation', {}).get('episode_ticks', 2100))

    @property
    def pool_manifest(self) -> dict:
        """Serializable sampling provenance saved alongside experiment configs."""
        from training.mapgen.randomize import PROFILE, PROFILE_HASH
        content = {}
        if self.stage_number == 5:
            for row in self.full_pool:
                path = row['resolved_wad']
                if path not in content: content[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return {
            **({'randomisation': dict(PROFILE, profile_hash=PROFILE_HASH)} if self.stage_number == 5 else {}),
            'stage': self.stage_number,
            'catalog_sha256': hashlib.sha256(self.catalog_path.read_bytes()).hexdigest() if self.full_pool else None,
            'full_level_share': self.full_level_share if self.stage_number in (4,5) else 0.0,
            'kinds_override': list(self.kinds) if self.kinds else None,
            'seeds_per_kind': self.seeds_per_kind,
            'generated_seed_range': [0, self.seeds_per_kind],
            'replay_weights': dict(self.stage.replay),
            'puzzle_share': self.stage.puzzle_share,
            'near_start': self.stage.near_start,
            'near_end': self.stage.near_end,
            'full_pool': [dict(id=r['id'], source=r['source'], tier=r['tier'],
                               episode_ticks=r['episode_ticks'],
                               canonical_identity=self._portable_identity(r['id']),
                               **({'wad_sha256':content[r['resolved_wad']]} if self.stage_number==5 else {})) for r in self.full_pool],
        }

    def map_metadata(self, wad: str, map_name: str) -> dict | None:
        """Copy full-level metadata for rollout diagnostics; replay has no catalog row."""
        identity = map_identity(f"{wad}:{map_name}")
        if identity in self._generated_metadata:
            return dict(self._generated_metadata[identity])
        return next((dict(row) for row in self.full_pool if map_identity(row['id']) == identity), None)

    @staticmethod
    def _portable_identity(spec: str) -> list[str]:
        """Canonical identity without machine-specific repository/package roots."""
        from scripts.map_spec import vizdoom_dir

        wad, name = map_identity(spec)
        path = Path(wad)
        for root, prefix in ((vizdoom_dir().resolve(), '@vizdoom/'), (REPO.resolve(), '')):
            try:
                return [prefix + path.relative_to(root).as_posix(), name]
            except ValueError:
                pass
        return [wad, name]
