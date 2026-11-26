"""Our copy of the official scorer, for the Mini-League.

The real weights are hidden (docs/EVALUATION.md), so these numbers are OUR GUESSES. They follow the
published rules: exit is worth the most (+ a small bonus for health and time left), then progression
(keys, doors, locked doors), exploration (unique areas, spinning earns nothing) and combat (kills,
minus small penalties for damage and dying early). Harder tiers weigh more.

Uses env.privileged_state() (ground truth), which is fine here: this is evaluation code in the
training factory, never part of the agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

POINTS = {
    "exit": 100.0,
    "exit_health_bonus": 10.0,  # x health/100 (capped at 1) when the exit is reached
    "exit_time_bonus": 10.0,  # x fraction of the tick cap left when the exit is reached
    "key": 15.0,
    "door": 3.0,
    "locked_door": 15.0,
    "area": 0.5,  # per unique AREA_CELL x AREA_CELL cell visited
    "kill": 2.0,
    "damage_per_hp": -0.05,
    "death": -20.0,  # x fraction of the tick cap left (dying early costs more)
}
TIER_WEIGHT = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
AREA_CELL = 128.0  # map units
DOOR_OPEN_RISE = 24.0  # a door counts as opened once its ceiling rose this much above the start


@dataclass
class MapResult:
    exit: bool = False
    keys: int = 0
    doors: int = 0
    locked_doors: int = 0
    areas: int = 0
    kills: int = 0
    damage: float = 0.0
    dead: bool = False
    tic: int = 0
    episode_ticks: int = 0
    health: float = 0.0
    buckets: dict = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ScoreTracker:
    """Call reset(state) after env.reset(), update(state) after every env.step()."""

    def reset(self, state: dict) -> None:
        self.cells: set[tuple[int, int]] = set()
        self.door_lock = np.asarray(state["door_lock"]) if state.get("door_lock") is not None else None
        self.start_gap = self._gaps(state)
        self.opened: set[int] = set()
        self.last = state
        self._visit(state)

    def update(self, state: dict) -> None:
        self._visit(state)
        gaps = self._gaps(state)
        if gaps is not None and self.start_gap is not None:
            risen = np.nonzero((gaps - self.start_gap >= DOOR_OPEN_RISE) & (self.door_lock >= 0))[0]
            self.opened.update(int(i) for i in risen)
        self.last = state

    def result(self) -> MapResult:
        s = self.last
        cap = max(int(s["episode_ticks"]), 1)
        time_left = max(0.0, 1.0 - int(s["tic"]) / cap)
        locks = self.door_lock if self.door_lock is not None else np.zeros(0)
        r = MapResult(
            exit=bool(s["exit_reached"]),
            keys=int(np.sum(np.asarray(s["keys"]) > 0)),
            doors=sum(1 for i in self.opened if locks[i] == 0),
            locked_doors=sum(1 for i in self.opened if locks[i] > 0),
            areas=len(self.cells),
            kills=int(s["kills"]),
            damage=float(s["damage_taken"]),
            dead=bool(s["dead"]),
            tic=int(s["tic"]),
            episode_ticks=cap,
            health=float(s["health"]),
        )
        p = POINTS
        r.buckets = {
            "exit": (p["exit"] + p["exit_health_bonus"] * min(max(r.health, 0.0) / 100.0, 1.0)
                     + p["exit_time_bonus"] * time_left) if r.exit else 0.0,
            "progression": p["key"] * r.keys + p["door"] * r.doors + p["locked_door"] * r.locked_doors,
            "exploration": p["area"] * r.areas,
            "combat": p["kill"] * r.kills + p["damage_per_hp"] * r.damage + (p["death"] * time_left if r.dead else 0.0),
        }
        r.buckets = {k: round(v, 2) for k, v in r.buckets.items()}
        r.score = round(sum(r.buckets.values()), 2)
        return r

    # ----------------------------------------------------------------------- helpers
    def _visit(self, state: dict) -> None:
        self.cells.add((int(state["x"] // AREA_CELL), int(state["y"] // AREA_CELL)))

    @staticmethod
    def _gaps(state: dict):
        floor, ceil = state.get("sector_floor"), state.get("sector_ceiling")
        if floor is None or ceil is None:
            return None
        return np.asarray(ceil, dtype=np.float64) - np.asarray(floor, dtype=np.float64)
