"""Training rewards (docs/04-TRAINING-PLAN.md §4.2): reward what the scorer rewards, plus helpers.

Uses env.privileged_state() (true position, kills, damage, keys, door heights). That is allowed:
rewards are computed in the training factory only; the agent never sees any of this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ENEMY_TYPES = (1, 2, 3)  # doom_env.spaces.ENTITY_TYPES: enemy_melee, enemy_hitscan, enemy_projectile
RAY_ENEMY, RAY_DOOR, RAY_LOCKED = 4, 2, 3
ATTACK, USE = 6, 7


@dataclass
class RewardConfig:
    exit: float = 10.0
    exit_health_bonus: float = 1.0  # x health/100 at the exit
    exit_time_bonus: float = 1.0  # x fraction of the tick cap left at the exit
    key: float = 3.0
    locked_door: float = 3.0
    door: float = 1.0
    new_area: float = 0.05  # first visit of a CELL x CELL cell this episode
    novelty: float = 0.02  # count-based curiosity: novelty / sqrt(visits) every step
    kill: float = 1.0
    damage_per_hp: float = -0.02
    death: float = -5.0
    step: float = -0.001
    stuck: float = -0.1  # moved less than STUCK_DIST over STUCK_STEPS steps
    shoot_nothing: float = -0.01  # ATTACK with no enemy in view
    use_nothing: float = -0.005  # USE with no door right ahead
    cell: float = 64.0
    stuck_steps: int = 16
    stuck_dist: float = 32.0
    door_open_rise: float = 24.0


class RewardCalculator:
    """reset(state) at episode start; step(state, action, obs) after every env.step()."""

    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()

    def reset(self, state: dict) -> None:
        self.visits: dict[tuple[int, int], int] = {}
        self.trail: list[tuple[float, float]] = []
        self.keys = np.asarray(state["keys"]).copy()
        self.kills = int(state["kills"])
        self.damage = float(state["damage_taken"])
        lock = state.get("door_lock")
        self.door_lock = None if lock is None else np.asarray(lock)
        self.start_gap = self._gaps(state)
        self.opened: set[int] = set()
        self.totals: dict[str, float] = {}
        self._visit(state)

    def step(self, state: dict, action: np.ndarray, obs: dict | None) -> float:
        c = self.cfg
        parts: dict[str, float] = {"step": c.step}

        keys = np.asarray(state["keys"])
        new_keys = int(np.sum((keys > 0) & (self.keys <= 0)))
        if new_keys:
            parts["key"] = c.key * new_keys
        self.keys = keys.copy()

        gaps = self._gaps(state)
        if gaps is not None and self.start_gap is not None and self.door_lock is not None:
            risen = np.nonzero((gaps - self.start_gap >= c.door_open_rise) & (self.door_lock >= 0))[0]
            for i in map(int, risen):
                if i not in self.opened:
                    self.opened.add(i)
                    name = "locked_door" if self.door_lock[i] > 0 else "door"
                    parts[name] = parts.get(name, 0.0) + getattr(c, name)

        kills = int(state["kills"])
        if kills > self.kills:
            parts["kill"] = c.kill * (kills - self.kills)
        self.kills = kills

        damage = float(state["damage_taken"])
        if damage > self.damage:
            parts["damage"] = c.damage_per_hp * (damage - self.damage)
        self.damage = damage

        first, count = self._visit(state)
        if first:
            parts["new_area"] = c.new_area
        parts["novelty"] = c.novelty / math.sqrt(count)

        if len(self.trail) > c.stuck_steps:
            x0, y0 = self.trail[-1 - c.stuck_steps]
            if math.hypot(state["x"] - x0, state["y"] - y0) < c.stuck_dist:
                parts["stuck"] = c.stuck

        if obs is not None:
            if action[ATTACK] and not self._enemy_visible(obs):
                parts["shoot_nothing"] = c.shoot_nothing
            if action[USE] and not self._door_ahead(obs):
                parts["use_nothing"] = c.use_nothing

        cap = max(int(state.get("episode_ticks", 1)), 1)
        time_left = max(0.0, 1.0 - int(state.get("tic", 0)) / cap)
        if state.get("exit_reached"):
            health = min(max(float(state["health"]), 0.0) / 100.0, 1.0)
            parts["exit"] = c.exit + c.exit_health_bonus * health + c.exit_time_bonus * time_left
        if state.get("dead"):
            parts["death"] = c.death

        for k, v in parts.items():
            self.totals[k] = self.totals.get(k, 0.0) + v
        return float(sum(parts.values()))

    # ----------------------------------------------------------------------- helpers
    def _visit(self, state: dict) -> tuple[bool, int]:
        x, y = float(state["x"]), float(state["y"])
        self.trail.append((x, y))
        cell = (int(x // self.cfg.cell), int(y // self.cfg.cell))
        n = self.visits.get(cell, 0) + 1
        self.visits[cell] = n
        return n == 1, n

    @staticmethod
    def _gaps(state: dict):
        floor, ceil = state.get("sector_floor"), state.get("sector_ceiling")
        if floor is None or ceil is None:
            return None
        return np.asarray(ceil, np.float64) - np.asarray(floor, np.float64)

    @staticmethod
    def _enemy_visible(obs: dict) -> bool:
        ents = obs["visible_entities"]
        mask = np.asarray(ents["mask"]).astype(bool)
        if np.isin(np.asarray(ents["entity_type"])[mask], ENEMY_TYPES).any():
            return True
        return bool(np.any(np.asarray(obs["rays"]["hit_type"]) == RAY_ENEMY))

    @staticmethod
    def _door_ahead(obs: dict, max_dist: float = 96.0 / 1024.0) -> bool:
        hits = np.asarray(obs["rays"]["hit_type"])[[62, 63, 0, 1, 2]]  # roughly +/- 11 degrees ahead
        dist = np.asarray(obs["rays"]["distance"])[[62, 63, 0, 1, 2]]
        return bool(np.any(np.isin(hits, (RAY_DOOR, RAY_LOCKED)) & (dist <= max_dist)))
