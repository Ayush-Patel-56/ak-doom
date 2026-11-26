"""Training-only helper rewards from scorer events, never actor inputs."""
from __future__ import annotations

import numpy as np

from training.scorer import ScoreTracker

DEFAULT_WEIGHTS = dict(explore=.02, fight=.05, collect=.05, landmark=.05, exit=.1)


class GoalHelpers:
    """Goals: Explore, Fight, Collect, Go to landmark, Go to exit.

    Collection combines item/key counters with positive health/ammo inventory
    changes using their maximum, avoiding duplicate credit for the same pickup. All event baselines advance on every call irrespective of
    active goal. Reset explicitly for each episode after environment reset.
    """
    def __init__(self, weights=None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            if set(weights) - set(self.weights):
                raise ValueError('unknown goal helper weight')
            self.weights.update(weights)
        if not all(np.isfinite(v) and v >= 0 for v in self.weights.values()):
            raise ValueError('goal helper weights must be finite and nonnegative')
        self.tracker = ScoreTracker()

    @staticmethod
    def _ammo(observation):
        if observation is None:
            return None
        vitals = np.asarray(observation['vitals'])
        if vitals.ndim != 1 or len(vitals) < 8 or not np.isfinite(vitals[4:8]).all():
            raise ValueError('observation vitals must contain finite ammo indices 4:8')
        return vitals[4:8].copy()

    def reset(self, state, observation=None):
        self.tracker.reset(state)
        self.prev = self.tracker.result()
        self.items = int(state.get('items', 0))
        self.health = float(state['health'])
        self.ammo = self._ammo(observation)

    def __call__(self, state, goal, observation=None):
        if not isinstance(goal, (int, np.integer)) or not 0 <= goal < 5:
            raise ValueError('goal must be an integer in [0, 5)')
        ammo = self._ammo(observation)
        health = float(state['health'])
        ammo_pickups = (int(np.count_nonzero(ammo > self.ammo))
                        if ammo is not None and self.ammo is not None else 0)
        inventory_pickups = int(health > self.health) + ammo_pickups
        self.tracker.update(state)
        now, prev = self.tracker.result(), self.prev
        items = int(state.get('items', 0))
        events = (
            max(now.areas - prev.areas, 0),
            max(now.kills - prev.kills, 0),
            max(items - self.items, now.keys - prev.keys, inventory_pickups, 0),
            max(now.doors - prev.doors, 0) + max(now.locked_doors - prev.locked_doors, 0),
            int(now.exit and not prev.exit),
        )
        self.prev, self.items = now, items
        self.health, self.ammo = health, ammo
        weight = self.weights[('explore', 'fight', 'collect', 'landmark', 'exit')[goal]]
        return float(weight * events[goal])
