"""Training rewards from env.privileged_state() (docs/04-TRAINING-PLAN.md §4.2).

Mirrors our scorer (training/scorer.py ScoreTracker, which itself mirrors the published rules) so
"reward" and "score" can never disagree about what counts as a key, a door or a new area:
the per-step reward is the CHANGE in each scorer counter times a training weight, plus a few helpers.
privileged_state is ground truth for the training signal only; the agent never sees it.
"""

from __future__ import annotations

import numpy as np

from training.scorer import ScoreTracker

WEIGHTS = {
    "exit": 10.0, "exit_health": 1.0, "exit_time": 1.0,  # + bonus for health and time left
    "key": 3.0, "locked_door": 3.0, "door": 1.0,
    "area": 0.05,  # per new 128-unit scorer cell
    "kill": 1.0,
    "damage_per_hp": -0.02,
    "death": -5.0,
    "step": -0.001,
    "stuck": -0.1, "stuck_steps": 70, "stuck_radius": 32.0,  # barely moved for ~10 s of decisions
    "use_spam": -0.05,  # USE pressed again within 4 decisions (doors reverse when spammed)
}


class RewardShaper:
    def __init__(self, weights: dict | None = None):
        self.w = dict(WEIGHTS, **(weights or {}))
        self.tracker = ScoreTracker()

    def reset(self, state: dict) -> None:
        self.tracker.reset(state)
        self.prev = self.tracker.result()
        self.anchor = (state["x"], state["y"])
        self.since_moved = 0
        self.since_use = 99
        self.total = 0.0

    def __call__(self, state: dict, action: np.ndarray) -> float:
        w = self.w
        self.tracker.update(state)
        now = self.tracker.result()
        p = self.prev
        r = w["step"]
        r += w["key"] * max(now.keys - p.keys, 0)
        r += w["locked_door"] * max(now.locked_doors - p.locked_doors, 0)
        r += w["door"] * max(now.doors - p.doors, 0)
        r += w["area"] * max(now.areas - p.areas, 0)
        r += w["kill"] * max(now.kills - p.kills, 0)
        r += w["damage_per_hp"] * max(now.damage - p.damage, 0.0)
        cap = max(now.episode_ticks, 1)
        time_left = max(0.0, 1.0 - now.tic / cap)
        if now.exit and not p.exit:
            r += w["exit"] + w["exit_health"] * min(max(now.health, 0.0) / 100.0, 1.0) + w["exit_time"] * time_left
        if now.dead and not p.dead:
            r += w["death"] * (0.5 + 0.5 * time_left)  # dying early costs more, like the scorer
        # Stuck: barely moved for a while.
        dx, dy = state["x"] - self.anchor[0], state["y"] - self.anchor[1]
        if dx * dx + dy * dy > w["stuck_radius"] ** 2:
            self.anchor, self.since_moved = (state["x"], state["y"]), 0
        else:
            self.since_moved += 1
            if self.since_moved >= w["stuck_steps"]:
                r += w["stuck"]
                self.since_moved = 0
        # USE re-pressed while a door may still be moving.
        if action[7]:
            if self.since_use < 4:
                r += w["use_spam"]
            self.since_use = 0
        else:
            self.since_use += 1
        self.prev = now
        self.total += r
        return float(r)

    def stats(self) -> dict:
        res = self.tracker.result()
        return {"exit": int(res.exit), "keys": res.keys, "locked_doors": res.locked_doors, "doors": res.doors,
                "areas": res.areas, "kills": res.kills, "dead": int(res.dead), "score": res.score}
