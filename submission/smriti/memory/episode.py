"""EpisodeMemory: odometer + mind map + landmark notebook, updated once per decision.

The SAME class runs inside DoomAgent (evaluation) and inside every training game
(training/rollout.py), so memory features can never differ between training and evaluation.

Call order per decision (mirrors docs/03-SYSTEM-DESIGN.md §3.3.1):
    memory.observe(previous_executed_action_or_None, raw_observation, cleaned_observation)
    feats = memory.features()            # what the network reads this step
Pass None as the action for the first observation of an episode (the reset observation is not a
transition). Never raises on bad data: odometry is marked invalid until reset and the map/notebook
stop drawing (they keep their last picture instead of drawing at a wrong position).
"""

from __future__ import annotations

import numpy as np

from .mindmap import CROP, NUM_CHANNELS, MindMap
from .notebook import NOTE_FEATURES, NUM_SLOTS, Notebook
from .odometer import Odometer, Pose


def empty_features() -> dict[str, np.ndarray]:
    return {
        "map": np.zeros((NUM_CHANNELS, CROP, CROP), np.uint8),
        "notes": np.zeros((NUM_SLOTS, NOTE_FEATURES), np.float32),
        "note_mask": np.zeros(NUM_SLOTS, np.bool_),
    }


class EpisodeMemory:
    def __init__(self, odometer: Odometer | None, use_mindmap: bool = True, use_notebook: bool = True):
        self.odometer = odometer
        self.use_mindmap = use_mindmap
        self.use_notebook = use_notebook
        self.mindmap = MindMap() if use_mindmap else None
        self.notebook = Notebook() if use_notebook else None
        self.reset()

    def reset(self) -> None:
        if self.odometer is not None:
            self.odometer.reset()
        if self.mindmap is not None:
            self.mindmap.reset()
        if self.notebook is not None:
            self.notebook.reset()
        self.valid = self.odometer is not None
        self.error: str | None = None if self.odometer is not None else "no odometer"

    @property
    def pose(self) -> Pose:
        return self.odometer.pose if self.odometer is not None else Pose()

    def observe(self, executed_action, raw_obs, clean_obs) -> None:
        if self.odometer is None:
            return
        if executed_action is not None and self.valid:
            try:
                self.odometer.update(executed_action, raw_obs)
            except ValueError as exc:
                self.valid = False
                self.error = str(exc)
        if not (self.mindmap or self.notebook):
            return
        if not self.valid:
            for part in (self.mindmap, self.notebook):
                if part is not None:
                    part.tick()
            return
        p = self.odometer.pose
        if self.mindmap is not None:
            self.mindmap.write(p.x, p.y, p.heading_deg, clean_obs["rays"])
        if self.notebook is not None:
            self.notebook.write(p.x, p.y, p.heading_deg, clean_obs)

    def features(self) -> dict[str, np.ndarray]:
        out = empty_features()
        if self.odometer is None:
            return out
        p = self.odometer.pose
        if self.mindmap is not None:
            out["map"] = self.mindmap.crop(p.x, p.y, p.heading_deg)
        if self.notebook is not None:
            out["notes"], out["note_mask"] = self.notebook.features(p.x, p.y, p.heading_deg)
        return out
