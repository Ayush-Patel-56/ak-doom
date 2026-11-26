"""Landmark notebook: a fixed number of slots for keys, locked doors and doors, in the spawn frame.

docs/03-SYSTEM-DESIGN.md §3.3.3. Pure numpy, no environment or privileged state.

Writing (from cleaned observations + odometer pose):
- a ray that ends on a door / locked door -> that point is a door / locked-door landmark;
- a visible entity of type `key` -> key landmark at its estimated position;
- a key flag in vitals flipping 0 -> 1 -> the nearest key landmark is marked done (picked up);
- a door landmark that rays now pass through -> marked done (open); seen closed again -> not done.
Nearby sightings of the same type update one slot (running mean); when all slots are full the
least-recently-seen slot (done ones first) is replaced.

Reading: features() returns, per slot, the landmark's type, distance and direction RELATIVE TO THE
AGENT NOW, how long ago it was seen and whether it is done. The network attends over these slots with
a query from its GRU state (smriti/brain.py NotebookReader). Nothing here chooses actions.
"""

from __future__ import annotations

import math

import numpy as np

NUM_SLOTS = 16
TYPES = ("key", "locked_door", "door")
KEY, LOCKED_DOOR, DOOR = 1, 2, 3  # 0 = empty slot
NOTE_FEATURES = 9  # one-hot type (3), distance, sin, cos, closeness, age, done
MERGE_RADIUS = {KEY: 96.0, LOCKED_DOOR: 128.0, DOOR: 128.0}
PICKUP_RADIUS = 192.0  # a key flag flip is matched to a key landmark this close to us
RAY_RANGE = 1024.0
NUM_RAYS = 64
RAY_STEP = 2 * math.pi / NUM_RAYS
ENTITY_KEY = 9  # doom_env.spaces.ENTITY_TYPES index of "key"
HIT_DOOR, HIT_LOCKED = 2, 3
KEY_FLAGS = slice(8, 14)  # vitals key_* flags (docs/CONTRACT.md §2.1)
OPEN_MARGIN = 64.0  # a ray passing this far beyond a door landmark means it is open
AGE_SCALE = 500.0
MAX_WEIGHT = 20.0  # running-mean weight cap so a landmark can still drift to a better estimate


class Notebook:
    def __init__(self, slots: int = NUM_SLOTS):
        self.n = slots
        self.kind = np.zeros(slots, np.int64)
        self.pos = np.zeros((slots, 2), np.float64)
        self.weight = np.zeros(slots, np.float64)
        self.last_seen = np.zeros(slots, np.float64)
        self.done = np.zeros(slots, np.float64)
        self.reset()

    def reset(self) -> None:
        self.kind.fill(0)
        self.pos.fill(0.0)
        self.weight.fill(0.0)
        self.last_seen.fill(0.0)
        self.done.fill(0.0)
        self.t = 0
        self.keys_held = None  # previous key flags; None until the first observation
        self.pickups = 0

    # ----------------------------------------------------------------------------- writing
    def write(self, x: float, y: float, heading_deg: float, obs: dict) -> None:
        self.t += 1
        h = math.radians(heading_deg)
        rays = obs["rays"]
        dist = np.clip(np.asarray(rays["distance"], np.float64), 0.0, 1.0) * RAY_RANGE
        hit = np.asarray(rays["hit_type"], np.int64)
        ang = h + np.arange(NUM_RAYS) * RAY_STEP
        # 1) door / locked-door sightings (one per contiguous run of rays, to avoid 10 slots per door)
        for hit_type, kind in ((HIT_LOCKED, LOCKED_DOOR), (HIT_DOOR, DOOR)):
            idx = np.nonzero((hit == hit_type) & (dist < RAY_RANGE - 1e-3))[0]
            for run in _runs(idx):
                d = dist[run] + 4.0
                px = float(np.mean(x + np.cos(ang[run]) * d))
                py = float(np.mean(y + np.sin(ang[run]) * d))
                self._sight(kind, px, py, done=0.0)
        # 2) visible keys
        ents = obs["visible_entities"]
        mask = np.asarray(ents["mask"]).astype(bool)
        etype = np.asarray(ents["entity_type"], np.int64)
        feats = np.asarray(ents["features"], np.float64)
        for i in np.nonzero(mask & (etype == ENTITY_KEY))[0]:
            d, sr, cr = feats[i, 0] * 1024.0, feats[i, 1], feats[i, 2]  # sin, cos of relative angle (left +)
            if not (math.isfinite(d) and math.isfinite(sr) and math.isfinite(cr)):
                continue
            ch, sh = math.cos(h), math.sin(h)  # rotate the agent-frame direction into the spawn frame
            self._sight(KEY, x + d * (ch * cr - sh * sr), y + d * (sh * cr + ch * sr), done=0.0)
        # 3) key pickups (flags only ever go 0 -> 1 within an episode)
        flags = np.asarray(obs["vitals"], np.float64)[KEY_FLAGS] > 0.5
        if self.keys_held is not None:
            new = int(np.sum(flags & ~self.keys_held))
            for _ in range(new):
                self._pickup(x, y)
        self.keys_held = flags
        # 4) doors seen open / closed again
        self._update_doors(x, y, ang, dist, hit)

    def tick(self) -> None:
        self.t += 1

    def _sight(self, kind: int, px: float, py: float, done: float) -> None:
        live = np.nonzero(self.kind == kind)[0]
        if live.size:
            d = np.hypot(self.pos[live, 0] - px, self.pos[live, 1] - py)
            j = int(np.argmin(d))
            if d[j] <= MERGE_RADIUS[kind]:
                s = int(live[j])
                if kind == KEY and self.done[s] > 0:
                    return  # a picked-up key cannot be seen again; ignore a stale duplicate
                w = self.weight[s]
                self.pos[s] = (self.pos[s] * w + (px, py)) / (w + 1)
                self.weight[s] = min(w + 1, MAX_WEIGHT)
                self.last_seen[s] = self.t
                if kind != KEY:
                    self.done[s] = done
                return
        s = self._free_slot()
        self.kind[s], self.pos[s], self.weight[s] = kind, (px, py), 1.0
        self.last_seen[s], self.done[s] = self.t, done

    def _free_slot(self) -> int:
        empty = np.nonzero(self.kind == 0)[0]
        if empty.size:
            return int(empty[0])
        # Replace the least recently seen, preferring done landmarks (they matter least).
        order = np.lexsort((self.last_seen, -self.done))
        return int(order[0])

    def _pickup(self, x: float, y: float) -> None:
        self.pickups += 1
        keys = np.nonzero((self.kind == KEY) & (self.done == 0))[0]
        if keys.size:
            d = np.hypot(self.pos[keys, 0] - x, self.pos[keys, 1] - y)
            j = int(np.argmin(d))
            if d[j] <= PICKUP_RADIUS:
                s = int(keys[j])
                self.done[s], self.last_seen[s], self.pos[s] = 1.0, self.t, (x, y)
                return
        s = self._free_slot()  # never saw it (e.g. picked up while out of view): note it here
        self.kind[s], self.pos[s], self.weight[s] = KEY, (x, y), 1.0
        self.last_seen[s], self.done[s] = self.t, 1.0

    def _update_doors(self, x, y, ang, dist, hit) -> None:
        doors = np.nonzero((self.kind == DOOR) | (self.kind == LOCKED_DOOR))[0]
        for s in doors:
            dx, dy = self.pos[s, 0] - x, self.pos[s, 1] - y
            d = math.hypot(dx, dy)
            if d < 1.0 or d > RAY_RANGE - 2 * OPEN_MARGIN:
                continue
            k = int(np.argmax(np.cos(ang) * dx + np.sin(ang) * dy))  # the ray pointing most at it
            if dist[k] > d + OPEN_MARGIN:
                self.done[s] = 1.0  # we can see past it: open
                self.last_seen[s] = self.t
            elif hit[k] in (HIT_DOOR, HIT_LOCKED) and abs(dist[k] - d) < OPEN_MARGIN:
                self.done[s] = 0.0  # closed again

    # ----------------------------------------------------------------------------- reading
    def features(self, x: float, y: float, heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """(NUM_SLOTS, NOTE_FEATURES) float32 egocentric slot features and (NUM_SLOTS,) bool mask."""
        out = np.zeros((self.n, NOTE_FEATURES), np.float32)
        mask = self.kind > 0
        if not mask.any():
            return out, mask
        h = math.radians(heading_deg)
        dx, dy = self.pos[:, 0] - x, self.pos[:, 1] - y
        d = np.hypot(dx, dy)
        fwd = dx * math.cos(h) + dy * math.sin(h)  # landmark in the agent frame (+left)
        left = -dx * math.sin(h) + dy * math.cos(h)
        safe = np.maximum(d, 1e-6)
        for kind in (KEY, LOCKED_DOOR, DOOR):
            out[:, kind - 1] = self.kind == kind
        out[:, 3] = np.minimum(d, 4096.0) / 1024.0
        out[:, 4] = np.where(d > 1e-6, left / safe, 0.0)
        out[:, 5] = np.where(d > 1e-6, fwd / safe, 1.0)
        out[:, 6] = np.exp(-d / 128.0)
        out[:, 7] = np.minimum(self.t - self.last_seen, 2 * AGE_SCALE) / AGE_SCALE
        out[:, 8] = self.done
        out[~mask] = 0.0
        return out, mask


def _runs(idx: np.ndarray):
    """Split sorted ray indices into contiguous runs, joining the 63 -> 0 wrap-around."""
    if idx.size == 0:
        return []
    runs, cur = [], [int(idx[0])]
    for i in idx[1:]:
        if int(i) == cur[-1] + 1:
            cur.append(int(i))
        else:
            runs.append(cur)
            cur = [int(i)]
    runs.append(cur)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == NUM_RAYS - 1:
        runs[0] = runs.pop() + runs[0]
    return [np.array(r) for r in runs]
