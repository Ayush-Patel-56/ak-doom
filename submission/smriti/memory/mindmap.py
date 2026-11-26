"""Mind map: an egocentric-readable occupancy sketch built from the odometer pose and the 64 rays.

docs/03-SYSTEM-DESIGN.md §3.3.2. Pure numpy, no environment or privileged state.

- Grid anchored at the spawn point (the odometer's frame: +x initially forward, +y initially left),
  GRID x GRID cells of CELL map units. Positions that fall outside the grid are simply not drawn.
- Layers: seen-free, wall, door, locked door, item, enemy-last-seen, visited count, visit recency.
- write(): free space along every ray, the hit type at its end, our own cell as visited.
- crop(): a CROP x CROP window around the agent, rotated so "up" is the facing direction, as uint8
  (C, CROP, CROP). The network's small CNN reads it (smriti/brain.py MapEncoder).

This is observation preprocessing only. There is no path search of any kind on this grid: the network
decides where to go.
"""

from __future__ import annotations

import math

import numpy as np

GRID = 256  # cells per side (8,192 map units at CELL=32, centred on spawn)
CELL = 32.0  # map units per cell
CROP = 32  # crop window cells per side (1,024 units across, the ray range)
RAY_RANGE = 1024.0
NUM_RAYS = 64
CHANNELS = ("free", "wall", "door", "locked_door", "item", "enemy", "visited", "recent")
NUM_CHANNELS = len(CHANNELS)
FREE, WALL, DOOR, LOCKED, ITEM, ENEMY, VISITED, RECENT = range(NUM_CHANNELS)
# ray hit_type (docs/CONTRACT.md §2.2) -> layer: 0 none, 1 wall, 2 door, 3 locked door, 4 enemy, 5 item
HIT_LAYER = {1: WALL, 2: DOOR, 3: LOCKED, 4: ENEMY, 5: ITEM}
ENEMY_DECAY_STEPS = 35.0  # enemy marks fade over ~5 s of game time (35 decisions)
RECENT_DECAY_STEPS = 350.0  # visit recency fades over ~50 s
VISIT_SATURATION = math.log1p(50.0)
SAMPLE_SPACING = CELL / 2  # free-space samples along a ray, half a cell apart
_SAMPLES = np.arange(SAMPLE_SPACING, RAY_RANGE, SAMPLE_SPACING, dtype=np.float32)  # (S,)
_RAY_ANGLES = np.arange(NUM_RAYS, dtype=np.float64) * (2 * math.pi / NUM_RAYS)  # index 0 ahead, CCW


def _crop_offsets() -> tuple[np.ndarray, np.ndarray]:
    """Agent-frame sample points: row 0 is farthest ahead, column 0 is leftmost.

    Each crop pixel takes the MAX over a 2 x 2 grid of sub-samples (a quarter cell from its centre), so
    a rotated crop never skips a one-cell feature (a door or a wall corner) between pixel centres.
    """
    idx = (np.arange(CROP, dtype=np.float64) - CROP / 2 + 0.5) * CELL
    fwd, left = [], []
    for df in (-CELL / 4, CELL / 4):
        for dl in (-CELL / 4, CELL / 4):
            fwd.append((-idx[:, None] + df) * np.ones((1, CROP)))
            left.append((-idx[None, :] + dl) * np.ones((CROP, 1)))
    return np.stack([f.ravel() for f in fwd]), np.stack([g.ravel() for g in left])  # (4, CROP*CROP)


_CROP_FWD, _CROP_LEFT = _crop_offsets()


class MindMap:
    def __init__(self):
        # float32: counts and step stamps (stamp = step + 1, 0 = never)
        self.grid = np.zeros((NUM_CHANNELS, GRID, GRID), np.float32)
        self.reset()

    def reset(self) -> None:
        self.grid.fill(0.0)
        self.t = 0
        self.last_cell: tuple[int, int] | None = None
        self.cells_visited = 0

    @staticmethod
    def to_cell(x, y):
        """Spawn-frame map units -> (col, row) integer cell indices (may be outside the grid)."""
        return (np.floor(np.asarray(x) / CELL).astype(np.int64) + GRID // 2,
                np.floor(np.asarray(y) / CELL).astype(np.int64) + GRID // 2)

    def write(self, x: float, y: float, heading_deg: float, rays: dict) -> None:
        """Draw one cleaned observation's rays seen from pose (x, y, heading)."""
        self.t += 1
        stamp = float(self.t)
        g = self.grid
        dist = np.clip(np.asarray(rays["distance"], np.float64), 0.0, 1.0) * RAY_RANGE  # (64,)
        hit = np.asarray(rays["hit_type"], np.int64)
        ang = math.radians(heading_deg) + _RAY_ANGLES
        c, s = np.cos(ang), np.sin(ang)
        # Free space: samples strictly before the hit (keep half a cell clear of the obstacle).
        free = _SAMPLES[None, :] < (dist[:, None] - SAMPLE_SPACING)  # (64, S)
        fx = x + c[:, None] * _SAMPLES[None, :]
        fy = y + s[:, None] * _SAMPLES[None, :]
        cx, cy = self.to_cell(fx[free], fy[free])
        ok = (cx >= 0) & (cx < GRID) & (cy >= 0) & (cy < GRID)
        cx, cy = cx[ok], cy[ok]
        g[FREE, cy, cx] = 1.0
        # A door we can now see through is open; a pickup we see through was taken.
        g[DOOR, cy, cx] = 0.0
        g[LOCKED, cy, cx] = 0.0
        g[ITEM, cy, cx] = 0.0
        # Hit points (a few units past the surface so they land in the obstacle's cell).
        hit_mask = (hit > 0) & (dist < RAY_RANGE - 1e-3)
        if hit_mask.any():
            d = dist[hit_mask] + 4.0
            hx, hy = self.to_cell(x + c[hit_mask] * d, y + s[hit_mask] * d)
            layers = hit[hit_mask]
            ok = (hx >= 0) & (hx < GRID) & (hy >= 0) & (hy < GRID)
            for h_type, layer in HIT_LAYER.items():
                sel = ok & (layers == h_type)
                if sel.any():
                    g[layer, hy[sel], hx[sel]] = stamp if layer == ENEMY else 1.0
        # Our own cell: breadcrumbs.
        ax, ay = self.to_cell(x, y)
        ax, ay = int(ax), int(ay)
        if 0 <= ax < GRID and 0 <= ay < GRID:
            if g[VISITED, ay, ax] == 0:
                self.cells_visited += 1
            g[VISITED, ay, ax] += 1.0
            g[RECENT, ay, ax] = stamp
            g[FREE, ay, ax] = 1.0
            self.last_cell = (ax, ay)

    def tick(self) -> None:
        """Advance time without drawing (pose unknown this step)."""
        self.t += 1

    def crop(self, x: float, y: float, heading_deg: float) -> np.ndarray:
        """(C, CROP, CROP) uint8 window, rotated so up = facing direction. Values 0..255."""
        h = math.radians(heading_deg)
        ch, sh = math.cos(h), math.sin(h)
        wx = x + _CROP_FWD * ch - _CROP_LEFT * sh
        wy = y + _CROP_FWD * sh + _CROP_LEFT * ch
        cx, cy = self.to_cell(wx, wy)  # (4, CROP*CROP)
        ok = (cx >= 0) & (cx < GRID) & (cy >= 0) & (cy < GRID)
        vals = np.zeros((NUM_CHANNELS,) + cx.shape, np.float32)
        vals[:, ok] = self.grid[:, cy[ok], cx[ok]]
        vals = vals.max(axis=1)  # (C, CROP*CROP): max over the 2 x 2 sub-samples
        out = vals.copy()
        t = float(self.t)
        for layer, tau in ((ENEMY, ENEMY_DECAY_STEPS), (RECENT, RECENT_DECAY_STEPS)):
            stamp = vals[layer]
            seen = stamp > 0
            out[layer] = np.where(seen, np.exp(-(t - stamp) / tau), 0.0)
        out[VISITED] = np.minimum(np.log1p(vals[VISITED]) / VISIT_SATURATION, 1.0)
        return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(NUM_CHANNELS, CROP, CROP)
