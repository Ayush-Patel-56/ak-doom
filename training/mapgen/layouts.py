"""Procedural layouts: seed -> GridMap. Every layout is reproducible from (kind, seed).

| kind    | tier | what it trains                                                |
|---------|------|---------------------------------------------------------------|
| arena   | 1    | combat & survival in one room with pillars, exit pad far away |
| maze    | 2    | pure navigation and memory in 1-cell corridors                |
| rooms   | 2    | rooms joined by corridors with doors (USE), exit in far room  |
| keydoor | 3    | find the keycard, unlock the door to the exit room            |
"""

from __future__ import annotations

import random
from collections import deque

from .udmf import GridMap

KINDS = {"arena": 1, "maze": 2, "rooms": 2, "keydoor": 3}
MONSTERS = "zzzsid"  # weighted: zombiemen are most common
KEYS = (("r", "R"), ("b", "B"), ("y", "Y"))
ANGLES = (0, 90, 180, 270)


class Canvas:
    def __init__(self, height: int, width: int):
        self.g = [["#"] * width for _ in range(height)]

    def __getitem__(self, rc):
        r, c = rc
        return self.g[r][c]

    def __setitem__(self, rc, ch):
        r, c = rc
        self.g[r][c] = ch

    @property
    def size(self):
        return len(self.g), len(self.g[0])

    def floor_cells(self):
        return [(r, c) for r, row in enumerate(self.g) for c, ch in enumerate(row) if ch == "."]

    def carve(self, r0, c0, r1, c1):
        for r in range(min(r0, r1), max(r0, r1) + 1):
            for c in range(min(c0, c1), max(c0, c1) + 1):
                self.g[r][c] = "."

    def text(self) -> str:
        return "\n".join("".join(row) for row in self.g)


def bfs(canvas: Canvas, start, passable: str = ".PrbyXzsidhaAS") -> dict:
    dist = {start: 0}
    q = deque([start])
    h, w = canvas.size
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist and canvas[nr, nc] in passable:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def scatter(canvas: Canvas, rng: random.Random, chars: str, count: int, away_from=None, min_dist: int = 4):
    cells = canvas.floor_cells()
    if away_from is not None:
        far = bfs(canvas, away_from)
        cells = [c for c in cells if far.get(c, 0) >= min_dist]
    rng.shuffle(cells)
    for cell in cells[:count]:
        canvas[cell] = rng.choice(chars)


# ----------------------------------------------------------------------------- arena (tier 1)
def arena(seed: int) -> GridMap:
    rng = random.Random(seed)
    h, w = rng.randint(10, 16), rng.randint(12, 20)
    cv = Canvas(h, w)
    cv.carve(1, 1, h - 2, w - 2)
    for _ in range(rng.randint(0, (h * w) // 30)):  # pillars for cover
        r, c = rng.randint(3, h - 4), rng.randint(3, w - 4)
        cv[r, c] = "#"
    start = (rng.randint(1, h - 2), 1)
    cv[start] = "P"
    reach = bfs(cv, start)
    exit_cell = max(reach, key=reach.get)
    cv[exit_cell] = "X"
    scatter(cv, rng, MONSTERS, rng.randint(3, 8), away_from=start, min_dist=5)
    scatter(cv, rng, "hhaA", rng.randint(1, 4))
    return GridMap.parse(cv.text(), player_angle=0, notes={"kind": "arena", "tier": 1, "seed": seed})


# ----------------------------------------------------------------------------- maze (tier 2)
def maze(seed: int) -> GridMap:
    rng = random.Random(seed)
    rows, cols = rng.randint(5, 9), rng.randint(6, 11)  # maze cells; grid is 2x+1
    cv = Canvas(2 * rows + 1, 2 * cols + 1)
    seen = {(0, 0)}
    stack = [(0, 0)]
    cv[1, 1] = "."
    while stack:
        r, c = stack[-1]
        options = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr, c + dc) not in seen]
        if not options:
            stack.pop()
            continue
        nr, nc = rng.choice(options)
        cv[2 * r + 1 + (nr - r), 2 * c + 1 + (nc - c)] = "."
        cv[2 * nr + 1, 2 * nc + 1] = "."
        seen.add((nr, nc))
        stack.append((nr, nc))
    for _ in range(rng.randint(0, rows)):  # a few loops so there is more than one way
        r, c = rng.randint(1, 2 * rows - 1), rng.randint(1, 2 * cols - 1)
        if cv[r, c] == "#" and (r % 2) != (c % 2):
            cv[r, c] = "."
    start = (1, 1)
    cv[start] = "P"
    reach = bfs(cv, start)
    cv[max(reach, key=reach.get)] = "X"
    scatter(cv, rng, "zz", rng.randint(0, 3), away_from=start, min_dist=8)
    return GridMap.parse(cv.text(), player_angle=rng.choice(ANGLES), notes={"kind": "maze", "tier": 2, "seed": seed})


# ----------------------------------------------------------------------------- rooms + doors (tiers 2-3)
def _room_layout(rng: random.Random, grid_rows: int, grid_cols: int, slot: int = 8):
    """Rooms in a grid of slots, linked by a random spanning tree (+ a few extra links)."""
    cv = Canvas(grid_rows * slot + 1, grid_cols * slot + 1)
    rooms = {}
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            rh, rw = rng.randint(3, slot - 3), rng.randint(3, slot - 3)
            r0 = gr * slot + 1 + rng.randint(0, slot - 2 - rh)
            c0 = gc * slot + 1 + rng.randint(0, slot - 2 - rw)
            cv.carve(r0, c0, r0 + rh - 1, c0 + rw - 1)
            rooms[(gr, gc)] = (r0, c0, r0 + rh - 1, c0 + rw - 1)
    nodes = list(rooms)
    rng.shuffle(nodes)
    in_tree, edges = {nodes[0]}, []
    while len(in_tree) < len(nodes):
        a = rng.choice(sorted(in_tree))
        nbrs = [(a[0] + dr, a[1] + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nbrs = [n for n in nbrs if n in rooms and n not in in_tree]
        if nbrs:
            b = rng.choice(nbrs)
            in_tree.add(b)
            edges.append((a, b))
    return cv, rooms, edges


def _connect(cv: Canvas, rooms, a, b, door: str | None):
    """Straight-ish corridor between rooms a and b, with an optional door next to room b."""
    ra, rb = rooms[a], rooms[b]
    if a[0] == b[0]:  # same slot row: horizontal corridor
        lo, hi = max(ra[0], rb[0]), min(ra[2], rb[2])
        row = (lo + hi) // 2 if lo <= hi else (ra[0] + ra[2]) // 2
        left, right = (ra, rb) if ra[1] < rb[1] else (rb, ra)
        cells = [(row, c) for c in range(left[3] + 1, right[1])]
        if lo > hi:  # rooms don't overlap vertically: add a vertical leg inside the corridor end
            r_end = (rb[0] + rb[2]) // 2
            cells += [(r, right[1] - 1) for r in range(min(row, r_end), max(row, r_end) + 1)]
    else:  # vertical corridor
        lo, hi = max(ra[1], rb[1]), min(ra[3], rb[3])
        col = (lo + hi) // 2 if lo <= hi else (ra[1] + ra[3]) // 2
        top, bottom = (ra, rb) if ra[0] < rb[0] else (rb, ra)
        cells = [(r, col) for r in range(top[2] + 1, bottom[0])]
        if lo > hi:
            c_end = (rb[1] + rb[3]) // 2
            cells += [(bottom[0] - 1, c) for c in range(min(col, c_end), max(col, c_end) + 1)]
    for cell in cells:
        if cv[cell] == "#":
            cv[cell] = "."
    if door and cells:
        _place_door(cv, cells, door)


def _place_door(cv: Canvas, cells, door: str) -> None:
    h, w = cv.size
    for r, c in cells:  # first corridor cell with walls on both sides across the corridor
        ns = [cv[r + d, c] if 0 <= r + d < h else "#" for d in (-1, 1)]
        ew = [cv[r, c + d] if 0 <= c + d < w else "#" for d in (-1, 1)]
        if (ns == ["#", "#"] and "#" not in ew) or (ew == ["#", "#"] and "#" not in ns):
            cv[r, c] = door
            return


def _room_center(room):
    return ((room[0] + room[2]) // 2, (room[1] + room[3]) // 2)


def rooms(seed: int) -> GridMap:
    rng = random.Random(seed)
    cv, rms, edges = _room_layout(rng, rng.randint(2, 3), rng.randint(2, 4))
    for a, b in edges:
        _connect(cv, rms, a, b, "D" if rng.random() < 0.6 else None)
    start_room = rng.choice(sorted(rms))
    start = _room_center(rms[start_room])
    cv[start] = "P"
    reach = bfs(cv, start, passable=".PXDzsidhaAS")
    far_room = max(rms, key=lambda k: reach.get(_room_center(rms[k]), -1))
    cv[_room_center(rms[far_room])] = "X"
    scatter(cv, rng, MONSTERS, rng.randint(2, 6), away_from=start, min_dist=6)
    scatter(cv, rng, "hha", rng.randint(1, 3))
    return GridMap.parse(cv.text(), player_angle=rng.choice(ANGLES), notes={"kind": "rooms", "tier": 2, "seed": seed})


def keydoor(seed: int) -> GridMap:
    """Exit room is a leaf behind a locked door; the key is in the farthest other room."""
    rng = random.Random(seed)
    for _ in range(50):  # retry until the tree has a usable leaf for the exit
        cv, rms, edges = _room_layout(rng, rng.randint(2, 3), rng.randint(2, 4))
        start_room = rng.choice(sorted(rms))
        adj = {k: [] for k in rms}
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        depth, parent, q = {start_room: 0}, {start_room: None}, deque([start_room])
        while q:
            k = q.popleft()
            for n in adj[k]:
                if n not in depth:
                    depth[n], parent[n] = depth[k] + 1, k
                    q.append(n)
        leaves = [k for k in rms if k != start_room and len(adj[k]) == 1]
        if leaves and len(rms) >= 3:
            break
    exit_room = max(leaves, key=lambda k: (depth[k], rng.random()))
    key_char, lock_char = rng.choice(KEYS)
    for a, b in edges:
        if {a, b} == {exit_room, parent[exit_room]}:
            _connect(cv, rms, parent[exit_room], exit_room, lock_char)
        else:
            _connect(cv, rms, a, b, "D" if rng.random() < 0.5 else None)
    start = _room_center(rms[start_room])
    cv[start] = "P"
    cv[_room_center(rms[exit_room])] = "X"
    others = [k for k in rms if k not in (start_room, exit_room)]
    # Prefer a key room that is NOT next to the locked door, so the agent must remember the door
    # and backtrack (the Tier 3 skill); fall back to the deepest room when there is no choice.
    key_room = max(others, key=lambda k: (k != parent[exit_room], depth[k], rng.random()))
    kr = rms[key_room]
    cv[(rng.randint(kr[0], kr[2]), rng.randint(kr[1], kr[3]))] = key_char
    scatter(cv, rng, MONSTERS, rng.randint(2, 6), away_from=start, min_dist=6)
    scatter(cv, rng, "hha", rng.randint(1, 3))
    return GridMap.parse(cv.text(), player_angle=rng.choice(ANGLES),
                         notes={"kind": "keydoor", "tier": 3, "seed": seed, "key": key_char})


GENERATORS = {"arena": arena, "maze": maze, "rooms": rooms, "keydoor": keydoor}


def generate(kind: str, seed: int) -> GridMap:
    return GENERATORS[kind](seed)
