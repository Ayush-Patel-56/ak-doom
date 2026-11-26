"""Procedural layouts: seed -> GridMap. Every layout is reproducible from (kind, seed).

| kind    | tier | what it trains                                                |
|---------|------|---------------------------------------------------------------|
| arena   | 1    | combat & survival in one room with pillars, exit pad far away |
| maze    | 2    | pure navigation and memory in 1-cell corridors                |
| rooms   | 2    | rooms joined by corridors with doors (USE), exit in far room  |
| keydoor | 3    | find the keycard, unlock the door to the exit room            |
| keynear | 3    | easy key puzzle (curriculum Stage 3 warm-up): key near spawn  |
"""

from __future__ import annotations

import random
from collections import deque

from .udmf import LOCKED, GridMap, validate

KINDS = {"arena": 1, "maze": 2, "rooms": 2, "keydoor": 3, "keynear": 3}
MONSTERS = "zzzsid"  # weighted: zombiemen are most common
KEYS = (("r", "R"), ("b", "B"), ("y", "Y"))
KEY_FOR = {"R": "r", "B": "b", "Y": "y"}
ANGLES = (0, 90, 180, 270)
# Floor, start, exit, open/locked doors, keys, monsters and items. Used so distance
# and placement can cross a whole map, not just the start room.
THROUGH_DOORS = ".PXDRBYrbyzsidhaAS"
MAX_ATTEMPTS = 64


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


def bfs(canvas: Canvas, start, passable: str = THROUGH_DOORS) -> dict:
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
    """Place up to `count` items on floor cells. Prefer cells at least min_dist from `away_from`,
    walking through open and locked doors, then relax the distance if there is not enough space."""
    if count <= 0:
        return
    floors = canvas.floor_cells()
    if away_from is None or not floors:
        cells = floors
    else:
        far = bfs(canvas, away_from, passable=THROUGH_DOORS)
        cells = []
        for dist in range(min_dist, -1, -1):
            cells = [c for c in floors if far.get(c, 0) >= dist and c != away_from]
            if len(cells) >= count:
                break
    rng.shuffle(cells)
    for cell in cells[:count]:
        canvas[cell] = rng.choice(chars)


# ----------------------------------------------------------------------------- solvability

def reachable(grid: GridMap, start, keys: set[str]) -> set:
    """Cells walkable from start, treating locked doors as open only when we hold the matching key."""
    seen, q = {start}, deque([start])
    while q:
        r, c = q.popleft()
        for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            ch = grid.at(*n)
            if n in seen or ch == "#" or (ch in LOCKED and KEY_FOR[ch] not in keys):
                continue
            seen.add(n)
            q.append(n)
    return seen


def solvable(grid: GridMap) -> tuple[bool, bool]:
    """(exit reachable with keys we can actually pick up, exit reachable with no keys)."""
    starts, exits = list(grid.cells("P")), list(grid.cells("X"))
    if len(starts) != 1 or not exits:
        return False, False
    start = starts[0]
    no_keys = reachable(grid, start, set())
    held = {grid.at(*c) for c in no_keys if grid.at(*c) in "rby"}
    with_keys = reachable(grid, start, held)
    return any(e in with_keys for e in exits), any(e in no_keys for e in exits)


def playable(grid: GridMap, kind: str) -> bool:
    """True if the layout is a sensible, solvable map of this kind."""
    try:
        validate(grid)
    except ValueError:
        return False
    starts, exits = list(grid.cells("P")), list(grid.cells("X"))
    if len(starts) != 1 or len(exits) != 1 or starts[0] == exits[0]:
        return False
    with_keys, without_keys = solvable(grid)
    if not with_keys:
        return False
    if kind == "keydoor":
        if without_keys or not any(grid.cells("rby")):
            return False
        if not any(grid.at(*c) in "rby" for c in reachable(grid, starts[0], set())):
            return False
    if kind == "keynear":  # same puzzle rules as keydoor
        return playable(grid, "keydoor")
    return True


def _attempt_seed(seed: int, attempt: int) -> int:
    """Deterministic sub-seed: attempt 0 is the public seed, later attempts are mixed from it."""
    if attempt == 0:
        return seed
    return seed * 1_000_003 + attempt * 2_000_029 + 17


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
    candidates = [c for c in reach if c != start]
    exit_cell = max(candidates, key=reach.get) if candidates else None
    if exit_cell is None:
        return GridMap.parse(cv.text(), player_angle=0, notes={"kind": "arena", "tier": 1, "seed": seed})
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
    candidates = [c for c in reach if c != start]
    if candidates:
        cv[max(candidates, key=reach.get)] = "X"
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
    """Corridor between adjacent slot-rooms. Uses a straight run when the rooms share a row or
    column, otherwise an L through the gap so both rooms always join. Optional door sits in the gap."""
    ra, rb = rooms[a], rooms[b]
    cells: list[tuple[int, int]] = []
    if a[0] == b[0]:  # same slot row: horizontal neighbours
        left, right = (ra, rb) if ra[1] < rb[1] else (rb, ra)
        r_left, r_right = (left[0] + left[2]) // 2, (right[0] + right[2]) // 2
        gap_lo, gap_hi = left[3] + 1, right[1] - 1
        mid_c = (gap_lo + gap_hi) // 2 if gap_lo <= gap_hi else left[3]
        for c in range(left[3] + 1, mid_c + 1):
            cells.append((r_left, c))
        for r in range(min(r_left, r_right), max(r_left, r_right) + 1):
            cells.append((r, mid_c))
        for c in range(mid_c, right[1]):
            cells.append((r_right, c))
    else:  # vertical neighbours
        top, bottom = (ra, rb) if ra[0] < rb[0] else (rb, ra)
        c_top, c_bot = (top[1] + top[3]) // 2, (bottom[1] + bottom[3]) // 2
        gap_lo, gap_hi = top[2] + 1, bottom[0] - 1
        mid_r = (gap_lo + gap_hi) // 2 if gap_lo <= gap_hi else top[2]
        for r in range(top[2] + 1, mid_r + 1):
            cells.append((r, c_top))
        for c in range(min(c_top, c_bot), max(c_top, c_bot) + 1):
            cells.append((mid_r, c))
        for r in range(mid_r, bottom[0]):
            cells.append((r, c_bot))
    for cell in cells:
        if cv[cell] == "#":
            cv[cell] = "."
    if door and cells:
        _place_door(cv, cells, door, rooms_ab=(ra, rb))


def _in_room(room, cell) -> bool:
    r, c = cell
    return room[0] <= r <= room[2] and room[1] <= c <= room[3]


def _place_door(cv: Canvas, cells, door: str, rooms_ab=None) -> bool:
    """Put the door on a 1-cell corridor cell in the gap (not inside either room)."""
    h, w = cv.size
    seen: list[tuple[int, int]] = []
    for r, c in cells:
        if (r, c) in seen:
            continue
        seen.append((r, c))
        if rooms_ab and any(_in_room(room, (r, c)) for room in rooms_ab):
            continue
        ns = [cv[r + d, c] if 0 <= r + d < h else "#" for d in (-1, 1)]
        ew = [cv[r, c + d] if 0 <= c + d < w else "#" for d in (-1, 1)]
        if (ns == ["#", "#"] and "#" not in ew) or (ew == ["#", "#"] and "#" not in ns):
            cv[r, c] = door
            return True
    return False


def _room_center(room):
    return ((room[0] + room[2]) // 2, (room[1] + room[3]) // 2)


def _place_start_and_exit(cv: Canvas, start, exit_cell) -> bool:
    """Write one P and one X on distinct reachable cells. Returns False if that is impossible."""
    if start == exit_cell or cv[start] == "#":
        return False
    cv[start] = "P"
    reach = bfs(cv, start, passable=THROUGH_DOORS)
    if exit_cell != start and exit_cell in reach and cv[exit_cell] != "P":
        cv[exit_cell] = "X"
        return True
    candidates = [c for c in reach if c != start]
    if not candidates:
        cv[start] = "."  # leave a clean failure for playable()
        return False
    cv[max(candidates, key=reach.get)] = "X"
    return True


def rooms(seed: int) -> GridMap:
    rng = random.Random(seed)
    cv, rms, edges = _room_layout(rng, rng.randint(2, 3), rng.randint(2, 4))
    for a, b in edges:
        _connect(cv, rms, a, b, "D" if rng.random() < 0.6 else None)
    start_room = rng.choice(sorted(rms))
    start = _room_center(rms[start_room])
    reach = bfs(cv, start, passable=THROUGH_DOORS)
    others = [k for k in rms if k != start_room and _room_center(rms[k]) in reach]
    if others:
        far_room = max(others, key=lambda k: reach.get(_room_center(rms[k]), -1))
        exit_cell = _room_center(rms[far_room])
    else:
        exit_cell = start  # _place_start_and_exit will pick the farthest floor cell instead
    if not _place_start_and_exit(cv, start, exit_cell):
        return GridMap.parse(cv.text(), player_angle=0, notes={"kind": "rooms", "tier": 2, "seed": seed})
    scatter(cv, rng, MONSTERS, rng.randint(2, 6), away_from=start, min_dist=6)
    scatter(cv, rng, "hha", rng.randint(1, 3))
    return GridMap.parse(cv.text(), player_angle=rng.choice(ANGLES), notes={"kind": "rooms", "tier": 2, "seed": seed})


def keydoor(seed: int) -> GridMap:
    """Exit room is a leaf behind a locked door; the key is in the farthest other room."""
    rng = random.Random(seed)
    leaves, rms, edges, cv, start_room, parent, depth, adj = [], {}, [], Canvas(3, 3), None, {}, {}, {}
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
    if not leaves or start_room is None:
        return GridMap.parse("###\n#P#\n###", notes={"kind": "keydoor", "tier": 3, "seed": seed})
    exit_room = max(leaves, key=lambda k: (depth[k], rng.random()))
    key_char, lock_char = rng.choice(KEYS)
    for a, b in edges:
        if {a, b} == {exit_room, parent[exit_room]}:
            _connect(cv, rms, parent[exit_room], exit_room, lock_char)
        else:
            _connect(cv, rms, a, b, "D" if rng.random() < 0.5 else None)
    start = _room_center(rms[start_room])
    if not _place_start_and_exit(cv, start, _room_center(rms[exit_room])):
        return GridMap.parse(cv.text(), player_angle=0, notes={"kind": "keydoor", "tier": 3, "seed": seed})
    others = [k for k in rms if k not in (start_room, exit_room)]
    # Prefer a key room that is NOT next to the locked door, so the agent must remember the door
    # and backtrack (the Tier 3 skill); fall back to the deepest room when there is no choice.
    if not others:
        return GridMap.parse(cv.text(), player_angle=0, notes={"kind": "keydoor", "tier": 3, "seed": seed})
    key_room = max(others, key=lambda k: (k != parent[exit_room], depth[k], rng.random()))
    kr = rms[key_room]
    key_cell = (rng.randint(kr[0], kr[2]), rng.randint(kr[1], kr[3]))
    if cv[key_cell] in "PX":
        floors = [(r, c) for r in range(kr[0], kr[2] + 1) for c in range(kr[1], kr[3] + 1) if cv[r, c] == "."]
        if not floors:
            return GridMap.parse(cv.text(), player_angle=0, notes={"kind": "keydoor", "tier": 3, "seed": seed})
        key_cell = rng.choice(floors)
    cv[key_cell] = key_char
    scatter(cv, rng, MONSTERS, rng.randint(2, 6), away_from=start, min_dist=6)
    scatter(cv, rng, "hha", rng.randint(1, 3))
    return GridMap.parse(cv.text(), player_angle=rng.choice(ANGLES),
                         notes={"kind": "keydoor", "tier": 3, "seed": seed, "key": key_char})


def keynear(seed: int) -> GridMap:
    """Curriculum Stage 3 warm-up (docs/04-TRAINING-PLAN.md §4.4: "key placed close at first").

    A small 2-4 room tree; the exit room is a leaf behind a locked door and the key lies in the start
    room or a room next to it, with at most two monsters. Same puzzle as keydoor, shorter backtrack.
    """
    rng = random.Random(seed)
    for _ in range(50):
        rows, cols = rng.choice(((1, 2), (1, 3), (2, 2)))
        if rows * cols < 2:
            continue
        cv, rms, edges = _room_layout(rng, rows, cols)
        start_room = rng.choice(sorted(rms))
        adj = {k: [] for k in rms}
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        leaves = [k for k in rms if k != start_room and len(adj[k]) == 1]
        if leaves:
            break
    else:
        return GridMap.parse("###\n#P#\n###", notes={"kind": "keynear", "tier": 3, "seed": seed})
    exit_room = rng.choice(sorted(leaves))
    parent = adj[exit_room][0]
    key_char, lock_char = rng.choice(KEYS)
    for a, b in edges:
        _connect(cv, rms, a, b, lock_char if {a, b} == {exit_room, parent} else None)
    start = _room_center(rms[start_room])
    if not _place_start_and_exit(cv, start, _room_center(rms[exit_room])):
        return GridMap.parse(cv.text(), player_angle=0, notes={"kind": "keynear", "tier": 3, "seed": seed})
    near = [k for k in [start_room] + adj[start_room] if k != exit_room]
    kr = rms[rng.choice(sorted(near))]
    floors = [(r, c) for r in range(kr[0], kr[2] + 1) for c in range(kr[1], kr[3] + 1) if cv[r, c] == "."]
    if not floors:
        return GridMap.parse(cv.text(), player_angle=0, notes={"kind": "keynear", "tier": 3, "seed": seed})
    cv[rng.choice(floors)] = key_char
    scatter(cv, rng, MONSTERS, rng.randint(0, 2), away_from=start, min_dist=6)
    scatter(cv, rng, "ha", rng.randint(0, 2))
    return GridMap.parse(cv.text(), player_angle=rng.choice(ANGLES),
                         notes={"kind": "keynear", "tier": 3, "seed": seed, "key": key_char})


GENERATORS = {"arena": arena, "maze": maze, "rooms": rooms, "keydoor": keydoor, "keynear": keynear}


def generate(kind: str, seed: int) -> GridMap:
    """Build a playable map of this kind. Reproducible from (kind, seed).

    If a layout is unsolvable (disconnected rooms, missing start, etc.), retries with a
    deterministic sub-seed derived from `seed` instead of emitting a bad map.
    """
    fn = GENERATORS[kind]
    for attempt in range(MAX_ATTEMPTS):
        grid = fn(_attempt_seed(seed, attempt))
        grid.notes["kind"] = kind
        grid.notes["tier"] = KINDS[kind]
        grid.notes["seed"] = seed
        if playable(grid, kind):
            return grid
    raise RuntimeError(f"could not generate a solvable {kind} map from seed {seed}")
