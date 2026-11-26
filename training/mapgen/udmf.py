"""Text drawing -> playable Doom map (UDMF, the format of the template's own test maps).

Each character is one 64 x 64 unit cell:

    #  wall                     .  floor                    P  player start
    D  door (opens with USE)    R B Y  locked door (red / blue / yellow key)
    r b y  keycard (red / blue / yellow)                    X  exit pad (walk onto it to finish)
    z  zombieman  s  shotgun guy  i  imp  d  demon        (monsters, present at every skill)
    h  medikit  a  ammo clip  A  green armor  S  shotgun

Connected floor cells become one room (sector); every door cell is its own sector, closed at the
start; connected X cells form the exit sector, whose border lines end the level when crossed.
Door lines face outward (the door sector is the back side), which is what the engine and the
official doom_env.mapinfo parser expect for "manual" doors.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from doom_env.wad import Lump, write_wad

CELL = 64
ROOM_CEILING = 128
WALL, FLOOR, DOOR, EXIT = "#", ".", "D", "X"
LOCKED = {"R": 129, "B": 130, "Y": 131}  # ZDoom lock numbers: any red / blue / yellow key
LOCK_TRACK = {129: "DOORRED", 130: "DOORBLU", 131: "DOORYEL"}
THING_TYPES = {
    "P": 1,
    "r": 13, "b": 5, "y": 6,  # red / blue / yellow keycard
    "z": 3004, "s": 9, "i": 3001, "d": 3002,  # zombieman, shotgun guy, imp, demon
    "h": 2012, "a": 2007, "A": 2018, "S": 2001,  # medikit, clip, green armor, shotgun
}
OPEN = set(".PXD") | set(LOCKED) | set(THING_TYPES)


@dataclass
class GridMap:
    rows: list[str]
    player_angle: int = 0  # degrees, 0 = east (+x), 90 = north
    name: str = "MAP01"
    notes: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str, **kw) -> "GridMap":
        rows = [ln for ln in text.strip("\n").splitlines()]
        width = max(len(r) for r in rows)
        return cls([r.ljust(width, WALL) for r in rows], **kw)

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])

    def at(self, r: int, c: int) -> str:
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.rows[r][c]
        return WALL

    def cells(self, chars: str | set[str]):
        for r, row in enumerate(self.rows):
            for c, ch in enumerate(row):
                if ch in chars:
                    yield r, c


def _kind(ch: str) -> str:
    if ch == DOOR or ch in LOCKED:
        return "door"
    if ch == EXIT:
        return "exit"
    return "room" if ch in OPEN else "wall"


def label_regions(grid: GridMap) -> dict[tuple[int, int], int]:
    """Cell -> sector index. Rooms and exit pads are flood-filled; every door cell is its own sector."""
    region: dict[tuple[int, int], int] = {}
    count = 0
    for r, c in grid.cells(OPEN):
        if (r, c) in region:
            continue
        kind = _kind(grid.at(r, c))
        region[(r, c)] = count
        if kind != "door":
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if (nr, nc) not in region and grid.at(nr, nc) in OPEN and _kind(grid.at(nr, nc)) == kind:
                        region[(nr, nc)] = count
                        queue.append((nr, nc))
        count += 1
    return region


def to_udmf(grid: GridMap) -> str:
    appearance = grid.notes.get("appearance", {})
    wall_tex = appearance.get("walls", "STARTAN2")
    room_floor = appearance.get("floors", "FLOOR0_1")
    ceiling_tex = appearance.get("ceilings", "CEIL1_1")
    light = int(appearance.get("light", 192))
    region = label_regions(grid)
    n_sectors = max(region.values(), default=-1) + 1
    sector_kind = ["room"] * n_sectors
    sector_lock = [0] * n_sectors
    for (r, c), s in region.items():
        sector_kind[s] = _kind(grid.at(r, c))
        sector_lock[s] = LOCKED.get(grid.at(r, c), 0)

    verts: dict[tuple[int, int], int] = {}

    def vid(x: int, y: int) -> int:
        return verts.setdefault((x, y), len(verts))

    def corners(r: int, c: int):
        x0, x1 = c * CELL, (c + 1) * CELL
        y_top, y_bot = (grid.height - r) * CELL, (grid.height - r - 1) * CELL
        # Clockwise around the cell, so the cell is always on the RIGHT (= front) side of each edge.
        return {
            (-1, 0): ((x0, y_top), (x1, y_top)),
            (0, 1): ((x1, y_top), (x1, y_bot)),
            (1, 0): ((x1, y_bot), (x0, y_bot)),
            (0, -1): ((x0, y_bot), (x0, y_top)),
        }

    one_sided: list[tuple[tuple, int]] = []  # (edge, sector)
    two_sided: dict[frozenset, list[tuple[tuple, int]]] = {}
    for (r, c), s in region.items():
        for (dr, dc), edge in corners(r, c).items():
            n = (r + dr, c + dc)
            if n not in region:
                one_sided.append((edge, s))
            elif region[n] != s:
                two_sided.setdefault(frozenset(edge), []).append((edge, s))

    sides: list[str] = []
    lines: list[str] = []

    def side(sector: int, top: str = "-", mid: str = "-") -> int:
        sides.append(f'sidedef {{ sector = {sector}; texturetop = "{top}"; texturemiddle = "{mid}"; '
                     f'texturebottom = "-"; }}')
        return len(sides) - 1

    for (a, b), s in one_sided:
        kind = sector_kind[s]
        mid = LOCK_TRACK.get(sector_lock[s], "DOORTRAK") if kind == "door" else wall_tex
        lines.append(f"linedef {{ v1 = {vid(*a)}; v2 = {vid(*b)}; sidefront = {side(s, mid=mid)}; blocking = true; "
                     f"dontpegbottom = true; }}")

    for halves in two_sided.values():
        # Front = the side that is NOT the door / exit pad (doors must be the back sector).
        halves.sort(key=lambda h: sector_kind[h[1]] in ("door", "exit"))
        (a, b), front = halves[0]
        back = halves[1][1] if len(halves) > 1 else front
        door = next((s for s in (front, back) if sector_kind[s] == "door"), None)
        exit_pad = any(sector_kind[s] == "exit" for s in (front, back))
        props = f"v1 = {vid(*a)}; v2 = {vid(*b)}; twosided = true;"
        top = "-"
        if door is not None:
            lock = sector_lock[door]
            top = "BIGDOOR2"
            if lock:
                props += f" special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = {lock}; playeruse = true; repeatspecial = true;"
            else:
                props += " special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = true; repeatspecial = true;"
        elif exit_pad:
            props += " special = 243; playercross = true;"
        lines.append(f"linedef {{ {props} sidefront = {side(front, top=top)}; sideback = {side(back, top=top)}; }}")

    sectors = []
    for s in range(n_sectors):
        kind = sector_kind[s]
        ceiling = 0 if kind == "door" else ROOM_CEILING
        floor_tex = {"exit": "GATE3", "door": "FLAT20"}.get(kind, room_floor)
        sectors.append(f'sector {{ texturefloor = "{floor_tex}"; textureceiling = "{ceiling_tex}"; heightfloor = 0; '
                       f"heightceiling = {ceiling}; lightlevel = {light}; }}")

    things = []
    for r, c in grid.cells(set(THING_TYPES)):
        ch = grid.at(r, c)
        x, y = c * CELL + CELL // 2, (grid.height - r - 1) * CELL + CELL // 2
        angle = grid.player_angle if ch == "P" else 0
        things.append(f"thing {{ x = {x}.0; y = {y}.0; angle = {angle}; type = {THING_TYPES[ch]}; skill1 = true; "
                      f"skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }}")

    vertex_lines = [f"vertex {{ x = {x}.0; y = {y}.0; }}" for (x, y), _ in sorted(verts.items(), key=lambda kv: kv[1])]
    return "\n".join(['namespace = "zdoom";', *vertex_lines, *sectors, *sides, *lines, *things]) + "\n"


def validate(grid: GridMap) -> None:
    """Raise if the drawing can't become a sensible map."""
    starts = list(grid.cells("P"))
    if len(starts) != 1:
        raise ValueError(f"need exactly one P, found {len(starts)}")
    for r, c in grid.cells({DOOR, *LOCKED}):
        ns, ew = grid.at(r - 1, c) + grid.at(r + 1, c), grid.at(r, c - 1) + grid.at(r, c + 1)
        corridor_ns = WALL not in ns and ew == WALL * 2
        corridor_ew = WALL not in ew and ns == WALL * 2
        if not (corridor_ns or corridor_ew):
            raise ValueError(f"door at row {r}, col {c} must sit in a 1-cell corridor (walls on both sides)")


def write_map(grid: GridMap, path: str | Path) -> Path:
    validate(grid)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wad(path, [Lump(grid.name, b""), Lump("TEXTMAP", to_udmf(grid).encode()), Lump("ENDMAP", b"")])
    return path
