"""Mind View: watch the agent play one map and see what it "thinks", recorded to MP4.

docs/08-DEMO-AND-BRIEF.md §8.1. Every frame shows, side by side:
- the game view (the 160 x 120 RGB buffer the agent can read),
- the mind map it has drawn so far (spawn-anchored grid; walls, doors, locked doors, items, enemies,
  breadcrumbs) with the notebook's landmarks pinned on it and the agent's pose,
- the manager's current goal and the RGB head's "exit in view?" score (brains that have them),
- the landmark notebook (type, distance, bearing, done),
- entity attention (which visible thing the brain is focusing on),
- vitals, keys held, the buttons pressed and the time per decision.

It drives the real submission agent (submission/agent.py) and only READS its state after each act(),
plus one forward hook that copies the entity-attention weights. Nothing in submission/ changes and the
agent pays no debug cost on game day. The end card shows the scorer's result for the run
(training/scorer.py, privileged state: dashboard and evaluation only, never an agent input).

    uv run python scripts/mind_view.py --map maps/curated/val/t3-keynear-100000.wad:MAP01 --out runs/demo/keynear.mp4
    uv run python scripts/mind_view.py --weights runs/x/step_500000.pt --map two_rooms --seed 1 --out demo.mp4
    uv run python scripts/mind_view.py --live --map maps/curated/val/t1-arena-100003.wad:MAP01 --map maps/curated/val/t3-keynear-100000.wad:MAP01

--live shows the dashboard in a desktop window while the agent plays (Esc stops), several maps in a row.
Each map's tick cap comes from maps/catalog.json (2,100 for maps not in it); --ticks overrides it.
Needs ffmpeg on PATH for MP4 (`--png` writes single frames without it); --live uses tkinter (ships with
Python). Test maps are refused: their one-time use goes through evaluate.py --final-test.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_spec import REPO, parse_map  # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submission"))
from doom_env import action_space, make_env, observation_space  # noqa: E402
from doom_env import spaces  # noqa: E402

from smriti.hierarchy import GOALS  # noqa: E402
from smriti.memory.mindmap import CELL, DOOR, ENEMY, ENEMY_DECAY_STEPS, FREE, GRID, ITEM, LOCKED, \
    RECENT, RECENT_DECAY_STEPS, VISITED, WALL  # noqa: E402
from smriti.memory.notebook import DOOR as NOTE_DOOR, KEY as NOTE_KEY, LOCKED_DOOR as NOTE_LOCKED  # noqa: E402

W, H = 1280, 720
GAME_W, GAME_H = 640, 480
MAP_PX = 480  # mind map panel (square)
MIN_CELLS, MAX_CELLS = 40, 200  # the view zooms to fit what has been explored (in cells of 32 units)
UNITS_PER_M = 32.0  # Doom convention: ~32 map units per metre
KEY_NAMES = ("blue card", "yellow card", "red card", "blue skull", "yellow skull", "red skull")
KEY_COLOURS = ((70, 120, 255), (240, 210, 40), (235, 60, 60), (70, 120, 255), (240, 210, 40), (235, 60, 60))
NOTE_STYLE = {NOTE_KEY: ("key", (250, 210, 40)), NOTE_LOCKED: ("locked door", (235, 60, 60)),
              NOTE_DOOR: ("door", (240, 150, 40))}

BG = (18, 20, 26)
PANEL = (28, 31, 40)
TEXT = (225, 228, 235)
DIM = (140, 146, 160)
ACCENT = (255, 205, 60)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")
    for name in names:
        for folder in (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")):
            path = folder / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


FONT = _font(17)
FONT_SMALL = _font(14)
FONT_BOLD = _font(19, bold=True)
FONT_BIG = _font(34, bold=True)


@dataclass
class Snapshot:
    """Everything one dashboard frame shows (all read from the agent after its act())."""
    step: int
    max_steps: int
    rgb: np.ndarray  # (120, 160, 3) uint8
    vitals: np.ndarray  # (17,)
    buttons: np.ndarray  # (9,) the executed action
    ms: float
    policy_name: str
    pose: tuple[float, float, float] | None = None  # x, y, heading_deg (spawn frame); None = odometry off
    grid: np.ndarray | None = None  # (C, GRID, GRID) mind map
    map_t: float = 0.0
    notes: list[dict] = field(default_factory=list)
    goal: str | None = None
    exit_probability: float | None = None
    attention: list[tuple[str, float]] = field(default_factory=list)
    banner: str | None = None


# ----------------------------------------------------------------------------- reading the agent
class AttentionTap:
    """Forward hook on the entity encoder: re-run its attention with weights on and keep them."""

    def __init__(self, net):
        self.weights = None
        self.enc = net.entities
        self.handle = self.enc.register_forward_hook(self._hook)
        self._busy = False

    def _hook(self, module, args, output):
        if self._busy:
            return
        self._busy = True
        try:
            import torch
            with torch.inference_mode():
                _, w = module(*args, return_weights=True)
            self.weights = w[-1].detach().float().cpu().numpy()  # (33,) index 0 = "nothing"
        finally:
            self._busy = False


def read_notes(memory) -> list[dict]:
    nb = getattr(memory, "notebook", None)
    if nb is None:
        return []
    px, py, heading = memory.pose.x, memory.pose.y, memory.pose.heading_deg
    out = []
    for i in np.flatnonzero(nb.kind > 0):
        dx, dy = nb.pos[i, 0] - px, nb.pos[i, 1] - py
        bearing = (math.degrees(math.atan2(dy, dx)) - heading + 180.0) % 360.0 - 180.0  # + = to the left
        out.append(dict(kind=int(nb.kind[i]), x=float(nb.pos[i, 0]), y=float(nb.pos[i, 1]),
                        dist_m=math.hypot(dx, dy) / UNITS_PER_M, bearing=bearing, done=bool(nb.done[i] > 0.5)))
    out.sort(key=lambda n: (n["done"], n["dist_m"]))
    return out


def read_attention(tap: AttentionTap | None, obs: dict) -> list[tuple[str, float]]:
    if tap is None or tap.weights is None:
        return []
    ents = obs["visible_entities"]
    mask = np.asarray(ents["mask"]).astype(bool)
    types = np.asarray(ents["entity_type"]).astype(int)
    w = tap.weights
    rows = [("nothing (null token)", float(w[0]))]
    for i in np.flatnonzero(mask):
        t = types[i]
        rows.append((spaces.ENTITY_TYPES[t] if 0 <= t < len(spaces.ENTITY_TYPES) else "?", float(w[i + 1])))
    rows.sort(key=lambda r: -r[1])
    return rows[:5]


# ----------------------------------------------------------------------------- drawing
def _panel(d: ImageDraw.ImageDraw, box, title: str | None = None) -> None:
    d.rounded_rectangle(box, radius=8, fill=PANEL)
    if title:
        d.text((box[0] + 12, box[1] + 8), title, font=FONT_BOLD, fill=ACCENT)


def _check(d: ImageDraw.ImageDraw, x: float, y: float, size: float, colour=(120, 255, 140)) -> None:
    """A drawn check mark (the UI fonts have no reliable glyph for it)."""
    d.line((x + size * 0.1, y + size * 0.55, x + size * 0.4, y + size * 0.85, x + size * 0.95, y + size * 0.15),
           fill=colour, width=3)


def view_window(g: np.ndarray, x: float, y: float) -> tuple[int, int, int]:
    """(cells, x0, y0): a square window that fits everything drawn so far plus the agent.

    Size and centre are snapped (16 / 4 cells) so the view does not jitter from frame to frame."""
    acx, acy = int(math.floor(x / CELL)) + GRID // 2, int(math.floor(y / CELL)) + GRID // 2
    ys, xs = np.nonzero((g[FREE] > 0) | (g[WALL] > 0))
    xmin, xmax = min(int(xs.min()) if xs.size else acx, acx), max(int(xs.max()) if xs.size else acx, acx)
    ymin, ymax = min(int(ys.min()) if ys.size else acy, acy), max(int(ys.max()) if ys.size else acy, acy)
    span = max(xmax - xmin, ymax - ymin) + 8
    cells = int(min(MAX_CELLS, max(MIN_CELLS, -(-span // 16) * 16)))
    cx, cy = (xmin + xmax + 1) // 2, (ymin + ymax + 1) // 2
    if cells == MAX_CELLS:  # too big to fit: follow the agent instead
        cx, cy = acx, acy
    cx, cy = 4 * round(cx / 4), 4 * round(cy / 4)
    return cells, cx - cells // 2, cy - cells // 2


def _world_to_px(x, y, cx, cy, scale, ox, oy):
    """Spawn-frame (x forward, y left) -> panel pixel; screen up = initial forward, screen left = +y."""
    return ox + MAP_PX / 2 - (y - cy) / CELL * scale, oy + MAP_PX / 2 - (x - cx) / CELL * scale


def draw_mindmap(img: Image.Image, snap: Snapshot, ox: int, oy: int) -> None:
    d = ImageDraw.Draw(img)
    d.rectangle((ox, oy, ox + MAP_PX - 1, oy + MAP_PX - 1), fill=(8, 9, 12))
    if snap.grid is None or snap.pose is None:
        d.text((ox + 20, oy + MAP_PX / 2 - 10), "no mind map (this brain has no memory)", font=FONT, fill=DIM)
        return
    x, y, heading = snap.pose
    g, t = snap.grid, snap.map_t
    cells, x0, y0 = view_window(g, x, y)
    window = np.zeros((g.shape[0], cells, cells), np.float32)  # [layer, y-cell, x-cell]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + cells, GRID), min(y0 + cells, GRID)
    if sx1 > sx0 and sy1 > sy0:
        window[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = g[:, sy0:sy1, sx0:sx1]
    rgb = np.zeros((cells, cells, 3), np.float32)
    free = window[FREE] > 0
    rgb[free] = (52, 56, 66)
    recent = np.where(window[RECENT] > 0, np.exp(-(t - window[RECENT]) / RECENT_DECAY_STEPS), 0.0)
    visited = window[VISITED] > 0
    trail = np.array((40, 110, 170), np.float32)[None] * (0.45 + 0.55 * recent[visited, None])
    rgb[visited] = trail
    for layer, colour in ((WALL, (215, 218, 225)), (DOOR, (240, 150, 40)), (LOCKED, (235, 50, 50)),
                          (ITEM, (70, 200, 110))):
        rgb[window[layer] > 0] = colour
    enemy = np.where(window[ENEMY] > 0, np.exp(-(t - window[ENEMY]) / ENEMY_DECAY_STEPS), 0.0)
    hot = enemy > 0.05
    rgb[hot] = (1 - enemy[hot, None]) * rgb[hot] + enemy[hot, None] * np.array((255, 40, 220), np.float32)
    # [y-cell, x-cell] -> screen: up = +x, left = +y  => transpose, then flip both axes
    screen = np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1, ::-1]).astype(np.uint8)
    tile = Image.fromarray(screen).resize((MAP_PX, MAP_PX), Image.NEAREST)
    img.paste(tile, (ox, oy))
    scale = MAP_PX / cells
    cx_world, cy_world = (x0 + cells / 2 - GRID // 2) * CELL, (y0 + cells / 2 - GRID // 2) * CELL
    for n in snap.notes:
        name, colour = NOTE_STYLE.get(n["kind"], ("?", TEXT))
        px, py = _world_to_px(n["x"], n["y"], cx_world, cy_world, scale, ox, oy)
        if not (ox <= px < ox + MAP_PX and oy <= py < oy + MAP_PX):
            continue
        r = 8
        if n["kind"] == NOTE_KEY:
            d.ellipse((px - r, py - r, px + r, py + r), outline=colour, width=3)
        else:
            d.rectangle((px - r, py - r, px + r, py + r), outline=colour, width=3)
        if n["done"]:
            _check(d, px - r, py - r, 2 * r)
    px, py = _world_to_px(x, y, cx_world, cy_world, scale, ox, oy)
    a = math.radians(heading)
    hx, hy = px - math.sin(a) * 22, py - math.cos(a) * 22  # forward (cos, sin) in (x, y) -> screen
    d.line((px, py, hx, hy), fill=ACCENT, width=4)
    d.ellipse((px - 7, py - 7, px + 7, py + 7), fill=ACCENT, outline=(0, 0, 0), width=2)
    d.text((ox + 10, oy + 8), "MIND MAP", font=FONT_BOLD, fill=ACCENT)
    d.text((ox + 10, oy + MAP_PX - 24), f"{cells * CELL / UNITS_PER_M:.0f} m across · up = spawn forward",
           font=FONT_SMALL, fill=DIM)


def draw_legend(d: ImageDraw.ImageDraw, ox: int, oy: int) -> None:
    items = (((215, 218, 225), "wall"), ((240, 150, 40), "door"), ((235, 50, 50), "locked door"),
             ((70, 200, 110), "item"), ((255, 40, 220), "enemy (fades)"), ((40, 110, 170), "visited"),
             ((52, 56, 66), "seen free"), (ACCENT, "me + heading"))
    d.text((ox, oy), "LEGEND", font=FONT_BOLD, fill=ACCENT)
    for i, (c, label) in enumerate(items):
        yy = oy + 34 + i * 27
        d.rectangle((ox, yy + 3, ox + 16, yy + 19), fill=c)
        d.text((ox + 24, yy), label, font=FONT_SMALL, fill=TEXT)
    yy = oy + 34 + len(items) * 27 + 10
    d.text((ox, yy), "notebook pins:", font=FONT_SMALL, fill=DIM)
    d.ellipse((ox, yy + 26, ox + 16, yy + 42), outline=(250, 210, 40), width=3)
    d.text((ox + 24, yy + 24), "key", font=FONT_SMALL, fill=TEXT)
    d.rectangle((ox, yy + 52, ox + 16, yy + 68), outline=(235, 60, 60), width=3)
    d.text((ox + 24, yy + 50), "door / locked", font=FONT_SMALL, fill=TEXT)
    _check(d, ox, yy + 78, 16)
    d.text((ox + 24, yy + 76), "= done", font=FONT_SMALL, fill=(120, 255, 140))


def _arrow(bearing: float) -> str:
    if abs(bearing) < 20:
        return "ahead"
    if abs(bearing) > 150:
        return "behind"
    return ("←" if bearing > 0 else "→") + f" {abs(bearing):.0f}°"


def render_frame(snap: Snapshot) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    game = Image.fromarray(np.asarray(snap.rgb, np.uint8)).resize((GAME_W, GAME_H), Image.NEAREST)
    img.paste(game, (0, 0))
    d.rectangle((0, 0, 250, 30), fill=(0, 0, 0))
    d.text((10, 5), "GAME VIEW (agent's RGB)", font=FONT_BOLD, fill=ACCENT)
    draw_mindmap(img, snap, GAME_W, 0)
    draw_legend(d, GAME_W + MAP_PX + 14, 14)

    # goal bar
    _panel(d, (6, 486, W - 6, 524))
    goal = snap.goal if snap.goal is not None else "— (this brain has no manager)"
    d.text((18, 494), "CURRENT GOAL:", font=FONT_BOLD, fill=ACCENT)
    d.text((170, 494), goal, font=FONT_BOLD, fill=TEXT)
    if snap.exit_probability is not None:
        d.text((620, 494), f"exit in view? {snap.exit_probability:.2f}", font=FONT, fill=TEXT)
        d.rectangle((800, 500, 1000, 512), outline=DIM)
        d.rectangle((800, 500, 800 + 200 * snap.exit_probability, 512), fill=(90, 220, 120))
    if snap.pose is not None:
        d.text((1030, 494), f"heading {snap.pose[2] % 360:5.0f}°", font=FONT, fill=DIM)

    # notebook
    _panel(d, (6, 530, 420, 714), "NOTEBOOK")
    if not snap.notes:
        d.text((18, 566), "(no landmarks yet)", font=FONT, fill=DIM)
    for i, n in enumerate(snap.notes[:5]):
        name, colour = NOTE_STYLE.get(n["kind"], ("?", TEXT))
        yy = 566 + i * 28
        if n["done"]:
            _check(d, 16, yy + 3, 16)
        else:
            d.ellipse((20, yy + 7, 30, yy + 17), fill=colour)
        d.text((40, yy), name, font=FONT, fill=colour)
        d.text((170, yy), f"{n['dist_m']:5.1f} m", font=FONT, fill=TEXT)
        d.text((260, yy), _arrow(n["bearing"]), font=FONT, fill=TEXT)
    if len(snap.notes) > 5:
        d.text((300, 690), f"+{len(snap.notes) - 5} more", font=FONT_SMALL, fill=DIM)

    # attention
    _panel(d, (426, 530, 850, 714), "ATTENTION (entities)")
    if not snap.attention:
        d.text((438, 566), "(no attention data)", font=FONT, fill=DIM)
    for i, (name, w) in enumerate(snap.attention):
        yy = 566 + i * 28
        d.text((438, yy), name[:22], font=FONT, fill=TEXT if i else ACCENT)
        d.rectangle((640, yy + 5, 640 + 160, yy + 19), outline=(60, 64, 76))
        d.rectangle((640, yy + 5, 640 + 160 * max(0.0, min(w, 1.0)), yy + 19), fill=(255, 120, 60))
        d.text((806, yy), f"{w:.2f}", font=FONT_SMALL, fill=DIM)

    # vitals and timing
    _panel(d, (856, 530, W - 6, 714), "STATUS")
    v = snap.vitals
    d.text((868, 566), f"health {v[0]:.0f}   armor {v[1]:.0f}   ammo {v[3]:.0f}", font=FONT, fill=TEXT)
    keys = [i for i in range(6) if v[8 + i] > 0.5]
    d.text((868, 594), "keys:", font=FONT, fill=TEXT)
    if not keys:
        d.text((920, 594), "none", font=FONT, fill=DIM)
    for j, k in enumerate(keys):
        d.rounded_rectangle((920 + j * 92, 596, 1006 + j * 92, 616), radius=4, fill=KEY_COLOURS[k])
        d.text((926 + j * 92, 596), KEY_NAMES[k][:11], font=FONT_SMALL, fill=(0, 0, 0))
    pressed = [spaces.BUTTONS[i].replace("_", " ").lower() for i in np.flatnonzero(snap.buttons)]
    d.text((868, 622), "buttons: " + (", ".join(pressed) if pressed else "none"), font=FONT_SMALL, fill=TEXT)
    d.text((868, 650), f"step {snap.step}/{snap.max_steps}   {snap.ms:.1f} ms/decision", font=FONT, fill=TEXT)
    d.text((868, 680), snap.policy_name[:44], font=FONT_SMALL, fill=DIM)

    if snap.banner:
        d.rounded_rectangle((70, 190, 570, 290), radius=12, fill=(0, 0, 0))
        d.text((90, 215), snap.banner, font=FONT_BIG, fill=ACCENT)
    return img


def end_card(last: Image.Image, title: str, lines: list[str]) -> Image.Image:
    img = Image.blend(last, Image.new("RGB", (W, H), (0, 0, 0)), 0.72)
    d = ImageDraw.Draw(img)
    d.text((80, 160), title, font=FONT_BIG, fill=ACCENT)
    for i, line in enumerate(lines):
        d.text((80, 230 + i * 36), line, font=FONT_BOLD, fill=TEXT)
    return img


# ----------------------------------------------------------------------------- recording
class Mp4Writer:
    def __init__(self, path: Path, fps: float):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise SystemExit("ffmpeg not found on PATH (needed for MP4; use --png for single frames)")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, img: Image.Image, repeat: int = 1) -> None:
        data = img.tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(data)

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit("ffmpeg failed")


class LiveWindow:
    """Shows dashboard frames in a desktop window at a steady pace (tkinter: ships with Python)."""

    def __init__(self, fps: float, scale: float = 1.0, title: str = "AK-Doom Mind View"):
        import tkinter as tk
        from PIL import ImageTk

        self._tk, self._imagetk = tk, ImageTk
        self.root = tk.Tk()
        self.root.title(f"{title}  (Esc to stop)")
        self.root.configure(background="black")
        self.label = tk.Label(self.root, bd=0, background="black")
        self.label.pack()
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _e: self.close())
        self.period, self.scale = 1.0 / fps, scale
        self.next_t = time.perf_counter()
        self._photo = None

    def show(self, img: Image.Image) -> bool:
        """Draw one frame; returns False once the viewer closed the window."""
        if self.closed:
            return False
        if self.scale != 1.0:
            img = img.resize((int(W * self.scale), int(H * self.scale)), Image.BILINEAR)
        self._photo = self._imagetk.PhotoImage(img)
        self.label.configure(image=self._photo)
        try:
            self.root.update()
        except self._tk.TclError:
            self.closed = True
            return False
        self.next_t += self.period
        wait = self.next_t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            self.next_t = time.perf_counter()
        return not self.closed

    def hold(self, img: Image.Image, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and self.show(img):
            pass

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                self.root.destroy()
            except self._tk.TclError:
                pass


def resolve_map(spec: str) -> str:
    if spec == "two_rooms":
        return str(REPO / "maps" / "two_rooms.wad") + ":MAP01"
    return spec


def default_ticks(spec: str) -> int:
    """The map's tick cap from maps/catalog.json (the same value the Mini-League uses), else 2,100."""
    cat = REPO / "maps" / "catalog.json"
    if cat.is_file():
        for m in json.loads(cat.read_text(encoding="utf-8"))["maps"]:
            if m["id"] == spec and m.get("episode_ticks"):
                return int(m["episode_ticks"])
    return 2100


def refuse_test_map(spec: str) -> None:
    if "/curated/test/" in spec.replace(chr(92), "/"):
        raise SystemExit(f"refusing test map {spec}: test maps are used once, through evaluate.py --final-test")


def record(weights: Path | None, spec: str, seed: int, out: Path | None, fps: float, ticks: int | None,
           png_steps: list[int], png_dir: Path | None, max_steps: int | None,
           live: LiveWindow | None = None) -> dict:
    sys.path.insert(0, str(REPO / "submission"))
    from agent import DoomAgent  # the real submission agent
    from training.scorer import ScoreTracker

    agent = DoomAgent(observation_space(), action_space(), "cpu", weights_path=weights)
    net = getattr(agent.policy, "net", None)
    tap = AttentionTap(net) if net is not None else None
    refuse_test_map(spec)
    wad, map_name = parse_map(resolve_map(spec))
    episode_ticks = ticks or default_ticks(spec)
    env = make_env(wad, map_name, episode_ticks=episode_ticks, render_buffers=True)
    tracker = ScoreTracker()
    writer = Mp4Writer(out, fps) if out is not None else None
    total_steps = episode_ticks // spaces.FRAME_SKIP
    frame = None
    keys_before = 0
    banner, banner_left = None, 0
    ms_all = []
    try:
        obs, _ = env.reset(seed=seed)
        tracker.reset(env.privileged_state())
        agent.reset()
        done, step = False, 0
        while not done and (max_steps is None or step < max_steps):
            t0 = time.perf_counter()
            action = np.asarray(agent.act(obs), np.int8)
            ms = 1000 * (time.perf_counter() - t0)
            ms_all.append(ms)
            mem = agent.memory
            mm = getattr(mem, "mindmap", None)
            goal = agent.active_goal
            keys_now = int(np.sum(np.asarray(obs["vitals"])[8:14] > 0.5))
            if keys_now > keys_before:
                banner, banner_left = "KEY PICKED UP", int(fps * 1.5)
            keys_before = keys_now
            snap = Snapshot(
                step=step + 1, max_steps=total_steps, rgb=np.asarray(obs["rgb"]), vitals=np.asarray(obs["vitals"]),
                buttons=action, ms=ms, policy_name=agent.policy_name,
                pose=(mem.pose.x, mem.pose.y, mem.pose.heading_deg) if mem.odometer is not None and mem.valid else None,
                grid=mm.grid if mm is not None else None, map_t=float(mm.t) if mm is not None else 0.0,
                notes=read_notes(mem), goal=GOALS[goal] if goal is not None else None,
                exit_probability=agent.exit_probability, attention=read_attention(tap, obs),
                banner=banner if banner_left > 0 else None)
            banner_left -= 1
            frame = render_frame(snap)
            if writer is not None:
                writer.write(frame)
            if live is not None and not live.show(frame):
                break
            if png_dir is not None and (step + 1) in png_steps:
                png_dir.mkdir(parents=True, exist_ok=True)
                frame.save(png_dir / f"mindview-step{step + 1:04d}.png")
            obs, _, terminated, truncated, _ = env.step(action)
            tracker.update(env.privileged_state())
            step += 1
            done = terminated or truncated
    finally:
        env.close()
        if tap is not None:
            tap.handle.remove()
    res = tracker.result().to_dict()
    summary = dict(map=spec, seed=seed, steps=step, exit=bool(res.get("exit")), keys=res.get("keys"),
                   locked_doors=res.get("locked_doors"), kills=res.get("kills"), dead=bool(res.get("dead")),
                   score=round(float(res.get("score", 0.0)), 1),
                   mean_ms=round(float(np.mean(ms_all)) if ms_all else 0.0, 2))
    if frame is not None and (writer is not None or live is not None):
        outcome = "EXIT REACHED" if summary["exit"] else ("DIED" if summary["dead"] else "TIME UP")
        lines = [f"map: {spec.replace(str(REPO), '').lstrip('/').lstrip(chr(92))}",
                 f"steps: {summary['steps']}   keys: {summary['keys']}   locked doors opened: {summary['locked_doors']}",
                 f"kills: {summary['kills']}   map score: {summary['score']}",
                 f"mean {summary['mean_ms']} ms per decision (budget 40 ms)"]
        card = end_card(frame, outcome, lines)
        if writer is not None:
            writer.write(card, repeat=int(fps * 3))
            writer.close()
        if live is not None:
            live.hold(card, 3.0)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, help="brain checkpoint (default: submission/weights/champion.pt)")
    ap.add_argument("--map", action="append", required=True,
                    help="wad[:MAPxx], or 'two_rooms' for the template test map; repeat with --live to play several")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ticks", type=int, help="episode length in game tics (default: the map's catalog cap)")
    ap.add_argument("--live", action="store_true", help="show the dashboard in a window while it plays")
    ap.add_argument("--scale", type=float, default=1.0, help="live window size (e.g. 0.75 on small screens)")
    ap.add_argument("--out", type=Path, help="MP4 path")
    ap.add_argument("--fps", type=float, default=17.5, help="17.5 = 2x real time (a decision is 4 tics)")
    ap.add_argument("--png", type=int, action="append", default=[], help="also save the frame at this step")
    ap.add_argument("--png-dir", type=Path, default=REPO / "runs" / "demo")
    ap.add_argument("--max-steps", type=int)
    a = ap.parse_args(argv)
    if a.out is None and not a.png and not a.live:
        ap.error("give --live, --out and/or --png")
    if len(a.map) > 1 and (a.out is not None or a.png):
        ap.error("several --map values only work with --live (one MP4 / PNG set per map)")
    live = LiveWindow(fps=a.fps, scale=a.scale) if a.live else None
    try:
        for spec in a.map:
            summary = record(a.weights, spec, a.seed, a.out, a.fps, a.ticks, a.png,
                             a.png_dir if a.png else None, a.max_steps, live)
            print(summary)
            if live is not None and live.closed:
                break
    finally:
        if live is not None:
            live.close()
    if a.out is not None:
        print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
