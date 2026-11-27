"""Results charts for the README, brief and demo (static SVG, no plotting library needed).

Reads docs/figures/results-val24.json (committed, so the charts can be redrawn from the repo) and writes:
- docs/figures/progress-val24.svg   weighted Mini-League total per brain, same 24-map val x 2 seeds
- docs/figures/tiers-val24.svg      per-tier score: random vs champion #1 vs champion #2
- docs/figures/ablation-memory.svg  memory on vs off (Phase 3 ablation, held-out Tier 3 maps)

    uv run python scripts/make_charts.py                 # redraw from the committed numbers
    uv run python scripts/make_charts.py --collect       # first refresh the numbers from runs/eval/*.json

Colours: the dataviz reference palette's first three categorical slots (validated for colour-blind
separation as a set); text always uses ink colours, never the series colour; each bar has a hover tooltip.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "docs" / "figures"
DATA = FIG / "results-val24.json"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # light mode slots 1..3
SERIES_DARK = ("#3987e5", "#d95926", "#199e70")
STYLE = """
<style>
  .surface { fill: #fcfcfb; } .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .grid { stroke: #e4e3df; }
  .axis { stroke: #8a8984; } .s0 { fill: %s; } .s1 { fill: %s; } .s2 { fill: %s; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; } .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; } .grid { stroke: #33332f; }
    .axis { stroke: #77766f; } .s0 { fill: %s; } .s1 { fill: %s; } .s2 { fill: %s; }
  }
  text { font-family: "Segoe UI", system-ui, -apple-system, sans-serif; }
</style>""" % (SERIES + SERIES_DARK)


def _svg(w: int, h: int, body: list[str], title: str) -> str:
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
                      f'role="img" aria-label="{escape(title)}">', f"<title>{escape(title)}</title>", STYLE,
                      f'<rect class="surface" x="0" y="0" width="{w}" height="{h}" rx="8"/>', *body, "</svg>"])


def _text(x, y, s, cls="ink", size=14, anchor="start", weight=400) -> str:
    return (f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{escape(str(s))}</text>')


def _bar(x, y, w, h, cls, tip, round_end: str) -> str:
    """A bar with a 4px rounded data end (the end away from the baseline) and a hover tooltip."""
    r = min(4.0, abs(w) / 2, abs(h) / 2)
    x0, x1, y0, y1 = x, x + w, y, y + h
    if round_end == "right":
        d = f"M{x0},{y0} H{x1 - r} Q{x1},{y0} {x1},{y0 + r} V{y1 - r} Q{x1},{y1} {x1 - r},{y1} H{x0} Z"
    elif round_end == "left":
        d = f"M{x1},{y0} H{x0 + r} Q{x0},{y0} {x0},{y0 + r} V{y1 - r} Q{x0},{y1} {x0 + r},{y1} H{x1} Z"
    elif round_end == "top":
        d = f"M{x0},{y1} V{y0 + r} Q{x0},{y0} {x0 + r},{y0} H{x1 - r} Q{x1},{y0} {x1},{y0 + r} V{y1} Z"
    else:  # bottom
        d = f"M{x0},{y0} V{y1 - r} Q{x0},{y1} {x0 + r},{y1} H{x1 - r} Q{x1},{y1} {x1},{y1 - r} V{y0} Z"
    return f'<path class="{cls}" d="{d}"><title>{escape(tip)}</title></path>'


def _ticks(lo: float, hi: float, step: float) -> list[float]:
    start = step * (lo // step)
    out, v = [], start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            out.append(v)
        v += step
    return out


def progress_chart(d: dict) -> str:
    rows = d["brains"]
    w, h = 820, 128 + 46 * len(rows)
    left, right = 300, 90
    lo = min(0.0, min(r["total"] for r in rows))
    hi = max(r["total"] for r in rows)
    lo, hi = 1000 * ((lo // 1000)), 1000 * (-(-hi // 1000))
    sx = (w - left - right) / (hi - lo)
    x0 = left + (0 - lo) * sx
    body = [_text(20, 32, "Mini-League weighted total, same 24 val maps × 2 seeds", size=17, weight=600),
            _text(20, 54, "Held-out maps never used in training. Higher is better.", "ink2", 13)]
    top = 76
    for t in _ticks(lo, hi, 1000):
        x = left + (t - lo) * sx
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - 26}" stroke-width="1"/>')
        body.append(_text(x, h - 8, f"{t:+,.0f}" if t else "0", "ink2", 12, "middle"))
    for i, r in enumerate(rows):
        y = top + 10 + i * 46
        body.append(_text(left - 12, y + 19, r["label"], "ink", 14, "end", 600 if r.get("highlight") else 400))
        if r.get("note"):
            body.append(_text(left - 12, y + 35, r["note"], "ink2", 11, "end"))
        val = r["total"]
        bw = val * sx
        cls = "s0" if r.get("highlight") else "s0"
        opacity = "" if r.get("highlight") else ' opacity="0.55"'
        bar = _bar(x0 if bw >= 0 else x0 + bw, y + 4, abs(bw), 24, cls, f'{r["label"]}: {val:+,.2f}',
                   "right" if bw >= 0 else "left")
        body.append(bar.replace("<path ", f"<path{opacity} ", 1))
        tx = x0 + bw + (8 if bw >= 0 else -8)
        body.append(_text(tx, y + 21, f"{val:+,.0f}", "ink", 13, "start" if bw >= 0 else "end", 600))
    body.append(f'<line class="axis" x1="{x0:.1f}" y1="{top}" x2="{x0:.1f}" y2="{h - 26}" stroke-width="1.5"/>')
    return _svg(w, h, body, "Mini-League weighted total per brain on the 24-map validation set")


def tiers_chart(d: dict) -> str:
    names = d["tier_compare"]
    rows = [next(r for r in d["brains"] if r["key"] == k) for k in names]
    tiers = ["1", "2", "3", "4"]
    tier_label = {"1": "Tier 1 · combat ×1", "2": "Tier 2 · navigation ×2", "3": "Tier 3 · keys & doors ×3",
                  "4": "Tier 4 · full levels ×4"}
    w, h = 820, 420
    left, right, top, bottom = 70, 20, 96, 70
    vals = [r["by_tier"][t] for r in rows for t in tiers]
    lo, hi = 500 * (min(0, min(vals)) // 500), 500 * (-(-max(vals) // 500))
    sy = (h - top - bottom) / (hi - lo)
    y0 = top + (hi - 0) * sy
    body = [_text(20, 32, "Score per tier (weighted), same 24 val maps × 2 seeds", size=17, weight=600),
            _text(20, 54, "Tier weights are the scorer's ×1–×4. Higher is better.", "ink2", 13)]
    lx = 20
    for i, r in enumerate(rows):  # legend (>= 2 series), one row above the plot
        body.append(f'<rect class="s{i}" x="{lx}" y="66" width="12" height="12" rx="2"/>')
        body.append(_text(lx + 18, 77, r["label"], "ink", 13))
        lx += 30 + 7.4 * len(r["label"])
    for t in _ticks(lo, hi, 500):
        y = top + (hi - t) * sy
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{w - right}" y2="{y:.1f}" stroke-width="1"/>')
        body.append(_text(left - 8, y + 4, f"{t:+,.0f}" if t else "0", "ink2", 12, "end"))
    group_w = (w - left - right) / len(tiers)
    bar_w = min(46, (group_w - 40) / len(rows))
    for gi, t in enumerate(tiers):
        gx = left + gi * group_w + (group_w - bar_w * len(rows) - 2 * (len(rows) - 1)) / 2
        for i, r in enumerate(rows):
            v = r["by_tier"][t]
            bh = v * sy
            x = gx + i * (bar_w + 2)
            y = y0 - bh if bh >= 0 else y0
            body.append(_bar(x, y, bar_w, abs(bh) or 0.5, f"s{i}", f'{r["label"]}, tier {t}: {v:+,.1f}',
                             "top" if bh >= 0 else "bottom"))
            if r["key"] == rows[-1]["key"]:  # selective direct label: the newest brain only
                ty = (y0 - bh - 6) if bh >= 0 else (y0 - bh + 15)
                body.append(_text(x + bar_w / 2, ty, f"{v:+,.0f}", "ink", 12, "middle", 600))
        body.append(_text(left + gi * group_w + group_w / 2, h - bottom + 22, tier_label[t], "ink", 13, "middle"))
    body.append(f'<line class="axis" x1="{left}" y1="{y0:.1f}" x2="{w - right}" y2="{y0:.1f}" stroke-width="1.5"/>')
    return _svg(w, h, body, "Score per tier: random agent vs champion 1 vs champion 2")


def ablation_chart(d: dict) -> str:
    a = d["ablation"]
    metrics = a["metrics"]  # [label, mem, nomem, better]
    arms = (a["arms"][0], a["arms"][1])
    w, h = 820, 400
    left, right, top, bottom = 70, 20, 96, 74
    sy = (h - top - bottom) / 100.0
    y0 = h - bottom
    body = [_text(20, 32, a["title"], size=17, weight=600), _text(20, 54, a["subtitle"], "ink2", 13)]
    lx = 20
    for i, name in enumerate(arms):
        body.append(f'<rect class="s{i}" x="{lx}" y="66" width="12" height="12" rx="2"/>')
        body.append(_text(lx + 18, 77, name, "ink", 13))
        lx += 40 + 7.4 * len(name)
    for t in range(0, 101, 25):
        y = y0 - t * sy
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{w - right}" y2="{y:.1f}" stroke-width="1"/>')
        body.append(_text(left - 8, y + 4, f"{t}%", "ink2", 12, "end"))
    group_w = (w - left - right) / len(metrics)
    bar_w = 46
    for gi, (label, mem, nomem, better) in enumerate(metrics):
        gx = left + gi * group_w + (group_w - 2 * bar_w - 2) / 2
        for i, v in enumerate((mem, nomem)):
            x = gx + i * (bar_w + 2)
            bh = v * sy
            body.append(_bar(x, y0 - bh, bar_w, max(bh, 0.5), f"s{i}", f"{arms[i]}, {label}: {v:g}%", "top"))
            body.append(_text(x + bar_w / 2, y0 - bh - 6, f"{v:g}%", "ink", 12, "middle", 600))
        body.append(_text(left + gi * group_w + group_w / 2, y0 + 20, label, "ink", 13, "middle"))
        body.append(_text(left + gi * group_w + group_w / 2, y0 + 37, f"({better} is better)", "ink2", 11, "middle"))
    body.append(f'<line class="axis" x1="{left}" y1="{y0:.1f}" x2="{w - right}" y2="{y0:.1f}" stroke-width="1.5"/>')
    body.append(_text(20, h - 10, a["footnote"], "ink2", 11))
    return _svg(w, h, body, a["title"])


def collect(d: dict) -> dict:
    """Refresh totals / tiers / exits for brains that have a runs/eval file (same map set only)."""
    fp = d["map_set_fingerprint"]
    for r in d["brains"]:
        src = r.get("source")
        if not src:
            continue
        path = REPO / src
        if not path.is_file():
            print(f"skip {r['key']}: {src} not found (keeping committed numbers)")
            continue
        e = json.loads(path.read_text(encoding="utf-8"))
        if e.get("map_set_fingerprint") != fp:
            raise SystemExit(f"{src}: different map set ({e.get('map_set_fingerprint')} != {fp})")
        r["total"] = round(float(e["total"]), 2)
        r["by_tier"] = {str(k): round(float(v), 2) for k, v in e["by_tier"].items()}
        r["exits"] = int(sum(bool(x["exit"]) for x in e["runs"]))
        r["runs"] = len(e["runs"])
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect", action="store_true", help="refresh numbers from runs/eval/*.json first")
    a = ap.parse_args(argv)
    d = json.loads(DATA.read_text(encoding="utf-8"))
    if a.collect:
        d = collect(d)
        DATA.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    for name, fn in (("progress-val24.svg", progress_chart), ("tiers-val24.svg", tiers_chart),
                     ("ablation-memory.svg", ablation_chart)):
        (FIG / name).write_text(fn(d) + "\n", encoding="utf-8")
        print("wrote", FIG / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
