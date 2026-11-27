"""Replay reviewer: an LLM (hackathon Creo API) reads our run traces like an organiser watching replays.

Input: a folder written by scripts/episode_trace.py (one .txt summary + .json trace per run). For each run
the model returns a judge-style JSON verdict (what a viewer sees, at which steps, how bad, likely cause,
training-side fixes); a final call ranks the problems across all runs. Output: one Markdown report and the
raw JSON in runs/llm/reviews/<name>/.

    uv run python scripts/episode_trace.py --split val --seeds 2 --jobs 8 --name val-flags
    uv run python scripts/replay_review.py runs/llm/traces/val-flags --flags-only          # free: rules only
    uv run python scripts/replay_review.py runs/llm/traces/val-flags --dry-run --limit 2   # check prompts, no credits
    uv run python scripts/replay_review.py runs/llm/traces/val-flags --only-flagged        # uses credits

Testing tool only: the reviewer never touches the agent, and its suggestions are training-side (rewards,
curriculum, data), because the rules forbid hand-coded decision rules in the submitted agent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from creo import Creo, parse_json_reply, usage_summary  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "llm" / "reviews"
MINOR = {"weapon_cycling"}  # whole-run habits: reported once in the summary, not a reason to review a run alone

SYSTEM = """You review replays of an autonomous DOOM agent for a hackathon (ViZDoom engine, Freedoom 2 levels).
Organisers watch the top teams' replays, and behaviour such as spinning on the spot, hugging walls, getting
stuck, running back and forth or constantly switching weapons looks bad even when the score is fine.

About the agent ("Smriti"): a small neural network (recurrent PPO, GRU memory) that sees 64 distance rays,
up to 32 visible entities and its vitals, plus its own mind map and a notebook of keys and doors it has
seen. It picks 9 buttons every 4 game tics (about 9 decisions per second). It gets no map coordinates.

You get a compact numeric trace of one run (from the scorer's ground truth) and automatic flags from simple
rules. Judge it as a viewer of the replay would. Be specific and honest; do not invent events that are not
in the trace. Fix ideas must be TRAINING-side (reward shaping, curriculum, maps, auxiliary losses, action
space design, more training) because hand-coded decision rules are not allowed in the agent.

Reply with ONLY a JSON object:
{"verdict": "looks intelligent" | "mixed" | "looks broken",
 "summary": "<2 sentences a judge would say>",
 "moments": [{"steps": "<a-b>", "kind": "<short label>", "severity": "low" | "medium" | "high",
              "viewer_sees": "<what it looks like on screen>", "likely_cause": "<why the policy does this>"}],
 "strengths": ["<good behaviour visible in this run>"],
 "fix_ideas": ["<training-side fix>"]}"""

SYNTH = """You are the lead reviewer. Below are per-run replay reviews of the same DOOM agent on held-out
validation maps, plus counts of automatic flags. Write a short Markdown report for the team with:
1. "Top problems a judge will notice": up to 5, ranked by how visible and how common they are, each with
   the runs and steps where it shows (map names), and why it probably happens.
2. "Best runs to show in a live demo": 3-5 runs that look intelligent, with one line each.
3. "Training-side fixes, in priority order": concrete, cheap first (no hand-coded decision rules).
4. "What to say in Q&A about these weaknesses": 2-4 honest one-liners.
Only use facts from the reviews. Keep it under 500 words."""


def load_runs(trace_dir: Path) -> list[dict]:
    runs = []
    for js in sorted(trace_dir.glob("*.json")):
        data = json.loads(js.read_text(encoding="utf-8"))
        txt = js.with_suffix(".txt")
        runs.append(dict(name=js.stem, trace=data, flags=data.get("flags", []),
                         summary=txt.read_text(encoding="utf-8") if txt.is_file() else ""))
    return runs


def flags_report(runs: list[dict]) -> str:
    counts: dict[str, int] = {}
    runs_with: dict[str, list[str]] = {}
    for r in runs:
        for f in r["flags"]:
            counts[f["kind"]] = counts.get(f["kind"], 0) + 1
            runs_with.setdefault(f["kind"], [])
            if r["name"] not in runs_with[f["kind"]]:
                runs_with[f["kind"]].append(r["name"])
    lines = ["## Automatic flags (numeric rules, no LLM)", "",
             f"{len(runs)} runs. Exits: {sum(r['trace']['result']['exit'] for r in runs)}, "
             f"deaths: {sum(r['trace']['result']['dead'] for r in runs)}.", "",
             "| Flag | Times | Runs affected |", "|---|---|---|"]
    for k, v in sorted(counts.items(), key=lambda kv: -len(runs_with[kv[0]])):
        lines.append(f"| {k} | {v} | {len(runs_with[k])} / {len(runs)} |")
    lines += ["", "### Runs with the most flagged steps", ""]
    def flagged_steps(r):
        covered = set()
        for f in r["flags"]:
            if f["kind"] not in MINOR:
                covered.update(range(f["start"], f["end"] + 1))
        return len(covered)
    ranked = sorted(runs, key=lambda r: -flagged_steps(r))[:10]
    lines += ["| Run | Outcome | Steps | Flagged steps | Flags |", "|---|---|---|---|---|"]
    for r in ranked:
        res = r["trace"]["result"]
        outcome = "exit" if res["exit"] else ("died" if res["dead"] else "time up")
        fl = ", ".join(f"{f['kind']} {f['start']}-{f['end']}" for f in r["flags"] if f["kind"] not in MINOR)
        lines.append(f"| {r['name']} | {outcome} | {len(r['trace']['steps'])} | {flagged_steps(r)} | {fl or '-'} |")
    return "\n".join(lines)


def review_run(client: Creo, run: dict) -> dict:
    reply = client.chat([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": run["summary"]}],
                        max_tokens=1200, temperature=0.2, json_mode=True, tag="replay_review")
    out = parse_json_reply(reply)
    out["run"] = run["name"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dir", type=Path)
    ap.add_argument("--name", help="report name (default: trace folder name)")
    ap.add_argument("--model", help="Creo model (default: CREO_MODEL)")
    ap.add_argument("--flags-only", action="store_true", help="no LLM: rule-based report only (free)")
    ap.add_argument("--dry-run", action="store_true", help="write the requests to runs/llm/dry-run/, send nothing")
    ap.add_argument("--only-flagged", action="store_true", help="review only runs with flags other than weapon_cycling")
    ap.add_argument("--limit", type=int, help="review at most N runs (most flagged first)")
    a = ap.parse_args(argv)
    runs = load_runs(a.trace_dir)
    if not runs:
        raise SystemExit(f"no traces in {a.trace_dir} (run scripts/episode_trace.py first)")
    name = a.name or a.trace_dir.name
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    report = [f"# Replay review: {name}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{a.trace_dir.as_posix()}` "
              f"({len(runs)} runs, policy {runs[0]['trace'].get('policy', '?')}).", "", flags_report(runs)]
    if a.flags_only:
        (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"wrote {(out / 'report.md').relative_to(REPO)}")
        return 0

    client = Creo(model=a.model, dry_run=True if a.dry_run else None)
    todo = [r for r in runs if not a.only_flagged or any(f["kind"] not in MINOR for f in r["flags"])]
    todo.sort(key=lambda r: -sum(f["end"] - f["start"] + 1 for f in r["flags"] if f["kind"] not in MINOR))
    if a.limit:
        todo = todo[:a.limit]
    reviews = []
    for i, r in enumerate(todo, 1):
        rv = review_run(client, r)
        reviews.append(rv)
        print(f"[{i}/{len(todo)}] {r['name']}: {rv.get('verdict', '?')}", flush=True)
    (out / "reviews.json").write_text(json.dumps(reviews, indent=1, ensure_ascii=False), encoding="utf-8")

    counts = {}
    for r in runs:
        for f in r["flags"]:
            counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    compact = [{k: v for k, v in rv.items() if k in ("run", "verdict", "summary", "moments", "strengths", "fix_ideas")}
               for rv in reviews]
    synthesis = client.chat([{"role": "system", "content": SYNTH},
                             {"role": "user", "content": json.dumps({"flag_counts": counts, "reviews": compact},
                                                                    ensure_ascii=False)}],
                            max_tokens=1800, temperature=0.3, tag="replay_synthesis")
    report += ["", "## Reviewer summary (LLM)", "", synthesis.strip(), "", "## Per-run verdicts (LLM)", ""]
    for rv in reviews:
        report.append(f"### {rv['run']}: {rv.get('verdict', '?')}")
        if rv.get("summary"):
            report.append(rv["summary"])
        for m in rv.get("moments", []) or []:
            report.append(f"- **{m.get('steps', '?')} {m.get('kind', '')}** ({m.get('severity', '?')}): "
                          f"{m.get('viewer_sees', '')} *Likely cause:* {m.get('likely_cause', '')}")
        if rv.get("_unparsed"):
            report.append("(reply was not valid JSON; raw text in reviews.json)")
        report.append("")
    u = usage_summary()
    report.append(f"_Tokens used so far (all Creo tools): {u['prompt_tokens']:,} in, {u['completion_tokens']:,} out, "
                  f"{u['requests']} requests._")
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {(out / 'report.md').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
