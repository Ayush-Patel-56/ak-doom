"""Mock desk review: an LLM (hackathon Creo API) plays the organiser who quizzes the team after the demo.

Stage 2 of the hackathon is a live demo plus technical questions "to verify your implementation and
understanding". This tool reads OUR repo (brief, README, Q&A notes, experiment log, key code) and either
quizzes one person interactively or writes a question bank with model answers for the whole team.

    uv run python scripts/mock_review.py bank --questions 25            # Markdown Q&A bank (one request)
    uv run python scripts/mock_review.py quiz --questions 6             # interactive: ask, you answer, it grades
    uv run python scripts/mock_review.py quiz --topic memory            # focus one area
    uv run python scripts/mock_review.py bank --dry-run                 # check the prompt, send nothing

Answers in `quiz` are typed in the terminal; finish an answer with an empty line. Transcripts go to
runs/llm/mock-review/. Testing tool only: never used by the agent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from creo import Creo, parse_json_reply  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "llm" / "mock-review"

# (path, max characters): the brief and Q&A notes in full, the rest trimmed to fit a mid-size context
CONTEXT_FILES = [
    ("docs/brief/BRIEF.md", 14000),
    ("README.md", 6000),
    ("docs/08-DEMO-AND-BRIEF.md", 9000),
    ("docs/09-DECISIONS-AND-RISKS.md", 9000),
    ("docs/experiments.md", 9000),
    ("docs/07-ROADMAP.md", 7000),
    ("docs/02-SYSTEM-ARCHITECTURE.md", 7000),
    ("submission/agent.py", 6000),
    ("submission/smriti/brain.py", 9000),
    ("submission/smriti/memory/episode.py", 3500),
    ("submission/smriti/memory/mindmap.py", 3500),
    ("submission/smriti/memory/notebook.py", 3500),
    ("training/ppo.py", 4000),
    ("training/rewards.py", 4000),
    ("training/curriculum.py", 3500),
]
TOPICS = {
    "all": "any part of the project",
    "sensing": "observations, encoders, rays, entity attention, why not raw pixels",
    "memory": "odometer, mind map, landmark notebook, GRU, how memory works without coordinates, drift",
    "training": "own PPO, rewards, curriculum, warm start, maps, exploration and curiosity, compute",
    "rules": "rule compliance: no hardcoded rules or pathfinding, training-only mods, observations only, original work",
    "results": "Mini-League, validation vs test, ablation honesty, Phase 4 not beating Phase 3, weaknesses",
    "engineering": "speed limits, safety guard, determinism, Docker validation, failure handling",
}

JUDGE = """You are an organiser on the technical desk-review panel of a hackathon track "Can It Play DOOM?"
(ViZDoom, Freedoom 2, hidden unseen maps in 4 tiers: combat, multi-room exploration, keycard -> locked door
-> exit, full levels). Teams must show a trained neural agent (no hardcoded if/else rules, pathfinders or
atan2 aimbots), their own training pipeline, a spatial memory design and an exploration method. Innovative
custom architectures are rewarded; plain off-the-shelf RL scripts are not. You verify that the team built
and understands what they claim, and you probe weaknesses politely but firmly.

You know the team's project only from the material below (their repo). Never invent facts about it.

=== TEAM MATERIAL ===
{context}
=== END ==="""


def build_context(budget_chars: int) -> str:
    parts, used = [], 0
    for rel, cap in CONTEXT_FILES:
        p = REPO / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        take = min(cap, budget_chars - used)
        if take <= 500:
            break
        chunk = text[:take] + ("\n[... trimmed ...]" if len(text) > take else "")
        parts.append(f"--- {rel} ---\n{chunk}")
        used += len(chunk)
    return "\n\n".join(parts)


def bank(client: Creo, context: str, n: int, topic: str) -> str:
    ask = (f"Write {n} questions you would really ask this team in the desk review, focused on {TOPICS[topic]}. "
           "Mix easy checks with hard probing ones (how exactly does X work, why this choice, what fails, "
           "prove it is your own work, what the numbers mean). For each, give a strong model answer that is "
           "true to the team material (3-6 sentences, concrete numbers where the material has them) and one "
           "likely follow-up question. Format as Markdown: '### Q<n>. <question>' then '**Model answer:** ...' "
           "then '**Follow-up:** ...'.")
    return client.chat([{"role": "system", "content": JUDGE.format(context=context)},
                        {"role": "user", "content": ask}], max_tokens=6000, temperature=0.4, tag="mock_bank")


def read_answer() -> str:
    print("Your answer (finish with an empty line):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def quiz(client: Creo, context: str, n: int, topic: str, log: list[str]) -> list[dict]:
    system = JUDGE.format(context=context)
    asked: list[str] = []
    results = []
    for i in range(1, n + 1):
        q = client.chat([{"role": "system", "content": system},
                         {"role": "user", "content": f"Ask ONE new desk-review question about {TOPICS[topic]}. "
                          f"Do not repeat these: {json.dumps(asked)}. Reply with the question only."}],
                        max_tokens=200, temperature=0.7, tag="mock_question").strip()
        asked.append(q)
        print(f"\nQ{i}. {q}")
        answer = read_answer()
        if not answer:
            print("(skipped)")
            log += [f"### Q{i}. {q}", "_skipped_", ""]
            continue
        grade_prompt = (f"Question: {q}\nTeam member's answer: {answer}\n\nGrade the answer against the team "
                        "material. Reply with ONLY JSON: {\"score\": 0-10, \"correct\": [\"...\"], "
                        "\"missing_or_wrong\": [\"...\"], \"ideal_answer\": \"<3-5 sentences>\", "
                        "\"follow_up\": \"<the question you would ask next>\"}")
        g = parse_json_reply(client.chat([{"role": "system", "content": system},
                                          {"role": "user", "content": grade_prompt}],
                                         max_tokens=900, temperature=0.2, json_mode=True, tag="mock_grade"))
        results.append(dict(question=q, answer=answer, **g))
        print(f"  score {g.get('score', '?')}/10")
        for m in g.get("missing_or_wrong", []) or []:
            print(f"  - missing/wrong: {m}")
        if g.get("ideal_answer"):
            print(f"  ideal: {g['ideal_answer']}")
        if g.get("follow_up"):
            print(f"  follow-up: {g['follow_up']}")
        log += [f"### Q{i}. {q}", f"**Answer:** {answer}", f"**Score:** {g.get('score', '?')}/10",
                "**Missing or wrong:** " + "; ".join(g.get("missing_or_wrong", []) or ["-"]),
                f"**Ideal answer:** {g.get('ideal_answer', '-')}", f"**Follow-up:** {g.get('follow_up', '-')}", ""]
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["bank", "quiz"])
    ap.add_argument("--questions", type=int, default=None, help="bank: 20, quiz: 6 by default")
    ap.add_argument("--topic", choices=sorted(TOPICS), default="all")
    ap.add_argument("--context-chars", type=int, default=90000, help="material sent to the model (~4 chars/token)")
    ap.add_argument("--model")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    client = Creo(model=a.model, dry_run=True if a.dry_run else None)
    context = build_context(a.context_chars)
    print(f"team material: {len(context):,} characters (~{len(context) // 4:,} tokens)")
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = a.name or f"{a.mode}-{a.topic}-{stamp}"
    if a.mode == "bank":
        text = bank(client, context, a.questions or 20, a.topic)
        path = OUT / f"{name}.md"
        path.write_text(f"# Desk-review question bank ({a.topic})\n\nGenerated {stamp} by `{client.model or 'dry run'}` "
                        f"from the repo; check every answer against the code before relying on it.\n\n{text}\n",
                        encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")
        return 0
    log = [f"# Mock desk review ({a.topic}), {stamp}", ""]
    results = quiz(client, context, a.questions or 6, a.topic, log)
    scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    if scores:
        log.insert(1, f"Average score: {sum(scores) / len(scores):.1f}/10 over {len(scores)} answers.\n")
        print(f"\naverage score {sum(scores) / len(scores):.1f}/10")
    path = OUT / f"{name}.md"
    path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"transcript: {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
