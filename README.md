# AK-Doom

**A DOOM-playing AI with its own memory: mind map, landmark notebook and curiosity-driven RL on ViZDoom.**

> **Agent name:** *Smriti* (स्मृति), which means "memory".
> **Track:** Yugaantar 2026 · DevForge — Track 1, *"Can It Play DOOM?"*
> **Built on:** the official [`devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template).
> **Status:** 📝 Planning. The agent in `submission/agent.py` is still the template's random baseline.

## What is this?

Smriti is an AI player for DOOM. It learns only by playing the game, with no recorded data and no hand-written moves. Then it plays **maps it has never seen**: it fights monsters, explores rooms, finds keycards, remembers where the locked doors are and reaches the exit.

## The idea in one picture

```mermaid
flowchart LR
    A["👀 What the agent sees<br/>(rays, enemies, health...)"] --> B["🧠 Understand<br/>(encoders)"]
    B --> C["📒 Remember<br/>(mind map + notebook)"]
    C --> D["🎯 Decide the goal<br/>(manager)"]
    D --> E["🎮 Press buttons<br/>(worker)"]
    E --> F["DOOM game"]
    F --> A
```

## What makes us different (our USP)

1. **It draws its own map in its head.** The agent builds a "mind map" of the places it has been, even though the game never gives it a map.
2. **It keeps a notebook of important things.** When it sees a locked door or a keycard, it writes down where that thing is and goes back for it later.
3. **It thinks in two levels.** A *manager* picks the goal ("explore", "fight", "go get the key"), and a *worker* handles the movement and shooting.
4. **It is curious.** It gets an extra reward for visiting new places, so it doesn't get stuck in corners.
5. **You can watch it think.** A live *Mind View* dashboard shows its map, its notebook and its current goal while it plays.

## Quick start

Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # installs vizdoom, gymnasium and the official wrapper (env/)
uv run python validate.py    # official pre-submission check
uv run pytest                # tests
```

Closest check to the real scorer (needs Docker):

```bash
docker build -t devforge-doom .
python validate.py --docker --image devforge-doom
```

Planned commands (added as the code lands):

```bash
uv run python scripts/train.py --stage 1      # train
uv run python scripts/evaluate.py             # our own Mini-League on held-out maps
uv run python scripts/benchmark_latency.py    # check the 40 ms per-decision budget
```

## Repository layout

```
env/                  official ViZDoom wrapper (doom_env): don't edit (the scorer ignores it)
maps/                 template test map + our training map tools
submission/agent.py   DoomAgent: the evaluation entry point
submission/smriti/    the agent's brain (encoders, memory, policy)
submission/weights/   champion weights (Git LFS)
training/             training pipeline (PPO, rewards, curiosity, curriculum)
scripts/              train / evaluate / benchmark entry points
dashboard/            Mind View demo
validate.py           official validator: run before every merge
tests/                template tests + our tests
docs/                 official docs + our planning docs
```

## Documentation

**Official (from the template; read these first):**

| Doc | What's inside |
| --- | --- |
| [CONTRACT.md](docs/CONTRACT.md) | Observation/action spaces and the `DoomAgent` interface |
| [RULES.md](docs/RULES.md) | What's allowed and what gets you disqualified |
| [SUBMISSION.md](docs/SUBMISSION.md) | Submission checklist, Git LFS, 1 GB limit, `requirements.txt` |
| [EVALUATION.md](docs/EVALUATION.md) | Map tiers, scoring, hardware, timeouts |

**Ours (read in order; each answers one question):**

| # | Document | Question it answers |
|---|----------|---------------------|
| 1 | [PRD](docs/01-PRD.md) | What are we building, and how do we know it's good? |
| 2 | [System Architecture](docs/02-SYSTEM-ARCHITECTURE.md) | What are the big parts, and how do they connect? |
| 3 | [System Design](docs/03-SYSTEM-DESIGN.md) | How does each part work inside? |
| 4 | [Training Plan](docs/04-TRAINING-PLAN.md) | How does the agent learn? |
| 5 | [Testing Plan](docs/05-TESTING-PLAN.md) | How do we prove it works? |
| 6 | [Repo & Workflow](docs/06-REPO-AND-WORKFLOW.md) | Where does the code go, and how do we work together? |
| 7 | [Roadmap](docs/07-ROADMAP.md) | In what order do we build it? |
| 8 | [Demo & Brief](docs/08-DEMO-AND-BRIEF.md) | How do we show it off and write the 2-page PDF? |
| 9 | [Decisions & Risks](docs/09-DECISIONS-AND-RISKS.md) | Why did we choose this, and what could go wrong? |
| 10 | [Compliance Checklist](docs/10-COMPLIANCE-CHECKLIST.md) | Are we following every rule? |
| — | [Glossary](docs/GLOSSARY.md) | What do these words mean? |
| — | [Experiment Log](docs/experiments.md) | What did each training run teach us? |

## Rules in one minute

- Actions must come from **trained neural parameters**: no rule trees, no `A*`/Dijkstra/BFS, no `atan2` aimbots, no unmodified off-the-shelf SB3.
- Change anything you want **during training**. Evaluation uses the **vanilla engine**, the official wrapper and **unseen maps**.
- The core architecture and training pipeline are built during the hackathon. Libraries are fine.

## Next steps

See [Roadmap Phase 0](docs/07-ROADMAP.md).
