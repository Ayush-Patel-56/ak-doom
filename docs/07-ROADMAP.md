# 7. Roadmap

> **One line:** Build in layers. Every phase ends with an agent that is **submittable**, so we always have something to hand in.

No dates here, only order. Each phase has a clear **"Done when"**.

```mermaid
flowchart LR
    P0["Phase 0<br/>Setup"] --> P1["Phase 1<br/>Walking skeleton"]
    P1 --> P2["Phase 2<br/>Baseline brain"]
    P2 --> P3["Phase 3<br/>Memory ⭐"]
    P3 --> P4["Phase 4<br/>Manager + curiosity"]
    P4 --> P5["Phase 5<br/>Generalise"]
    P5 --> P6["Phase 6<br/>Polish & present"]
```

> 💡 **Start training early.** Training takes the longest, so get Phase 2 training running as soon as possible and build Phases 3–4 while it runs.

---

## Phase 0 — Setup

- [x] Repo `ak-doom` on GitHub, built on the official template
- [x] Read `CONTRACT.md` → facts recorded in the [PRD §1.9](01-PRD.md#19-contract-facts-answered-by-the-official-template)
- [x] Planning docs merged (PR #1)
- [x] Setup works on Windows (`uv sync`, `validate.py` passes; see §6.1 for Windows notes). Every team member: [Repo & Workflow §6.1](06-REPO-AND-WORKFLOW.md#61-setting-up-step-by-step) (`uv sync`, `validate.py`, `pytest`)
- [x] Build the Docker image and run `validate.py --docker` once: **passed** on `main` after Phase 1 (1 CPU pinned, 44 MB RAM, ~0.4 ms per step)
- [x] Run the test map and **print one full observation** (`scripts/inspect_observation.py`)
- [x] Calibrate the odometer (`scripts/measure_odometry.py` → `configs/odometry.json`): turn model; production-estimator drift is **3.85%** on `two_rooms.wad` (seed 1 in `configs/odometry.json`). Held-out seed 733, 600 steps per map: final position drift is **0.40%** on `two_rooms.wad`, **0.68%** on the Tier 2 maze, and **4.02%** on the Tier 3 key/door map (see System Design §3.3.1 and `configs/odometry-validation.json`)
- [x] Add `.gitignore` lines and the `maps/curated/**` LFS rule
- [x] Protect `main` on GitHub: require a pull request before merging

**Done when:** everyone can run a game locally, and `main` only changes through PRs.

---

## Phase 1 — Walking skeleton (a submittable agent)

- [x] `submission/agent.py` keeps the template's random baseline, plus the safety wrapper (`submission/smriti/safety.py`)
- [x] `submission/smriti/` package wired into `agent.py` (random policy in `smriti/policy/`)
- [x] Contract + unit tests (`tests/agent/`, 30 tests) and `validate.py` passing
- [x] First version of `scripts/evaluate.py` (Mini-League; scorer in `training/scorer.py`)
- [x] First version of `scripts/benchmark_latency.py`
- [x] Map tools: `scripts/build_map_catalog.py` → `maps/catalog.json` + `maps/lists/{train,val,test}.txt` (Freedoom 2 + ViZDoom scenarios, 45 maps)
- [x] Our own small Tier 1/2/3 maps for val/test (`training/mapgen/`, 16 val + 16 test maps)
- [ ] OBLIGE generated full levels (needs a separate install; optional, see Training Plan §4.5)

**Done when:** a random agent passes the validator and gets a (low) Mini-League score. **We have a baseline.**

---

## Phase 2 — Baseline brain + our PPO

- [x] Ray, entity (attention) and vitals encoders (`submission/smriti/brain.py`)
- [x] GRU memory (256)
- [x] Simple flat policy (6 action groups → 9 buttons) + critic; `BrainPolicy` in the agent, random fallback without weights
- [x] Our recurrent PPO + parallel game runner (`training/ppo.py`, `training/rollout.py`)
- [x] Reward calculator (§4.2) + helpers (`training/rewards.py`)
- [x] Curriculum Stage 1 (combat) and Stage 2 (navigation) (`training/curriculum.py`); first 6M-step run: val **+2,021** vs random −201
- [x] TensorBoard logging + checkpoint → Mini-League → champion export (`scripts/export_champion.py`); `--resume` for long runs
- [x] First champion exported (`p2-stage12-v1` final, +2,021 on val): `submission/weights/champion.pt` + `champion.json`; passes `validate.py --docker` (init 7.4 s, 1.2 ms per decision, 968 MB RAM)

**Done when:** the champion clearly beats random on Tier 1 and Tier 2 validation maps and stays within the speed budget.

---

## Phase 3 — Memory system ⭐ (our USP)

- [x] Odometer (+ real-engine training-ground-truth tests; measured drift and limitations in System Design §3.3.1)
- [x] Mind map (grid update + rotated crop + small CNN): `smriti/memory/mindmap.py`, `MapEncoder`
- [x] Landmark notebook (slots + attention from the GRU state): `smriti/memory/notebook.py`, `NotebookReader`
- [x] Curriculum Stage 3 (key & door puzzles): `training/curriculum.py`, new `keynear` layout (key near spawn → far `keydoor` over the run, 20% replay), 4 + 4 held-out `t3-keynear` val/test maps
- [x] Ablation pipeline: with vs without memory (`scripts/ablation.py`, `--memory on|off`); first short CPU run logged in `docs/experiments.md`. A long GPU run is still needed for the *Done when* criterion
- [ ] Long Phase 3 training run: `p3-mem-ws-v1` (memory on, warm-started from champion #1, 4.5M steps, 40% replay); then Mini-League vs the champion and export if better

**Done when:** the agent picks up keycards and opens matching locked doors on held-out Tier 3 maps, and the ablation shows memory helps.

---

## Phase 4 — Manager/worker + stronger curiosity

- [ ] Manager (5 goals, every 8 decisions) + goal-conditioned worker
- [ ] Goal helper rewards
- [ ] RND curiosity (optional)
- [ ] RGB encoder + "exit in view?" auxiliary task (rays and entities never show the exit)
- [ ] Depth encoder (if speed allows)
- [ ] Curriculum Stage 4 (full levels)
- [ ] Ablations: − manager, − curiosity

**Done when:** exit rate on Tier 3/4 validation maps improves over Phase 3.

---

## Phase 5 — Generalise and harden

- [ ] Stage 5: mixed + heavily randomised maps
- [ ] Bigger, more varied map pool
- [ ] Teacher–student training (stretch)
- [ ] Auxiliary tasks (stretch)
- [ ] Stress & chaos tests (§5.7) all passing
- [ ] Speed tuning (TorchScript, etc.) → < 20 ms average

**Done when:** the score on **test maps** (used for the first time here) is close to the validation score, meaning no overfitting.

---

## Phase 6 — Polish & present

- [ ] Final champion exported + fresh-clone `validate.py --docker` check
- [ ] Final PR merged to `main` **before the deadline** (the scorer evaluates the last commit on the default branch)
- [ ] Mind View dashboard
- [ ] Demo videos (good runs + honest failure cases)
- [ ] Ablation chart
- [ ] 2-page brief PDF ([outline](08-DEMO-AND-BRIEF.md#82-the-2-page-brief-outline))
- [ ] Q&A practice: every member can explain every part
- [ ] [Compliance checklist](10-COMPLIANCE-CHECKLIST.md) fully ticked

**Done when:** submitted, and the demo runs without problems.

---

## If things go badly: fallback ladder

| If… | Then… |
|-----|-------|
| PPO won't learn at all | Simplify: fewer buttons, Stage 1 only, check rewards with unit tests |
| Odometer is unreliable | Drop the mind map and keep the notebook + GRU (Decision D-04) |
| Manager makes things worse | Ship the flat policy and keep the manager as an ablation result |
| Too slow | Shrink the image encoders (keep a tiny RGB for the exit), shrink the GRU, use TorchScript |
| Running out of time | Submit the latest champion. Every phase ends submittable. |
