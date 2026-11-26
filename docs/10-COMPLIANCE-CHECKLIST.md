# 10. Compliance Checklist

> **One line:** Every rule from the official Track 1 problem statement, as a tick-box list. Check it before every major submission.

**Last audited:** 2026-11-27 on `main` @ `01b6ca9` (after #17), with champion #1 in `submission/weights/`.
**Result:** 37 of 41 items verified ✅. Still open: sending the URL, the final merge before the deadline, and the 2-page brief (two items). Re-audit after every new champion and right before submitting.

---

## Repository
- [x] Public GitHub repo: `Ayush-Patel-56/ak-doom` is **PUBLIC**
- [ ] Repo URL sent to organisers before the deadline (team action)
- [x] **Built on the official starter template**: GitHub reports it as generated from `Reinforce-SST/devforge-doom-template`
- [x] Created **during the hackathon**: repo created 2026-11-26 09:12 UTC; every commit is dated 2026-11-26 or 2026-11-27
- [x] Total size **≤ 1 GB** including Git LFS objects: **4.7 MB** (`validate.py`), of which 3.6 MB in 41 LFS files
- [x] Model weights under `submission/` via **Git LFS**: `git lfs ls-files` lists `submission/weights/champion.pt` (`3196b6098e`)
- [x] Nothing under `submission/` is git-ignored, untracked or uncommitted: `git status --ignored` and `git check-ignore` are clean; `validate.py` checks it too
- [x] Training code is in the repo: `training/` (16 modules: rollout, PPO, rewards, curriculum, map generator, warm start, distillation, …) + `scripts/`
- [ ] Final work merged to the **default branch** (`main`) before the deadline (the scorer uses its last commit)

## Agent interface
- [x] File `submission/agent.py` exists
- [x] Defines a **stateful** class `DoomAgent` (recurrent memory, mind map, notebook, odometer)
- [x] `__init__(self, observation_space, action_space, device)` loads weights **once**, relative to `submission/` (`WEIGHTS_DIR / "champion.pt"`)
- [x] `reset(self)` clears memory before every map: `test_reset_clears_per_map_state_and_is_reproducible`, `test_reset_twice_in_a_row_is_fine`
- [x] `act(self, observation)` returns an array of shape `(9,)` with values in `{0, 1}`: contract tests + 600 chaos inputs × 4 brain types (`tests/agent/stress/`)

## Environment & dependencies
- [x] We don't submit our own Dockerfile and don't edit `env/`: `env/`, `Dockerfile`, `docker/` and `validate.py` are **byte-identical** to the template's initial commit `2fa44e6`
- [x] Works in the official base image: `validate.py --docker --image devforge-doom` **passed** on `01b6ca9`
- [x] Extra packages in `submission/requirements.txt`: **none** (0 packages)
- [x] No network access needed at runtime: the Docker check runs with `--network none`; no network imports in `submission/`
- [x] Never writes to the repo: the Docker check mounts it read-only; contract test `test_submission_never_writes_files`

## Limits (official Docker image, 1 CPU pinned, `01b6ca9`)
- [x] Load ≤ 60 s: **9.9 s**
- [x] `reset()` ≤ 10 s: well under (benchmarks report ~0.000 s)
- [x] First `act()` per map ≤ 10 s: **5 ms** (the network is warmed up in `__init__`)
- [x] Any other `act()` ≤ 5 s: worst **7 ms**
- [x] Total agent time per map ≤ **40 ms × decision steps**: **1.5 s of 21 s** on the 2,100-tick map (p50 2.7 ms per decision)
- [x] ≤ 2 GB VRAM, ≤ 8 GB RAM, **1 CPU thread**: inference runs on **CPU** (`INFERENCE_DEVICE = "cpu"`, so no VRAM used; Windows can't measure it per process); peak RAM **747 MB**; `torch.set_num_threads(1)`
- [x] Checked against the exact values in `env/doom_env/limits.py` (unchanged from the template: 60 s / 10 s / 10 s / 5 s / 40 ms / 2,048 MB / 8,192 MB / 1 thread / 1 GB)
- [x] `python validate.py` passes **and** `python validate.py --docker` passes

## Ground rules
- [x] Decisions come from **trained neural parameters**: `act()` returns `guard.act(observation, policy)` where the policy is the trained SmritiNet (champion: 6,000,640 training steps). The only non-learned paths are safety fallbacks: a fixed safe action if the network raises, and the random policy if no weights load
- [x] **No** hardcoded if/else rule trees for decisions (see above)
- [x] **No** A*, Dijkstra, BFS over a map or other classical pathfinding/planning: no `heapq`, `astar`, `dijkstra`, `bfs`, `deque` or flood fill anywhere in `submission/` (the mind map is an input to the network, not a planner)
- [x] **No** `atan2`/trigonometry aimbots: no `atan2`/`arctan` anywhere in `submission/`
- [x] Engine mods / cheats used **only in training**: `submission/` never imports `vizdoom`, `doom_env` or `training/` (contract test `test_submission_never_imports_training_code`)
- [x] Agent uses **only the given observations** (same test; privileged state exists only in `training/`)
- [x] No attempts to access the engine, evaluation maps, network or host: no `vizdoom`, `socket`, `urllib`, `requests`, `subprocess` or `os.system` in `submission/`; the test-map guard (`scripts/test_use.py`) keeps evaluation off held-out test maps during development
- [x] No deliberate hogging of GPU, CPU or disk: one CPU thread, no GPU use at inference, ~750 MB RAM, no file writes
- [x] Open-source libraries allowed, but **core architecture + training pipeline are our own**: SmritiNet, memory, PPO, curriculum and map generator are all written in this repo (history from 2026-11-26); libraries: PyTorch, NumPy, ViZDoom, Gymnasium

## Brief
- [ ] 2-page PDF (max 2 pages)
- [ ] Covers USP, neural architecture, spatial memory design, training/exploration pipeline

## Final sanity
- [x] Fresh `git clone` → LFS weights download → `validate.py` passes: clone of `main` @ `01b6ca9` got the real 2,242,253-byte `champion.pt` and passed
- [x] Mini-League: 0 crashes: champion #1, 24 val maps × 2 seeds: **0 of 48** runs stopped (`champion1-val24`; reproduced exactly under #17's code: +2,511.03)
- [x] Latest champion is the one in `submission/weights/`: `champion.json` records its source, SHA-256 and +2,511.03 on map set `8fda02b8a085ba01`
