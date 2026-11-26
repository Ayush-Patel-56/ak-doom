# 10. Compliance Checklist

> **One line:** Every rule from the official Track 1 problem statement, as a tick-box list. Check it before every major submission.

---

## Repository
- [ ] Public GitHub repo, URL sent to organisers before the deadline
- [ ] **Built on the official starter template** ([`devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template))
- [ ] Created **during the hackathon**
- [ ] Total size **≤ 1 GB** including Git LFS objects
- [ ] Model weights under `submission/` via **Git LFS** (`git lfs ls-files` lists them)
- [ ] Nothing under `submission/` is git-ignored, untracked or uncommitted
- [ ] Training code is in the repo (`training/`; reviewers read it)
- [ ] Final work merged to the **default branch** before the deadline (the scorer uses its last commit)

## Agent interface
- [ ] File `submission/agent.py` exists
- [ ] Defines a **stateful** class `DoomAgent`
- [ ] `__init__(self, observation_space, action_space, device)` loads weights **once**, relative to `submission/`
- [ ] `reset(self)` clears memory before every map
- [ ] `act(self, observation)` returns an array of shape `(9,)` with values in `{0, 1}` (one decision every 4 ticks)

## Environment & dependencies
- [ ] We don't submit our own Dockerfile and don't edit `env/` (the scorer ignores it)
- [ ] Works in the official base image: Python 3.13, vizdoom 1.3.3, gymnasium, torch, jax/flax, numpy, scipy, opencv-headless
- [ ] Extra packages (if any) in `submission/requirements.txt`: only `package==version` lines, max 50, no URLs/paths/options
- [ ] No network access needed at runtime
- [ ] Never writes to the repo (read-only); scratch files only in `/tmp` (counts toward RAM)

## Limits
- [ ] Load ≤ 60 s
- [ ] `reset()` ≤ 10 s
- [ ] First `act()` per map ≤ 10 s
- [ ] Any other `act()` ≤ 5 s
- [ ] Total agent time per map ≤ **40 ms × decision steps**
- [ ] ≤ 2 GB VRAM, ≤ 8 GB RAM, **1 CPU thread**
- [ ] Checked against the exact values in `env/doom_env/limits.py`
- [ ] `python validate.py` passes, ideally `python validate.py --docker`

## Ground rules
- [ ] Decisions come from **trained neural parameters**
- [ ] **No** hardcoded if/else rule trees for decisions
- [ ] **No** A*, Dijkstra, BFS over a map or other classical pathfinding/planning
- [ ] **No** `atan2`/trigonometry aimbots
- [ ] Engine mods / cheats used **only in training**; the agent works on the unmodified game
- [ ] Agent uses **only the given observations**
- [ ] No attempts to access the engine, evaluation maps, network or host
- [ ] No deliberate hogging of GPU, CPU or disk
- [ ] Open-source libraries allowed, but **core architecture + training pipeline are our own**

## Brief
- [ ] 2-page PDF (max 2 pages)
- [ ] Covers USP, neural architecture, spatial memory design, training/exploration pipeline

## Final sanity
- [ ] Fresh `git clone` → LFS weights download → `validate.py` passes
- [ ] Mini-League: 0 crashes
- [ ] Latest champion is the one in `submission/weights/`
