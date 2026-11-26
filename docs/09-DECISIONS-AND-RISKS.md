# 9. Decisions & Risks

> **One line:** Why we chose what we chose, and what could go wrong (with a plan for each).

---

## 9.1 Decision log

Add a new row whenever we make or change a big choice. Never delete old rows; mark them *Replaced*.

| ID | Decision | Why | Alternatives we rejected | Status |
|----|----------|-----|--------------------------|--------|
| D-01 | Use **rays + entities + vitals** as the main input, not raw pixels | The organisers say pixels waste ~90% of training on edge detection; rays are fast on CPU | Pixel-only CNN | ✅ Active |
| D-02 | **PyTorch** over JAX | Team familiarity, simpler export, in the base image | JAX/Flax | ✅ Active |
| D-03 | **Write our own recurrent PPO** | Innovation points; lets us add manager, curiosity and asymmetric critic | Stable-Baselines3 default (scores poorly by rule) | ✅ Active |
| D-04 | **Mind map from odometry**, with a fallback to GRU + notebook only | A spatial memory is our USP. Phase 0 measured it on `two_rooms.wad`: button-based turn model + friction-aware velocity formula gives 0.3% drift. Re-check on larger maps; the fallback stays if they disagree. | Pure LSTM memory only | ✅ Active (measured) |
| D-05 | **Landmark notebook** with attention | Tier 3 (keycard → door) needs long-term memory of specific objects | Hoping the GRU remembers | ✅ Active |
| D-06 | **Manager/worker** with 5 named goals | Long levels need planning above button presses; easy to explain | Flat policy only (kept as ablation) | ✅ Active |
| D-07 | **Entity attention** | Handles any number of enemies; gives explainable focus | Fixed top-K list | ✅ Active |
| D-08 | **Curriculum + replay of old stages** | RL learns hard tasks much faster with steps; replay prevents forgetting | Train on full levels only | ✅ Active |
| D-09 | **OBLIGE procedural maps** as the main training source | Unlimited variety → generalises to hidden maps | Only Freedoom levels (too few) | ✅ Active |
| D-10 | **Target 20 ms**, not 40 ms | A safety margin for slower hardware and bad steps | Use the full 40 ms | ✅ Active |
| D-11 | **Safety wrapper** returning a safe action | A crash = standing still for the rest of the map | Let errors raise | ✅ Active |
| D-12 | **Champion rule**: replace only if better + 0 crashes + fast enough | `main` stays submittable | Always ship the latest | ✅ Active |
| D-13 | **Brain code lives in `submission/smriti/`** | Weights load relative to `submission/`, and the validator checks its imports; training imports the same code | Separate `smriti/` at the repo root + copy step | ✅ Active |
| D-14 | **6 grouped action heads → 9 buttons** | Blocks silly combos (forward + back); smaller choice space | 9 independent on/off heads | ✅ Active |
| D-15 | **RGB encoder is required**, not optional | The exit never appears in rays or entities, and key/door colour isn't given | Rays + entities only | ✅ Active |
| D-16 | **No direct commits to `main`**; every change goes through a PR | `main` is what the scorer evaluates at the deadline | Committing straight to `main` | ✅ Active |
| D-17 | **Mini-League scorer weights are our guesses** (`training/scorer.py`) | The real weights are hidden; we follow the published rules (exit biggest, then progression, exploration, combat; tier weights 1–4) | Waiting for real weights | ✅ Active, revisit if organisers publish more |
| D-18 | **First map library = Freedoom 2 + ViZDoom scenarios** (both ship with vizdoom) | No download needed; 45 maps; real levels with doors, keys and exits | Download community WADs / OBLIGE first | ✅ Active; OBLIGE + own maps next |
| D-19 | **Safety guard cleans every observation and action** (`smriti/safety.py`) | A crash or invalid action = standing still for the rest of the map | Trusting the policy output | ✅ Active |
| D-20 | **Run the brain on CPU even when `device="cuda"`** | Batch of 1, ~0.5M parameters: one CPU thread does a step in ~2–3 ms; the GPU is shared with other teams and adds launch/transfer cost | GPU inference | ✅ Active; revisit with image encoders |
| D-21 | **Warm the network up inside `__init__`** | PyTorch's first forward pass took 5 s, half the 10 s first-`act()` limit; `__init__` has 60 s | Warm up in the first `act()` | ✅ Active |
| D-22 | **No (or unloadable) trained weights → random policy** | `main` stays submittable before the first champion exists; a broken or un-fetched LFS `champion.pt` must not crash `__init__` (0 on every map) | Fail without weights | ✅ Active |
| D-24 | **Game workers ask Windows for 1 ms timer resolution** | Default 15.6 ms ticks made lockstep stepping ~10× slower on Windows | Accept the slowdown | ✅ Active |
| D-25 | **Reuse each training map for 8 episodes** (`--switch-every`) | A map load (~0.6 s) stalls every game in the batch | New map every 2 episodes | ✅ Active |
| D-26 | **Skip maps that fail to load instead of crashing** | The first 6M-step run died at 3.62M on one bad generated map | Fail fast | ✅ Active |
| D-27 | **One training stack: Aneek's (`training/rollout.py`, `ppo.py`, `train.py`)** | It trains the memory brain with the exact agent code path; the separate PR #8 trainer was retired after its long-run fixes were ported | Keep two trainers | ✅ Active |
| D-28 | **Warm-start memory brains from the champion** (new memory inputs start at zero) | The memory brain then starts with the champion's skills (exit ~75% at step 0 vs ~2% from scratch) and only learns to use memory | Train each memory brain from scratch | ✅ Active |
| D-29 | **More replay (40%) when warm-starting** (`--puzzle-share 0.6`) | Most of the champion's score comes from arenas and mazes; 80% key puzzles risks forgetting them | Stage 3 default (80% puzzles) | ✅ Active for `p3-mem-ws-v1` |
| D-23 | **Write our own map generator** (text drawing → UDMF) instead of depending on OBLIGE first | No extra install; maps are tiny, exact and reproducible from a seed; we control the skill being tested (e.g. key away from its door) | OBLIGE only | ✅ Active; OBLIGE stays optional for big full levels |

---

## 9.2 Risk register

**Likelihood / Impact:** 🟢 Low · 🟡 Medium · 🔴 High

| ID | Risk | Likelihood | Impact | What we do about it |
|----|------|------------|--------|----------------------|
| R-01 | RL doesn't learn / is unstable | 🟡 | 🔴 | Curriculum; reward unit tests; start with fewer buttons; watch TensorBoard; keep the champion |
| R-02 | **Overfitting** to training maps | 🟡 | 🔴 | Big varied map pool, randomisation, strict train/val/test split |
| R-03 | Too slow (> 40 ms) | 🟢 | 🔴 | Speed test on every merge; small model; optional encoders can be switched off |
| R-04 | Crash on hidden maps | 🟡 | 🔴 | Safety wrapper; chaos tests; NaN cleaning |
| R-05 | Odometer drifts | 🟢 | 🟡 | Measured 0.3% drift on the test map; re-check on big maps; mind map stores "age" so old info fades; fallback D-04 |
| R-06 | Training-time cheats leak into habits | 🟡 | 🟡 | Final stages use the normal game; Mini-League uses the normal game only |
| R-07 | Exit not visible in rays/entities (**confirmed** by the contract) | 🔴 | 🔴 | RGB encoder + "exit in view?" helper task (D-15) |
| R-08 | Repo > 1 GB | 🟢 | 🔴 | LFS only for the champion + a curated set of WADs; generated maps are git-ignored |
| R-09 | LFS weights missing in the submitted repo | 🟢 | 🔴 | Fresh-clone validator check before submission |
| R-10 | Reward hacking (e.g. spinning for novelty, door open/close loops) | 🟡 | 🟡 | Count each door/area once per episode; watch videos |
| R-11 | Rule-compliance doubts (mind map seen as "hardcoded") | 🟢 | 🟡 | RULES.md allows preprocessing and learned memory; no pathfinding/BFS; explained in the brief |
| R-12 | Not enough training compute | 🟡 | 🟡 | CPU-friendly inputs; small model; many parallel games; prioritise phases |
| R-13 | Contract changes during the event (it's `1.0.0-draft`) | 🟢 | 🟡 | Only patch fixes are allowed; watch announcements and pull official fixes into `env/` via PR |
| R-14 | Key and locked-door **colours are not given** | 🔴 | 🟡 | RGB colour guess; otherwise try every locked door after any key pickup |
| R-15 | `USE` spam makes doors bounce | 🟡 | 🟡 | Previous-action input + door-bounce penalty in training |
| R-16 | Turning isn't observed, so heading drifts | 🟢 | 🟡 | Measured on `two_rooms.wad`: turn per step is predictable from our buttons (heading error < 0.06°). Re-check on larger maps. |
| R-17 | Entities only cover the 90° view | 🟡 | 🟢 | Rays cover 360° (they report enemies too); GRU remembers enemies that left the view |
| R-22 | Training is slow on one Windows laptop (~375 steps/s with memory on ≈ 1.35M steps/hour) | 🟡 | 🟡 | Keep runs to a few million steps; use `--resume`; try more workers × fewer games; a GPU/Linux box helps the update step |
| R-21 | Odometer drifts while pinned in a corner (the engine keeps reporting velocity, but the true position does not change) | 🔴 | 🟡 | Measured: with random wall-bumping over 600 steps, the max position error per run has a median of ~550 units on the val maps. The memory crop is local and old breadcrumbs fade; fix candidates: ray matching or a contact-aware correction (System Design §3.3.2) |
| R-20 | `__init__` takes ~19 s inside the official image (limit 60 s) | 🟡 | 🔴 | Mostly importing PyTorch on 1 CPU; re-measure with `validate.py --docker` whenever the brain grows |
| R-19 | Validation/test maps are only big Freedoom levels (no Tier 1/2) | 🟢 | 🟡 | ✅ Fixed: 16 generated val + 16 test maps (arena, maze, rooms, keydoor); val is now 4/8/5/3 maps per tier |
| R-18 | Startup after `reset()` is more than a single ignored step | 🔴 | 🟢 | Step 1 is a no-op, and the steady turn/move ramp only appears after ≥2 no-op steps (held turn from the first action: 0°, 3.515625°, 14.0625°). The odometer must use `startup_noop_steps`, not just skip step 1. See `configs/odometry.json`. |
