# 1. Product Requirements Document (PRD)

> **One line:** Build a neural network that plays DOOM levels it has never seen: survive, explore, use keycards and reach the exit, all within tight speed limits.

---

## 1.1 The problem

Most game-playing AIs **memorise** one level. Move a wall or a monster and they fall apart.
The organisers test us on **hidden maps**, so memorising is useless. The agent has to actually *understand* its surroundings and *remember* what it has seen.

## 1.2 Who this is for

| Who | What they care about |
|-----|----------------------|
| 🤖 **Automated scorer** | Points on hidden maps. The agent never crashes and stays inside the time limits. |
| 🧑‍💻 **Desk reviewers** (Top 10) | A live demo, clean code, and team members who can explain every part |
| 🏆 **Final judges** (Top 2) | Original ideas (custom memory, curiosity, hierarchy) and a clear story |
| 👥 **Our team** | Docs that are easy to follow, fast experiments and no surprises |

## 1.3 Goals

1. **Score high on the hidden map suite** across all 4 tiers.
2. **Never lose points to crashes or timeouts.**
3. **Be clearly original**, so judges see custom-built parts and not a copied library script.
4. **Be easy to explain**, with a live dashboard that shows what the agent is thinking.

## 1.4 Non-goals (what we will NOT do)

- ❌ Hand-written if/else bots, A*/Dijkstra pathfinding or `atan2` aim scripts, because the rules ban them.
- ❌ Learning from raw screen pixels only, which is too slow to train.
- ❌ Changing the game engine at evaluation time (allowed only during training).
- ❌ Human-level speed-running. Our goal is steady points, not perfect play.
- ❌ Reading anything except the official observation (that would mean disqualification).

## 1.5 How the scorer counts points (what we optimise for)

Each map gives points for:

| Bucket | What earns points | Weight |
|--------|-------------------|--------|
| 🚪 **Exit** | Reaching the level exit, plus a small bonus for remaining health and time | **Biggest** |
| 🔑 **Progression** | Keycards picked up, doors opened, locked doors unlocked | Medium |
| 🗺️ **Exploration** | Unique areas visited (tracked secretly by the scorer; spinning in place earns nothing) | Medium |
| ⚔️ **Combat** | Enemies killed, minus a little for damage taken and dying early | Smaller |

The maps fall into 4 tiers, and **harder tiers are worth more**:
Tier 1 Combat → Tier 2 Multi-room → Tier 3 Keycard puzzle → Tier 4 Full gauntlet.

> 💡 **Key insight:** each map lasts only about **2,100–4,200 ticks**. At one decision every 4 ticks, that is **~525–1,050 decisions, or 1–2 minutes of game time**. The agent must be **efficient**, not just careful.

> 💡 **Key insight 2:** partial progress always counts. An agent that **explores well and survives** already earns good points, even when it can't solve the keycard puzzle.

## 1.6 Success metrics

We measure these on **our own held-out maps** (see [Testing Plan](05-TESTING-PLAN.md)).

| Metric | Minimum (must) | Target (want) |
|--------|----------------|---------------|
| Official `validate.py` | ✅ Passes | ✅ Passes |
| Crashes / invalid actions | 0 | 0 |
| Average decision time (1 CPU thread) | < 40 ms | **< 20 ms** (safety margin) |
| Tier 1: survive and kill | Survives most maps | Clears most arenas |
| Tier 2: area explored | Clearly better than random | Visits most rooms |
| Tier 3: keycard picked up | Sometimes | Most maps |
| Tier 3/4: exit reached | Occasionally | Regularly |
| Ablations (memory, curiosity, manager) | Each part helps | Each part helps clearly |

## 1.7 Requirements

### Functional requirements (what it must DO)

| ID | Requirement |
|----|-------------|
| FR-1 | `submission/agent.py` defines a `DoomAgent` class with `__init__(observation_space, action_space, device)`, `reset()` and `act(observation) -> np.ndarray`. The action has shape `(9,)` with values in `{0, 1}`. |
| FR-2 | `__init__` loads the weights **once**, from paths relative to `submission/`, onto `device` (`"cuda"` or `"cpu"`). |
| FR-3 | `reset()` **fully clears memory** (mind map, notebook, recurrent state) before every map. |
| FR-4 | `act()` **always** returns a valid action, even when something goes wrong inside. |
| FR-5 | The agent uses **only** the official observation dictionary. |
| FR-6 | The agent can fight, explore, pick up keycards, open doors, unlock locked doors and reach exits. |
| FR-7 | The agent works on **maps it has never seen**. |
| FR-8 | All decisions come from **trained neural network weights**. |
| FR-9 | The training code lives in the repo and reproduces our model. |

### Non-functional requirements (how WELL it must work)

| ID | Requirement | Limit |
|----|-------------|-------|
| NFR-1 | Load time | ≤ 60 s |
| NFR-2 | `reset()` time | ≤ 10 s |
| NFR-3 | First `act()` per map (warm-up) | ≤ 10 s |
| NFR-4 | Any other `act()` | ≤ 5 s each; total agent time per map ≤ **40 ms × decision steps** (e.g. 21 s for a 2,100-tick map) |
| NFR-5 | Memory | ≤ 2 GB GPU, ≤ 8 GB RAM (including `/tmp`), **1 CPU thread** |
| NFR-6 | Repo size (including Git LFS) | ≤ 1 GB |
| NFR-7 | Environment | Python 3.13, organisers' base image, no network, read-only repo (write only to `/tmp`) |
| NFR-8 | Extra packages | Only `package==version` lines in `submission/requirements.txt`, max 50, kept small |
| NFR-9 | Robustness | No crash on any map, seed or odd observation |

### Innovation requirements (what judges reward)

| ID | Requirement | Our answer |
|----|-------------|------------|
| IR-1 | Custom neural modules | Our own ray, entity and vitals encoders |
| IR-2 | Custom spatial memory | Mind map + landmark notebook |
| IR-3 | Intrinsic curiosity | Novelty reward for new places |
| IR-4 | Hierarchical control | Manager (goal) + worker (buttons) |
| IR-5 | Our own RL implementation | PPO written by us, not a Stable-Baselines3 default script |

## 1.8 Deliverables

1. Public GitHub repo **created from the official template**, ≤ 1 GB
2. `submission/agent.py` + weights (Git LFS)
3. Full training pipeline (code + configs + map tools)
4. 2-page project brief (PDF): USP, network design, memory design, training/exploration pipeline
5. Mind View demo dashboard (for the desk review and the finals)

## 1.9 Contract facts (answered by the official template)

These came from [CONTRACT.md](CONTRACT.md), [EVALUATION.md](EVALUATION.md) and `env/doom_env/`. The contract version is `1.0.0-draft`, so re-check it if organisers announce a patch.

| # | Question | Answer | What it means for us |
|---|----------|--------|----------------------|
| Q1 | Action format? | `MultiBinary(9)`: forward, back, strafe L, strafe R, turn L, turn R, attack, use, next weapon | Grouped action heads (see Design §3.4). `USE` fires only when the button goes down, and spamming it makes doors bounce. |
| Q2 | Observation keys? | `rgb` (120×160×3), `depth` (120×160), `vitals` (17), `rays` (64 × distance + hit type), `visible_entities` (32 slots × 5 features + type + mask) | Encoders sized to match |
| Q3 | Velocity frame? | Player's own frame (forward, right, up), map units per tic | Odometer works for movement. Turning has to be learned or calibrated (we can't see our heading). |
| Q4 | Entity types? | melee/hitscan/projectile enemy, **projectile**, health, armor, ammo, weapon, **key**, barrel, other | We can see keys and incoming projectiles. **Key colour is not given.** |
| Q5 | Is the exit visible in rays/entities? | **No.** Rays only report none/wall/door/locked door/enemy/item. | An **RGB encoder is needed** to spot exit signs and switches |
| Q6 | Where can agent code live? | Weights load relative to `submission/`, and the validator checks every package `submission/` imports | Brain code lives **inside `submission/smriti/`** |
| Q7 | Limits? | 60 s init, 10 s reset, 10 s first act, 5 s per act, 40 ms × steps per map | See NFR table |
| Q8 | GPU available? | `device` is `"cuda"` or `"cpu"`, 2 GB VRAM | Support both and pick by benchmark |

**Still unknown:** exact scorer weights (kept hidden on purpose), and the colour of keys and locked doors (rays and entities don't give it; RGB might).

Other useful facts:
- The entity list only covers the **90° field of view**; rays cover the full 360°.
- A ray stops at anything that **blocks movement** (including ledges higher than 24 units). Open doors let rays through.
- `env.privileged_state()` gives the true position, kills and damage **for training rewards only**.
- The scorer evaluates the **last commit on the default branch at the deadline**, so `main` must always be submittable.
- Organisers manually review the **code and replays** of the top 4–5 teams.

## 1.10 Top risks (details in [Decisions & Risks](09-DECISIONS-AND-RISKS.md))

- RL training is slow or unstable → start early, use a curriculum, keep a champion checkpoint.
- Overfitting to our training maps → large, varied map pool + held-out test maps.
- Too slow at play time → small model + latency tests on every build.
- A crash means standing still for the rest of the map → a safety wrapper around `act()`.
