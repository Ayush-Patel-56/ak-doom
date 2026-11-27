# AK-Doom: Smriti (स्मृति, "memory")

**Project Brief** · Yugaantar DevForge, Track 1 "Can It Play DOOM?" · Team LLF

> **A DOOM agent that draws its own map and remembers where the locked door was.**
> It senses only what a player could sense, gets no coordinates, and every button press comes from one small neural network.

<!--
This Markdown file is the source of truth for the brief text. brief.html is the print layout
(2 A4 pages, page break before section 5) and brief.pdf is rendered from it with headless Edge.
All values are filled in (team LLF; champion #2 results; one-time test-map run).
Sources: docs/experiments.md, docs/03-SYSTEM-DESIGN.md, docs/04-TRAINING-PLAN.md, submission/smriti/.
-->

---

## Page 1: what it is and how it thinks

### 1. Our USP

- **Spatial memory without coordinates.** The game never says where the agent is. Smriti estimates it with a *calibrated odometer*, draws a *mind map* of walls, doors and visited places, and keeps a *landmark notebook* of keys and doors. That is what Tier 3 needs: *"I have the key; where was that locked door?"*
- **Every action comes from the network.** The mind map and notebook are preprocessing and learned memory, which the rules allow. There is **no pathfinding, no BFS and no `atan2` aiming**.
- **Built from scratch, measured honestly.** Our own recurrent PPO (no RL library), procedural map generator and "Mini-League" evaluator (scores are only compared on the same map set). A learned manager/worker, RND curiosity and image encoders are built as optional switches (Phase 4).

### 2. What it sees / what it does

- **Sees:** ego-centric only, **no global position**: **64 rays** all around (distance + hit type: wall, door, locked door, enemy, item), **up to 32 visible entities** (distance, direction, speed, type) and **17 vitals** (health, armour, ammo, key flags, own velocity).
- **Does:** one decision every **4 game ticks**. The 9 buttons are chosen as **6 action groups** (move, strafe, turn, attack, use, next weapon), one option per group, so contradictions like forward + back are impossible.

### 3. Neural architecture: SmritiNet

```text
INPUTS                  ENCODERS                               FUSE     MEMORY      DECIDE                      OUTPUT
64 rays (360°)       -> circular 1-D conv over the ring -> 128 --+
0-32 entities        -> entity attention (learned query +    ----+
                        "nothing here" token) -> 128             |
17 vitals + prev act -> vitals MLP -> 32 ------------------------+-> fuse -> GRU 256 -+-> [manager: 1 of 5 goals / 8 steps]*
[RGB / depth]*       -> [small CNNs + exit-in-view head]* -------+   256      ^   |   |          | goal
own buttons + velocity -> odometer -> mind map (256x256, 8 layers,|          |   |   +-> worker policy (6 groups, -> 9 buttons
                        rotated 32x32 crop -> CNN -> 128) -------+          |   |   |    15 logits)                (every 4 ticks)
                     -> landmark notebook (16 slots) -- attention read -----+   |   +-> value head (training only)
                                    ^------------- query = GRU state ----------+
* = built, optional Phase 4 parts
```

Grey = Phase 2 brain (**557,632 parameters**); blue = Phase 3 spatial memory (memory brain: **749,968 parameters**); dashed = built, optional Phase 4 parts. Rays are also drawn into the mind map at the odometer's pose.

- **Rays:** a **circular 1-D convolution** wraps around the 360° ring and spots shapes like "corridor ahead" or "door on the left".
- **Entities:** an MLP per entity, then **attention with a learned query** plus a learned **"nothing here" token**, so 0 to 32 entities all work.
- **Vitals + previous action:** an MLP (USE only fires when the button goes *down*, so the brain must know what it pressed).
- Everything is fused into a **GRU (256)**; the policy head gives **15 logits = 6 action groups**, the value head is used only in training.

### 4. Spatial memory design (Phase 3)

1. **Odometer: "how did I move?"** Turning is predicted from our own buttons: a held turn rotates **7.03°, then 12.30°, then 14.06°** per step (measured in the engine). Movement uses the reported velocity with Doom's friction: **step distance = 2.86·v_now + 1.56·v_prev**. The sum is our pose relative to the spawn point; drift on fresh held-out runs was **0.4–0.7% of the distance** on open maps and mazes and **~4%** on a key-door map (wall contact makes it worse).
2. **Mind map: "what have I seen?"** A spawn-anchored grid (256 × 256 cells of 32 map units; 8 layers: free, wall, door, locked door, item, enemy, visited, recently visited). Every ray is drawn in each step. The brain reads a **32 × 32 crop rotated so "up" = the facing direction** through a small CNN, so it always sees the map "through its own eyes".
3. **Landmark notebook: "what must I remember?"** 16 slots for **keys, locked doors and doors** (spawn-frame position, age, done flag). Each step they become distance and direction *from where the agent is now*, read by **attention with the query from the GRU state**: after picking up a key, the brain can "ask" for the locked door.

All memory is wiped at the start of every map, and the **same memory code** runs in the agent and in training.

---

## Page 2: how it learns and how well it plays

### 5. Training pipeline

- **Our own recurrent PPO** (no RL library): GAE advantages, clipped objective, the GRU replayed from the memory stored at the start of each training sequence, and reward normalisation.
- **Rewards mirror the scorer:** exit, keys, doors, locked doors, new areas and kills earn reward; damage, death and getting stuck cost reward. A **count-based novelty bonus** pays for new places.
- **Parallel game workers.** On Windows the default 15.6 ms timer made lock-step stepping ~10× slower; requesting 1 ms timer resolution took training from **~110 to ~375–450 steps/s**.
- **Warm start.** The memory brain copies *every* weight of the Phase 2 champion and starts its new memory inputs at zero, so it begins playing exactly like the champion (tested) and only has to learn *how to use* memory.

### 6. Exploration & generalisation

- **Own procedural maps** (text drawing → UDMF): arenas (T1), mazes and rooms (T2), key-door puzzles (T3: `keynear`, key close; `keydoor`, key in another branch, so the agent must remember the door and backtrack).
- **Seed ranges never overlap:** train 0–99,999, validation 100,000+, test 200,000+. Test maps are never used for decisions.
- **Curriculum:** stage 1 combat → stage 2 navigation → stage 3 key & door (`keynear` → `keydoor`), with **replay of older maps** so earlier skills are not forgotten.
- **Built (Phase 5):** stage 5 mixed / randomised maps, teacher distillation, a dynamics auxiliary task, and a guard that records any use of test maps.

### 7. Results: Mini-League

The Mini-League is our copy of the published scoring rules (exit, progression, exploration, combat) with **guessed weights** and tier weights ×1 to ×4. Validation = **24 maps** (4 Freedoom 2 levels + 20 generated), 2 seeds each = 48 runs. Totals are only compared on the same map set (map fingerprint + weights SHA-256).

| Agent (24-map validation set) | Steps | Total | T1 | T2 | T3 | T4 | Crashes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Champion #1 (Phase 2 brain) | 6.0 M | +2,511.03 | +498.1 | +1,390.4 | +721.5 | −98.9 | 0 / 48 |
| Memory brain, early check | 0.39 M | +2,463.58 | +446.7 | +879.7 | +1,200.4 | −63.2 | 0 / 48 |
| **Champion #2 (memory brain)** | 4.5 M | **+5,182.69** | +526.1 | +1,949.6 | +2,736.4 | −29.3 | 0 / 48 |

*Random agent on the same 24 maps: −381.25 (0 exits). **Held-out test maps** (25 maps × 4 seeds, run once at the end, after the brain was fixed): champion #2 **+4,990.69**, 0 / 100 crashes, ≈90% of its validation score per run.*

- **Champion #1** (6 M steps, stages 1–2) reaches the exit on every arena and maze, but on 0 of 8 runs of the full key-door maps.
- **Champion #2** is the memory brain warm-started from champion #1 and trained on stage 3 for 4.5 M steps. It **doubles** champion #1's total and is better on every tier: it solves **all 4 `keynear` maps** on both seeds (key → locked door → exit; champion #1: 1 of 4), one far-key `keydoor` map, and **5 of 8** doored-room runs (was 1 of 8), while keeping every arena and maze. An early check (387k steps) had briefly forgotten mazes; replay brought them back.

**Memory ablation:** same PPO and seed, ~1 M steps per arm, held-out Tier 3 maps (40 episodes per split). Memory on vs GRU only:

| Measure | Memory on | GRU only |
|---|---:|---:|
| Key picked up (val) | 45% | 27.5% |
| Deaths (val; lower is better) | 12.5% | 62.5% |
| Exit reached (test) | 45% | 12.5% |
| Score (val) | 56.9 | 19.9 |

*One training seed only: we treat this as evidence, not proof.* (In `brief.html` this is drawn as a small bar chart.)

### 8. Speed & robustness

Official Docker image, **1 CPU pinned**, median time per decision: champion #1 **1.2–2.7 ms**, memory brain **2.1 ms**, full Phase 4 brain **4.8 ms**, champion #2 2.4 ms; budget ~40 ms. Start-up ≤ ~14 s (limit 60 s); RAM < 1 GB.

**Stress tests:** 600 chaos observations × 4 brain types and a 20-map soak with flat speed and memory use: **0 fallbacks** to the safety action.

### 9. Honest limitations & next steps

- **Big Freedoom levels** still score worse than random (T4 negative): the agent rushes in and dies.
- **Doored rooms**: 5 of 8 exits; **far-key `keydoor`**: 1 of 4 solved.
- **Phase 4** (manager/worker, RND curiosity, RGB "exit in view" head), trained on a GPU from champion #2, matches it but does not beat it on 4 seeds (+5,077 vs +5,309), so it is not submitted.
- Most results are **single training seeds**. Next: 3 seeds per ablation arm and longer runs.
