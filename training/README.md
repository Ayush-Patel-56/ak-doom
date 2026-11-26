# Training

Put your training pipeline here: environment setup, reward shaping, curriculum, model code, logs.
Judges review this folder during manual review.

You may modify the engine, maps and game variables freely **for training**. Evaluation always uses
the vanilla engine through the official `doom_env` wrapper, so train on `doom_env` observations.

## What is here (AK-Doom)

| File | What |
|------|------|
| `mapgen/` | Procedural UDMF maps: `arena`, `maze`, `rooms`, `keydoor`, `keynear` (Stage 3 warm-up) |
| `scorer.py` | Our copy of the scoring rules (Mini-League) |
| `rewards.py` | Training reward = change in each scorer counter × weight, plus step / stuck / USE-spam helpers |
| `curriculum.py` | Stage → map pool; Stage 3 moves from `keynear` to `keydoor` during the run, 20% replay |
| `rollout.py` | Parallel games (spawned processes); each game runs the same `EpisodeMemory` + feature code as `DoomAgent` |
| `ppo.py` | Our recurrent PPO (GAE, clipped losses, whole-sequence minibatches) |
| `train.py` | Entry point: `uv run python -m training.train --stage 3 --memory on --name s3-mem` |
| `checkpoints.py` | Atomic checkpoint saving |

Outputs go to `runs/<name>/` (git-ignored): `latest.pt`, `step_<n>.pt`, `metrics.jsonl`, TensorBoard events
(`uv run tensorboard --logdir runs`). Memory ablation: `uv run python scripts/ablation.py run --steps 2000000`.
Add memory to a trained brain without starting over: `--warm-start <brain without memory>` (new memory inputs start at
zero, so it plays exactly like that brain at step 0); `--puzzle-share` sets the key-puzzle share of stage 3 (rest = replay).

Long runs: `--resume` continues `runs/<name>/` (weights, optimizer, reward scale, step count); maps are reused for
`--switch-every 8` episodes; a map that fails to load is skipped; on Windows the workers ask for 1 ms timer
resolution (~3× faster training here).

Export a checkpoint to `submission/weights/champion.pt` (Git LFS) only after the Mini-League says it is better:
`uv run python scripts/export_champion.py runs/<name>/latest.pt --eval runs/eval/<name>.json` (checks it loads,
gives valid actions, stays under 20 ms, had 0 Mini-League crashes and beats the current champion).
On the NVIDIA machine add `--device cuda`; the games always run on CPU.

## Phase 4 features

Train the learned five-goal manager (default interval: eight decisions), goal-conditioned worker,
goal helper rewards, RGB encoder/exit head, and optional RND on the existing Stage 3 map pool:

```bash
uv run python -m training.train --stage 3 --memory on --manager on --rgb on \
  --goal-helpers on --rnd on --workers 1 --games 2 --threads 2 \
  --steps 4096 --ticks 400 --name phase4-smoke
```

All new switches default off, preserving older architectures. This is a smoke configuration, not a
recommended training budget for a champion. Depth, Stage 4 maps, and manager/curiosity ablations are described below. The five goal identities are sampled from neural logits; no rules pick them.

Worker PPO replays the exact stored goals. Manager PPO updates only on completed goal intervals,
using discounted **extrinsic** rewards and the actual interval length (including terminal intervals).
Pending intervals cross rollout boundaries. Manager heads optimize detached boundary GRU features;
the shared encoders/GRU learn through the worker and visual auxiliary losses. Manager decisions are
not differentiated through categorical sampling. Value-only bootstrap never advances cadence/RNG.

`goal_rewards.py` adds event-based worker bonuses, with default weights Explore=.02 per new scorer
area, Fight=.05 per kill, Collect=.05 per item/key or positive health/ammo inventory event, Go to landmark=.05 per door opening,
Go to exit=.1 per successful exit. These are training hyperparameters, not action rules. Override
weights with `--helper-weights '{"explore":0.01}'`, or disable helpers entirely. Collection avoids
double-counting progress also present in item/key counters. Inventory baselines advance under every
goal, and ammo expenditure never earns a collection bonus. Real ammo pickups are tested because
they can leave the engine's item counter unchanged.

`rnd.py` uses legal rays/vitals only. The target is frozen and seeded; predictor error is normalized
using preceding statistics and capped at 5 before weighting. `--rnd-coef .01` sets the initial
intrinsic weight, decaying linearly to zero over the requested run. Terminal/reset replacement
observations are excluded. Networks, predictor optimizer, and normalization moments are restored by
`--resume`. Intrinsic, helper, and extrinsic rewards are logged separately.

RGB is real rendered uint8 data, downsampled to 60x80. `exit_visibility.py` derives training labels
from actual generated-map exit actions and floor geometry, live camera/door heights, and renderer
flat/sprite masks. It enables the labels buffer before the first reset without changing the official
wrapper or actor observation. Supported maps use the generator's flat 64-unit geometry and the
pinned 160x120 software renderer with zero camera pitch/roll. Unsupported or ambiguous frames are
masked, not labeled negative. Exit identity is independent of floor texture. Labels and privileged
state never enter actor features; `--exit-aux .1` weights the masked binary supervision loss.

Check `exit_labels_known` **and** `exit_labels_positive`: an all-negative smoke run cannot establish
an exit detector, even if its loss is low. Real-engine tests additionally train the visual head on
both known positive and negative views and verify the loss falls.

Use identical feature/reward flags with `--resume`; mismatches fail explicitly. Model, optimizer,
normalization, and learner RNG states are restored. Games restart, so unfinished manager intervals
and per-game discounted-return accumulators reset. This is run continuation, not an exact engine
snapshot. Expanding an older model requires explicit `--warm-start path.pt`; copied and newly
initialized parameter names are printed. Newly initialized heads are not presented as trained.

Inference exposes `DoomAgent.active_goal` and `exit_probability` for future dashboard consumers.
RND, helper rewards, label geometry, and their optimizer state are training-only. Export uses the
existing inference checkpoint path; do not promote smoke weights to champion without held-out
evaluation. See `configs/phase4-smoke.json` for measured smoke-run evidence and limitations.


## Depth and full-level curriculum

`--depth on` adds a small CNN over real rendered depth, downsampled to 1×60×80 and
normalized from uint8 by 255. Values are renderer depth codes, not metric distances.
RGB and depth independently enable image buffers in training and inference. Depth defaults off;
use `--warm-start path.pt` to expand older models while preserving their initial outputs.

`--stage 4 --full-level-share .8` samples usable **train** Tier 4 Freedoom2 levels from
`maps/catalog.json`, with 20% generated-map replay. Catalog identities are checked for split
aliases (including physical file aliases); invalid metadata, empty pools and missing assets fail.
Full levels use catalog episode tick caps; replay uses 2100 ticks. `--ticks N` is an explicit
smoke override. `--catalog PATH` selects a catalog. `curriculum.json` and checkpoints record the
pool, catalog hash, mixture and tick override. Resume rejects changed provenance; Stage 4 and
depth runs require saved provenance. Engine state itself is restarted on resume.

```bash
uv run python -m training.train --stage 4 --memory on --manager on --rgb on \
  --depth on --rnd on --warm-start submission/weights/champion.pt --name stage4-depth
```

## Controlled Phase 4 comparisons

```bash
uv run python scripts/phase4_ablation.py run --name phase4-comparison \
  --source submission/weights/champion.pt --depth on --rgb on
```

The full, no-manager, no-curiosity and Phase 3 architecture arms share initialization,
training pool, seed and decision budget. Goal helpers are off in every arm to isolate the manager.
The default 1,999,872 decisions is divisible by the default rollout batch; custom budgets must be
exact multiples of T×workers×games. Repeat `--train-seed` for paired training seeds.
Only held-out Tier 3/4 maps are evaluated. Strict checkpoint/config/budget validation prevents
random fallback or mislabeled arms; raw episodes, errors, hashes and paired differences are saved
under `runs/ablation/`. `--train-args` accepts only non-controlled training settings.
`eval` supports explicit `--arm name=checkpoint` inputs; use `--help` for all options.

`configs/depth-stage4-smoke.json` preserves measured pipeline evidence. Each arm trained for only
512 decisions; all 16 shortened evaluation episodes had zero exits. This verifies execution,
not policy effectiveness. Champion weights remain unchanged. Mac latency measurements are not
an official pinned Linux scorer benchmark.


## Phase 5, Part 1: varied training and hardening

`--stage 5` mixes all usable **training** catalog tiers with generated arena, maze, rooms,
keydoor and keynear maps. The default full-level share is 0.5 (`--full-level-share` overrides it).
Generated seeds span [0, 100000); validation/test seed namespaces remain excluded. Versioned
randomisation varies safe player starts/headings, enemy and item density/composition, verified
Freedoom textures and light. Key-door puzzles are checked for solvability and key bypasses.
Generated caps vary deterministically from 2100 to 4200 ticks; catalog caps remain unchanged.
Skill stays at 3. Profile hashes appear in cache filenames and curriculum/checkpoint provenance;
resume rejects changed provenance. `--ticks` remains an explicit smoke override.

```bash
uv run python -m training.train --stage 5 --memory on --manager on --rgb on --depth on \
  --rnd on --warm-start PATH_TO_PHASE4_CHECKPOINT --name stage5
uv run python scripts/stress_agent.py --weights runs/stage5/latest.pt \
  --output runs/stage5-stress.json
uv run python scripts/benchmark_latency.py --weights runs/stage5/latest.pt \
  --map maps/two_rooms.wad:MAP01 --json runs/stage5-latency.json
```

Stress defaults to 20 actual generated maps, each capped at 4200 ticks. It covers dense enemies,
empty maps, a constrained spawn, repeated resets, invalid/extreme observations, latency and RSS
where Linux /proc is available. Thresholds are declared in arguments before execution; behavioral
failures remain failures. A successful Mac reliability run does not establish RSS or official pinned
CPU compliance. The benchmark rejects missing checkpoints/random fallback. TorchScript is not
required: retain optimisations only when equivalence and measured speed justify them.

Validation evaluation and final comparison:

```bash
uv run python scripts/generalisation.py run --weights runs/stage5/latest.pt \
  --split val --seeds 3 --output runs/stage5-val.json
# Final evaluation only after training/model selection is frozen:
uv run python scripts/generalisation.py run --weights runs/stage5/latest.pt \
  --split test --final-test --seeds 3 --output runs/stage5-test.json
uv run python scripts/generalisation.py compare --validation runs/stage5-val.json \
  --test runs/stage5-test.json --gap-tolerance .1 --min-mean-score 10 \
  --output runs/stage5-gap.json
```

The gap and useful-score values above are **example acceptance settings**, not measured results
or official scorer thresholds. Declare suitable values before inspecting final test outcomes.
Comparison requires matching checkpoint/configuration, tier/seed composition and tick overrides.
Default minimum evidence is 30 episodes per tier; inspect map diversity and seed variation too.
Project evaluation, ablation and benchmark commands recognize catalog aliases and byte-identical WAD copies. Known test maps require explicit opt-in and a persisted use record under `runs/final-test-ledger/`;
repeated reservations require `--allow-repeat-test`. This ledger records future use; historical test
freshness is unknown. Keep/copy the ledger with experiments and do not delete it to claim first use.
Development smokes use train/validation maps only. No score-gap claim is made without final evidence.

## Phase 5, Part 2: teacher and auxiliary learning

```bash
uv run python -m training.train --stage 5 --memory on --manager on --rgb on --depth on \
  --warm-start PATH_TO_PHASE4_CHECKPOINT --teacher PATH_TO_TRAINED_TEACHER \
  --distill-coef .1 --distill-temperature 2 --dynamics-coef .1 --name stage5-student
```

The frozen teacher consumes legal observations from actual student-visited trajectories. It owns
its recurrent state, goal cadence and sampling generator; its action-group distributions supply KL
supervision. Hidden sizes and modalities may differ. Rollouts enable the union of required image
and deterministic spatial-memory features; the teacher's recurrent notebook queries remain its own.
The spatial history reflects the **student-executed** actions for both agents. Teacher parameters
receive no gradients. Checkpoint hash, configuration, positive training-step provenance, loss weight
and temperature are recorded and verified on resume. Provenance is required; it is not evidence that
a teacher is capable. This implements legal-observation distillation; a privileged full-map teacher
still requires its own observation contract and trained weights.

`--dynamics-coef` enables a training-only head predicting changes in legal ray proximity and normalized
vitals from recurrent features plus executed action groups. Targets come from actual consecutive
observations. Terminal/reset replacements are masked. The head is unused during inference; older
checkpoints default it off. Expand existing actors with explicit `--warm-start`, and retain the same
architecture/loss configuration on resume. Distillation and dynamics default off.

Measured evidence will be stored in `configs/phase5-smoke.json`. Pipeline smokes do not establish
teacher superiority, final test generalisation or champion quality; champion weights are unchanged.

The official `validate.py` is preserved unchanged and does not enforce this project experiment ledger.
Use it on train/validation maps during development; manually invoking it on test maps is outside the
project runner guard. This guard is experiment discipline, not a security sandbox against arbitrary
ViZDoom code. Teacher and student sharing deterministic spatial features does not share recurrent state.
The benchmark reports requested and effective devices separately; the submitted agent intentionally
uses CPU inference even when its constructor receives CUDA. Constructor timing excludes checkpoint
preflight and ledger scans. Completed training episodes are appended to `episodes.jsonl`, including
actual engine seed, WAD/map identity, cap and randomisation metadata.
