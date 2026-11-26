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
recommended training budget for a champion. Depth, Stage 4 maps, and manager/curiosity ablations are
separate roadmap work. The five goal identities are sampled from neural logits; no rules pick them.

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
