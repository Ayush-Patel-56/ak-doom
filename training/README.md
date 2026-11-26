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
Long runs: `--resume` continues `runs/<name>/` (weights, optimizer, reward scale, step count); maps are reused for
`--switch-every 8` episodes; a map that fails to load is skipped; on Windows the workers ask for 1 ms timer
resolution (~3× faster training here).

Export a checkpoint to `submission/weights/champion.pt` (Git LFS) only after the Mini-League says it is better:
`uv run python scripts/export_champion.py runs/<name>/latest.pt --eval runs/eval/<name>.json` (checks it loads,
gives valid actions, stays under 20 ms, had 0 Mini-League crashes and beats the current champion).
On the NVIDIA machine add `--device cuda`; the games always run on CPU.
