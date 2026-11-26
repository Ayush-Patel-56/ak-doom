"""Train SmritiNet with our recurrent PPO on the curriculum (docs/04-TRAINING-PLAN.md).

    uv run python scripts/train.py --name stage12-v1 --steps 5000000 --stages 1:0,2:1500000
    uv run python scripts/train.py --name stage12-v1 --resume          # continue a run
    uv run tensorboard --logdir runs                                     # watch curves

--stages "1:0,2:1500000" = stage 1 from step 0, stage 2 from step 1.5M (training/curriculum.py).
Outputs go to runs/<name>/: latest.pt (+ ckpt_NNNM.pt every 1M steps), train_state.pt, tb/, log.txt.
Then: scripts/evaluate.py --weights runs/<name>/latest.pt, and scripts/export_champion.py.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "submission"), str(REPO / "scripts")]

from training.envpool import EnvPool, fine_timer  # noqa: E402
from training.ppo import PPOConfig, PPOTrainer  # noqa: E402


def parse_stages(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        stage, frm = part.split(":")
        out.append((int(float(frm)), int(stage)))
    out.sort()
    if out[0][0] != 0:
        raise SystemExit("--stages must start at step 0, e.g. 1:0,2:1500000")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=float, default=5e6, help="total environment steps")
    ap.add_argument("--stages", default="1:0,2:1500000")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="start from these brain weights (fine-tuning)")
    ap.add_argument("--episodes-per-map", type=int, default=4)
    for f in fields(PPOConfig):  # expose every PPO setting, e.g. --lr 1e-4
        if f.name not in ("num_envs", "total_steps"):
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default) if f.type != "bool" else int,
                            default=None)
    args = ap.parse_args()

    overrides = {f.name: getattr(args, f.name) for f in fields(PPOConfig) if getattr(args, f.name, None) is not None}
    cfg = PPOConfig(num_envs=args.envs, total_steps=int(args.steps), **overrides)
    run_dir = REPO / "runs" / args.name
    log_file = None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    fine_timer()
    schedule = parse_stages(args.stages)
    pool = EnvPool(cfg.num_envs, stage=schedule[0][1], seed=args.seed, episodes_per_map=args.episodes_per_map)
    try:
        trainer = PPOTrainer(pool, cfg, run_dir, init_weights=args.init, log=log)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(run_dir / "log.txt", "a", encoding="utf-8")
        if args.resume and trainer.resume():
            log(f"resumed {args.name} at step {trainer.steps:,}")
        params = sum(p.numel() for p in trainer.net.parameters())
        log(f"run {args.name}: {cfg.num_envs} envs, {params:,} params, target {cfg.total_steps:,} steps, stages {schedule}")
        trainer.train(schedule)
        log(f"done at step {trainer.steps:,}: {run_dir / 'latest.pt'}")
    finally:
        pool.close()
        if log_file:
            log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
