"""Train SmritiNet with our recurrent PPO on a curriculum stage.

    # Phase 3: key & door puzzles, memory ON (mind map + notebook)
    uv run python -m training.train --stage 3 --memory on  --name s3-mem   --steps 2000000
    # ablation arm: same everything, memory OFF (GRU only)
    uv run python -m training.train --stage 3 --memory off --name s3-nomem --steps 2000000

Outputs go to runs/<name>/ (git-ignored): latest.pt, step_<n>.pt, metrics.jsonl, TensorBoard events.
Export a checkpoint to submission/weights/champion.pt only after the Mini-League says it is better.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "submission"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from smriti.brain import BrainConfig, SmritiNet, load_checkpoint  # noqa: E402
from training.checkpoints import save_checkpoint  # noqa: E402
from training.ppo import PPO, PPOConfig, Rollout  # noqa: E402
from training.rollout import VecGames  # noqa: E402


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--memory", choices=["on", "off", "map", "notebook"], default="on",
                    help="on = mind map + notebook; off = GRU only (ablation); map / notebook = one of them")
    ap.add_argument("--kinds", help="comma list of map kinds overriding the stage mix (smoke tests)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games", type=int, default=4, help="games per worker")
    ap.add_argument("--steps", type=int, default=200_000, help="total decisions (all games)")
    ap.add_argument("--T", type=int, default=128, help="rollout length per game")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01, help="entropy bonus at start (decays to ent/10)")
    ap.add_argument("--ticks", type=int, default=2100, help="episode tick cap")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--init", type=Path, help="start from this checkpoint (must have the same memory config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--save-every", type=int, default=50_000)
    ap.add_argument("--threads", type=int, default=4, help="torch threads for the learner")
    return ap.parse_args(argv)


def memory_flags(mode: str) -> tuple[bool, bool]:
    return {"on": (True, True), "off": (False, False), "map": (True, False), "notebook": (False, True)}[mode]


def main(argv=None) -> int:
    a = build_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, a.threads))
    use_map, use_notes = memory_flags(a.memory)
    name = a.name or f"s{a.stage}-{a.memory}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = REPO / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    if a.init:
        net, _ = load_checkpoint(a.init, dev)
        if (net.cfg.use_mindmap, net.cfg.use_notebook) != (use_map, use_notes):
            raise SystemExit(f"--init checkpoint memory config {net.cfg} does not match --memory {a.memory}")
    else:
        net = SmritiNet(BrainConfig(use_mindmap=use_map, use_notebook=use_notes)).to(dev)
    net.eval()
    cfg = PPOConfig(T=a.T, lr=a.lr, ent_coef=a.ent, epochs=a.epochs, minibatches=a.minibatches)
    kinds = a.kinds.split(",") if a.kinds else None
    vec = VecGames(a.workers, a.games, a.stage, a.seed, use_map, use_notes, episode_ticks=a.ticks, kinds=kinds)
    n = vec.n
    ppo = PPO(net, cfg, n, dev)
    print(f"[train] {name}: stage {a.stage}, memory {a.memory}, {n} games, "
          f"{sum(p.numel() for p in net.parameters()):,} params", flush=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(str(out))
    except Exception:  # noqa: BLE001  (tensorboard is optional)
        tb = None
    metrics = (out / "metrics.jsonl").open("a", encoding="utf-8")
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2), encoding="utf-8")

    feats = vec.first
    state = net.initial_state(n, dev)
    starts = np.ones(n, np.bool_)
    recent = deque(maxlen=100)
    steps, it, t_start, next_save = 0, 0, time.time(), a.save_every
    try:
        while steps < a.steps:
            progress = steps / a.steps
            buf = Rollout(cfg.T, n)
            h0 = state.clone()
            t0 = time.time()
            for t in range(cfg.T):
                for k in buf.f:
                    buf.f[k][t] = feats[k]
                buf.start[t] = starts
                choices, logp, v, state = ppo.act(feats, state, starts)
                feats, r, done, infos = vec.step(choices, progress)
                buf.a[t], buf.logp[t], buf.v[t] = choices, logp, v
                buf.r[t] = ppo.rscale.scale(r, done)
                buf.done[t] = done
                starts = done.copy()
                recent.extend(i for i in infos if i is not None)
            _, _, last_v, _ = ppo.act(feats, state, starts)
            last_v = last_v * (1.0 - starts)  # a fresh episode's value does not bootstrap the old one
            t_roll = time.time() - t0
            t0 = time.time()
            stats = ppo.update(buf, h0, last_v, lr=a.lr * (1.0 - 0.9 * progress),
                               ent_coef=a.ent * (1.0 - 0.9 * progress))
            t_upd = time.time() - t0
            steps += cfg.T * n
            it += 1
            row = {"steps": steps, "it": it, "sps": round(cfg.T * n / (t_roll + t_upd), 1),
                   "t_roll": round(t_roll, 2), "t_upd": round(t_upd, 2), **{k: round(v, 5) for k, v in stats.items()}}
            if recent:
                for key in ("ret", "len", "exit", "keys", "locked_doors", "doors", "areas", "kills", "dead", "score"):
                    row[f"ep_{key}"] = round(float(np.mean([e[key] for e in recent])), 3)
                row["episodes"] = len(recent)
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            if tb is not None:
                for k, v in row.items():
                    if isinstance(v, (int, float)) and k not in ("steps", "it"):
                        tb.add_scalar(k, v, steps)
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            extra = {"steps": steps, "stage": a.stage, "memory": a.memory, "name": name}
            save_checkpoint(out / "latest.pt", net, extra)
            if steps >= next_save:
                save_checkpoint(out / f"step_{steps}.pt", net, extra)
                next_save += a.save_every
    finally:
        vec.close()
        metrics.close()
        if tb is not None:
            tb.close()
    print(f"[train] done: {steps} steps in {time.time() - t_start:.0f}s -> {out / 'latest.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
