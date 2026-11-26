"""Train SmritiNet with our recurrent PPO on a curriculum stage.

    # Phase 3: key & door puzzles, memory ON (mind map + notebook)
    uv run python -m training.train --stage 3 --memory on  --name s3-mem   --steps 2000000
    # ablation arm: same everything, memory OFF (GRU only)
    uv run python -m training.train --stage 3 --memory off --name s3-nomem --steps 2000000
    # continue an interrupted run (same name; restores weights, optimizer, reward scale and step count)
    uv run python -m training.train --stage 3 --memory on  --name s3-mem   --steps 2000000 --resume

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

from smriti.brain import BrainConfig, SmritiNet, load_checkpoint, VITALS_SCALE
from smriti.features import NUM_RAYS, NUM_VITALS
from training.rnd import RND  # noqa: E402
from training.checkpoints import save_checkpoint  # noqa: E402
from training.ppo import PPO, PPOConfig, Rollout  # noqa: E402
from training.rollout import VecGames, fine_timer  # noqa: E402


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, default=3, choices=[1, 2, 3, 4, 5])
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
    ap.add_argument("--ticks", type=int, default=None, help="explicit tick override; default: catalog cap for full levels, 2100 for replay")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--init", type=Path, help="start from this checkpoint (must have the same memory config)")
    ap.add_argument("--warm-start", type=Path,
                    help="transfer shared weights from a checkpoint; new encoder inputs start at zero, "
                         "new hierarchy heads initialize normally (training/warmstart.py)")
    ap.add_argument("--puzzle-share", type=float,
                    help="share of games on key & door puzzles (default: the stage's own, 0.8 for stage 3); "
                         "the rest replays older maps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--save-every", type=int, default=50_000)
    ap.add_argument("--threads", type=int, default=4, help="torch threads for the learner")
    ap.add_argument("--switch-every", type=int, default=8,
                    help="episodes per map before loading a new one (a map load stalls every game ~0.6 s)")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name> from latest.pt + train_state.pt")
    for flag in ('manager', 'rgb', 'depth', 'rnd', 'goal-helpers'):
        ap.add_argument('--' + flag, choices=['on', 'off'], default='off')
    ap.add_argument('--goal-interval', type=int, default=8)
    ap.add_argument('--helper-weights', type=json.loads, help='JSON object overriding goal helper weights')
    ap.add_argument('--rnd-coef', type=float, default=.01, help='initial intrinsic weight; linearly decays to zero')
    ap.add_argument('--exit-aux', type=float, default=.1)
    ap.add_argument('--full-level-share', type=float, default=None)
    ap.add_argument('--teacher', type=Path)
    ap.add_argument('--distill-coef', type=float, default=0.)
    ap.add_argument('--distill-temperature', type=float, default=1.)
    ap.add_argument('--catalog', type=Path, default=REPO / 'maps/catalog.json')
    a = ap.parse_args(argv)
    if not np.isfinite(a.distill_coef) or a.distill_coef < 0 or not np.isfinite(a.distill_temperature) or a.distill_temperature <= 0:
        ap.error('distillation coefficient must be nonnegative and temperature positive (finite)')
    if a.distill_coef and a.teacher is None:
        ap.error('--distill-coef requires --teacher checkpoint')
    if a.ticks is not None and a.ticks < 1:
        ap.error('--ticks must be positive')
    if a.full_level_share is not None and (not np.isfinite(a.full_level_share) or not 0 <= a.full_level_share <= 1):
        ap.error('--full-level-share must be in [0, 1]')
    if any(getattr(a, k) < 1 for k in ('workers','games','steps','T','epochs','minibatches','threads','goal_interval','save_every','switch_every')):
        ap.error('counts and goal interval must be positive')
    if a.goal_helpers == 'on' and a.manager != 'on':
        ap.error('goal helpers require --manager on')
    if a.warm_start and (a.init is not None or a.resume):
        ap.error('use --warm-start PATH separately from --init or --resume')
    if not all(np.isfinite(v) and v >= 0 for v in (a.rnd_coef, a.exit_aux, a.ent)) or not np.isfinite(a.lr) or a.lr <= 0:
        ap.error('loss/reward weights must be finite and nonnegative; lr must be positive')
    if a.helper_weights is not None:
        from training.goal_rewards import GoalHelpers
        try:
            if not isinstance(a.helper_weights, dict):
                raise ValueError('helper weights must be a JSON object')
            GoalHelpers(a.helper_weights)
        except (ValueError, TypeError) as exc:
            ap.error(str(exc))
    return a


def memory_flags(mode: str) -> tuple[bool, bool]:
    return {"on": (True, True), "off": (False, False), "map": (True, False), "notebook": (False, True)}[mode]


def save_train_state(path: Path, ppo: PPO, steps: int, it: int, next_save: int) -> None:
    """Everything --resume needs besides the weights (written atomically next to latest.pt)."""
    rs = ppo.rscale
    tmp = path.with_suffix(".tmp")
    torch.save({"opt": ppo.opt.state_dict(), "rscale": {"mean": rs.mean, "var": rs.var, "count": rs.count,
                                                         "ret": rs.ret.copy()},
                "manager_opt": ppo.manager_opt.state_dict() if ppo.manager_opt else None,
                "rnd": ppo.rnd.state_dict() if ppo.rnd else None,
                "training_config": ppo.training_config,
                "curriculum_config": ppo.curriculum_config,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "steps": steps, "it": it, "next_save": next_save}, tmp)
    tmp.replace(path)


def load_train_state(path: Path, ppo: PPO, n_games: int) -> dict:
    st = torch.load(path, map_location="cpu", weights_only=False)
    saved_config = st.get('training_config')
    legacy_config = dict(rnd='off', rnd_coef=.01, helpers='off', helper_weights=None, exit_aux=.1)
    if (saved_config if saved_config is not None else legacy_config) != ppo.training_config:
        raise ValueError('--resume training feature/reward config differs from saved run')
    if 'curriculum_config' in st and st['curriculum_config'] != ppo.curriculum_config:
        raise ValueError('--resume curriculum/tick configuration differs from saved run')
    needs_provenance = ppo.net.cfg.use_depth or ppo.curriculum_config['pool']['stage'] >= 4
    if 'curriculum_config' not in st and needs_provenance:
        raise ValueError('Stage 4/5/depth resume requires saved curriculum provenance')
    ppo.opt.load_state_dict(st["opt"])
    if ppo.manager_opt is not None:
        ppo.manager_opt.load_state_dict(st['manager_opt'])
    if ppo.rnd is not None:
        ppo.rnd.load_state_dict(st['rnd'])
    if 'torch_rng' in st:
        torch.set_rng_state(st['torch_rng'])
        np.random.set_state(st['numpy_rng'])
    rs = ppo.rscale
    rs.mean, rs.var, rs.count = st["rscale"]["mean"], st["rscale"]["var"], st["rscale"]["count"]
    # Environments restart on resume; old per-game returns and unfinished manager intervals do not.
    rs.ret = np.zeros(n_games, np.float64)
    return st


def main(argv=None) -> int:
    a = build_args(argv)
    fine_timer()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, a.threads))
    use_map, use_notes = memory_flags(a.memory)
    name = a.name or f"s{a.stage}-{a.memory}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = REPO / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    resume_state = out / "train_state.pt"
    if a.resume:
        if not ((out / "latest.pt").is_file() and resume_state.is_file()):
            raise SystemExit(f"--resume: no latest.pt + train_state.pt in {out}")
        a.init = out / "latest.pt"
    requested = BrainConfig(use_mindmap=use_map, use_notebook=use_notes,
                            use_manager=a.manager == 'on', use_rgb=a.rgb == 'on', use_depth=a.depth == 'on',
                            goal_interval=a.goal_interval)
    transferred = None
    if a.init:
        net, _ = load_checkpoint(a.init, dev)
        flags = ('use_mindmap', 'use_notebook', 'use_manager', 'use_rgb', 'use_depth', 'goal_interval')
        if any(getattr(net.cfg, k) != getattr(requested, k) for k in flags):
            raise SystemExit('--init architecture does not match flags; use --warm-start PATH to expand it')
    else:
        net = SmritiNet(requested).to(dev)
        if a.warm_start:
            from training.warmstart import warm_start_from
            source, _ = load_checkpoint(a.warm_start, dev)
            transferred = warm_start_from(net, source)
            print('[train] explicit warm start: ' + json.dumps(transferred), flush=True)
    net.eval()
    cfg = PPOConfig(T=a.T, lr=a.lr, ent_coef=a.ent, epochs=a.epochs, minibatches=a.minibatches, exit_aux_coef=a.exit_aux,
                    distill_coef=a.distill_coef, distill_temperature=a.distill_temperature)
    kinds = a.kinds.split(",") if a.kinds else None
    from training.curriculum import MapSampler
    curriculum = MapSampler(a.stage, a.seed, kinds=kinds, puzzle_share=a.puzzle_share,
                            full_level_share=a.full_level_share, catalog_path=a.catalog).pool_manifest
    n = a.workers * a.games
    ppo = PPO(net, cfg, n, dev)
    ppo.rnd = RND(NUM_RAYS + NUM_VITALS, dev, seed=a.seed) if a.rnd == 'on' else None
    ppo.training_config = dict(rnd=a.rnd, rnd_coef=a.rnd_coef, helpers=a.goal_helpers,
                               helper_weights=a.helper_weights, exit_aux=a.exit_aux)

    teacher = None
    if a.distill_coef:
        from training.distill import FrozenTeacher
        teacher = FrozenTeacher(a.teacher, n, dev, seed=a.seed+9187)
        ppo.training_config['distillation'] = dict(teacher=teacher.provenance, coefficient=a.distill_coef,
                                                  temperature=a.distill_temperature)
    ppo.curriculum_config = {'pool': curriculum, 'ticks_override': a.ticks}
    resumed = load_train_state(resume_state, ppo, n) if a.resume else None
    vec = VecGames(a.workers, a.games, a.stage, a.seed, use_map or (teacher is not None and teacher.net.cfg.use_mindmap),
                   use_notes or (teacher is not None and teacher.net.cfg.use_notebook), episode_ticks=a.ticks,
                   switch_every=a.switch_every, kinds=kinds, use_rgb=net.cfg.use_rgb or (teacher is not None and teacher.net.cfg.use_rgb),
                   goal_helpers=a.goal_helpers == 'on', helper_weights=a.helper_weights, puzzle_share=a.puzzle_share,
                   use_depth=net.cfg.use_depth or (teacher is not None and teacher.net.cfg.use_depth), full_level_share=a.full_level_share, catalog_path=a.catalog)

    print(f"[train] {name}: stage {a.stage}, memory {a.memory}, {n} games, "
          f"{sum(p.numel() for p in net.parameters()):,} params", flush=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(str(out))
    except Exception:  # noqa: BLE001  (tensorboard is optional)
        tb = None
    metrics = (out / "metrics.jsonl").open("a", encoding="utf-8")
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2), encoding="utf-8")

    (out / 'curriculum.json').write_text(json.dumps(curriculum, indent=2), encoding='utf-8')
    feats = vec.first
    state = net.initial_state(n, dev)
    starts = np.ones(n, np.bool_)
    recent = deque(maxlen=100)
    steps, it, t_start, next_save = 0, 0, time.time(), a.save_every
    if resumed:
        steps, it, next_save = resumed["steps"], resumed["it"], resumed["next_save"]
        print(f"[train] resumed {name} at step {steps:,}", flush=True)
    try:
        while steps < a.steps:
            progress = steps / a.steps
            buf = Rollout(cfg.T, n)
            if teacher is not None:
                from smriti.policy.actions import NUM_LOGITS
                buf.teacher_logits = np.zeros((cfg.T,n,NUM_LOGITS),np.float32)
            h0 = state.clone()
            t0 = time.time()
            rnd_inputs = []
            reward_components = []
            label_counts = [0, 0]
            for t in range(cfg.T):
                for k in buf.f:
                    buf.f[k][t] = feats[k]
                buf.start[t] = starts
                buf.exit_target[t] = feats['_exit_target']
                buf.exit_known[t] = feats['_exit_known']
                label_counts[0] += int(feats['_exit_known'].sum())
                label_counts[1] += int((feats['_exit_target'] * feats['_exit_known']).sum())
                if teacher is not None:
                    buf.teacher_logits[t] = teacher.logits(feats,starts).cpu().numpy()
                choices, logp, v, state = ppo.act(feats, state, starts)
                goals = ppo.goals.goal.cpu().numpy() if net.cfg.use_manager else None
                if goals is not None:
                    buf.goals[t] = goals
                feats, r, done, infos = vec.step(choices, progress, goals)
                if ppo.intervals is not None:
                    ppo.intervals.step(feats['_extrinsic'], done)
                intrinsic = np.zeros(n, np.float32)
                if ppo.rnd is not None and (~done).any():
                    legal = np.concatenate([feats['ray_dist'][~done], feats['vitals'][~done] / VITALS_SCALE], axis=-1)
                    intrinsic[~done] = ppo.rnd.bonus(legal) * a.rnd_coef * (1.0 - progress)
                    rnd_inputs.append(legal)
                reward_components.append([float(feats['_extrinsic'].mean()), float(feats['_helper'].mean()), float(intrinsic.mean())])
                r = r + intrinsic
                buf.a[t], buf.logp[t], buf.v[t] = choices, logp, v
                buf.r[t] = ppo.rscale.scale(r, done)
                buf.done[t] = done
                starts = done.copy()
                recent.extend(i for i in infos if i is not None)
            last_v = ppo.bootstrap(feats, state, starts)
            last_v = last_v * (1.0 - starts)  # a fresh episode's value does not bootstrap the old one
            t_roll = time.time() - t0
            t0 = time.time()
            stats = ppo.update(buf, h0, last_v, lr=a.lr * (1.0 - 0.9 * progress),
                               ent_coef=a.ent * (1.0 - 0.9 * progress))
            stats['rnd_loss'] = ppo.rnd.update(np.concatenate(rnd_inputs)) if rnd_inputs else 0.
            stats.update(zip(('reward_extrinsic', 'reward_helper', 'reward_intrinsic'), np.mean(reward_components, axis=0).tolist()))
            stats['exit_labels_known'], stats['exit_labels_positive'] = label_counts
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
            extra = {"steps": steps, "stage": a.stage, "memory": a.memory, "name": name,
                     "manager": a.manager, "rgb": a.rgb, "depth": a.depth, "curriculum": ppo.curriculum_config, "training": ppo.training_config, "warm_start": transferred}
            save_checkpoint(out / "latest.pt", net, extra)
            save_train_state(resume_state, ppo, steps, it, next_save)
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
