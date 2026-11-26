"""Controlled Stage 4 experiments; short runs verify the pipeline, not effectiveness.

run trains four matched arms; eval accepts canonical name=checkpoint arms without training.
All arms share Stage 4 exposure and disable goal helpers. phase3 is a broader architecture reference.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts'), str(REPO / 'submission')]


def arm_configs(rgb=True, depth=False):
    base = dict(memory='on', manager='on', rnd='on', rgb='on' if rgb else 'off',
                depth='on' if depth else 'off', **{'goal-helpers':'off'})
    return {'full':base, 'no-manager':base | {'manager':'off'},
            'no-curiosity':base | {'rnd':'off'},
            'phase3':base | dict(manager='off', rnd='off', rgb='off', depth='off')}


def extra_args(value):
    args = shlex.split(value)
    # Allowlist avoids argparse abbreviation, aliases, and future controlled switches.
    allowed = {'--T','--epochs','--minibatches','--lr','--ent','--device','--threads',
               '--save-every','--switch-every','--goal-interval','--rnd-coef','--exit-aux'}
    for token in args:
        if token.startswith('--') and token.split('=',1)[0] not in allowed:
            raise ValueError(f'controlled or unsupported training argument: {token}')
    return args


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_arms(values):
    arms = {}
    for value in values:
        name, separator, path = value.partition('=')
        if not separator or name not in arm_configs():
            raise ValueError('arm must be canonical-name=checkpoint: full, no-manager, no-curiosity, phase3')
        if name in arms:
            raise ValueError(f'duplicate arm: {name}')
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        arms[name] = p
    return arms


def strict_agent(weights, expected=None, device="cpu"):
    import agent_loader
    agent_loader.apply_eval_thread_limits()
    from smriti.brain import load_checkpoint
    from doom_env import observation_space, action_space
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    net, extra = load_checkpoint(weights, 'cpu')
    if type(extra.get("steps")) is not int or extra["steps"] <= 0:
        raise ValueError("evidence requires a trained checkpoint with positive decision provenance")
    cfg = net.cfg
    if expected:
        flags = {'memory':cfg.use_mindmap and cfg.use_notebook, 'manager':cfg.use_manager,
                 'rgb':cfg.use_rgb, 'depth':getattr(cfg,'use_depth',False)}
        for key, actual in flags.items():
            if actual != (expected[key] == 'on'):
                raise ValueError(f'{weights}: {key} does not match requested arm')
        training = extra.get('training', {})
        if training.get('rnd') != expected['rnd'] or training.get('helpers') != 'off':
            raise ValueError(f'{weights}: RND/helper metadata does not match requested arm')
    constructor = agent_loader.load_agent_class()
    t0 = time.perf_counter()
    agent = constructor(observation_space(), action_space(), device, weights_path=weights)
    agent.scored_init_s = time.perf_counter() - t0
    if not agent.policy_name.startswith('brain:') or not hasattr(agent.policy, 'net'):
        raise ValueError(f'{weights}: trained policy failed to load (random fallback rejected)')
    if asdict(agent.policy.net.cfg) != asdict(cfg):
        raise ValueError('loaded policy configuration changed')
    return agent, asdict(cfg), extra


def validate_budget(steps, workers, games, rollout_length):
    quantum = workers * games * rollout_length
    if min(steps, workers, games, rollout_length) < 1 or steps % quantum:
        raise ValueError(f'steps must be positive and divisible by T*workers*games ({quantum})')


def validate_checkpoint_budget(extra, requested):
    actual = extra.get('steps')
    if type(actual) is not int or actual != requested:
        raise ValueError(f'checkpoint decisions {actual!r} differ from requested budget {requested}')


def episode_errors(arm, rows):
    return [dict(arm=arm, stage='evaluation', map=row['map'], seed=row['seed'],
                 error=f"stopped episode: {row['stopped']}")
            for row in rows if row.get('stopped')]


def summarize(results):
    summaries = {}
    identities = {}
    for name, result in results.items():
        keys = [(r['map'],r['seed']) for r in result['rows']]
        if len(set(keys)) != len(keys):
            raise ValueError(f'duplicate map/seed rows in {name}')
        identities[name] = {(r['map'],r['seed'],r['tier']) for r in result['rows']}
    if 'full' in identities:
        for name, keys in identities.items():
            if keys != identities['full']:
                raise ValueError(f'unpaired episode identities for full and {name}')
    for name, result in results.items():
        rows = result['rows']
        tiers = {}
        for tier in (3,4):
            group = [r for r in rows if r['tier'] == tier]
            n, wins = len(group), sum(bool(r['exit']) for r in group)
            tiers[str(tier)] = dict(episodes=n, exits=wins, exit_rate=wins/n if n else None)
        summaries[name] = dict(episodes=len(rows), tiers=tiers,
                               stopped=[r for r in rows if r.get('stopped')])
    paired = {}
    if 'full' in results:
        full = {(r['map'],r['seed']):r for r in results['full']['rows']}
        for name, result in results.items():
            if name == 'full':
                continue
            other = {(r['map'],r['seed']):r for r in result['rows']}
            keys = sorted(full.keys() & other.keys())
            entry = {'episodes':len(keys), 'exit_rate_delta':
                     sum(int(full[k]['exit'])-int(other[k]['exit']) for k in keys)/len(keys) if keys else None}
            entry['tiers'] = {}
            for tier in (3,4):
                subset = [k for k in keys if full[k]['tier'] == tier]
                entry['tiers'][str(tier)] = {'episodes':len(subset), 'exit_rate_delta':
                    sum(int(full[k]['exit'])-int(other[k]['exit']) for k in subset)/len(subset) if subset else None}
            paired[f'full-minus-{name}'] = entry
    return dict(arms=summaries, paired=paired)


def evaluation_pool(split, limit=None):
    from training.map_pool import load_catalog_pool
    pool = load_catalog_pool(split, [3,4])
    if not pool:
        raise ValueError('empty evaluation pool')
    if {entry['tier'] for entry in pool} != {3,4}:
        raise ValueError('evaluation requires usable Tier 3 and Tier 4 maps')
    pool = [dict(entry, spec=f"{entry['resolved_wad']}:{entry['map_name']}") for entry in pool]
    if limit is not None:
        if limit < 2:
            raise ValueError('limit-maps must be at least 2 to include both tiers')
        chosen = []
        if limit >= 2:
            for tier in (3,4):
                match = next((entry for entry in pool if entry['tier'] == tier), None)
                if match:
                    chosen.append(match)
        pool = (chosen + [entry for entry in pool if entry not in chosen])[:limit]
    return pool


def evaluate_arm(weights, pool, seeds, expected, eval_ticks=None, final_test=False, allow_repeat_test=False):
    import evaluate as league
    agent, cfg, extra = strict_agent(weights, expected)
    rows = []
    from scripts.test_use import reserve_test_use, finish_test_use
    token = reserve_test_use([entry['spec'] for entry in pool], weights,
                             final_test=final_test, allow_repeat=allow_repeat_test)
    try:
        for entry in pool:
            meta = dict(entry)
            if eval_ticks is not None:
                meta['episode_ticks'] = eval_ticks
            for seed in range(seeds):
                row = league.play(agent, entry['spec'], meta, seed,
                                  render_buffers=cfg['use_rgb'] or cfg.get('use_depth',False))
                row.update(map=entry['spec'], seed=seed, tier=entry['tier'])
                rows.append(row)
    finally:
        finish_test_use(token, status='completed' if len(rows)==len(pool)*seeds and not any(r.get('stopped') for r in rows) else 'partial')
    return dict(checkpoint=str(weights), sha256=digest(weights), config=cfg, training=extra, rows=rows)


def write_report(name, manifest, results, errors):
    folder = REPO / 'runs' / 'ablation'
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(manifest=manifest, results=results, summary=summarize(results), errors=errors,
                  interpretation='Pipeline verification only when smoke overrides or short training are used. '
                  'Exit-rate differences do not establish robust improvement; effectiveness remains unproven.')
    if 'paired_by_train_seed' in manifest:
        report['summary']['paired_by_train_seed'] = manifest['paired_by_train_seed']
    path = folder / f'{name}.json'
    path.write_text(json.dumps(report, indent=2, default=str))
    lines = [f'# Phase 4 comparison {name}', '', report['interpretation'], '',
             '| arm | tier | episodes | exits | exit rate |', '|---|---|---|---|---|']
    for arm, summary in report['summary']['arms'].items():
        for tier, stats in summary['tiers'].items():
            lines.append(f"| {arm} | {tier} | {stats['episodes']} | {stats['exits']} | {stats['exit_rate']} |")
    if 'paired_by_train_seed' in report['summary']:
        lines += ['', 'Paired comparisons by training seed: `'+json.dumps(report['summary']['paired_by_train_seed'])+'`']
    lines += ['', 'Paired comparisons: `'+json.dumps(report['summary']['paired'])+'`', '',
              'Errors: `'+json.dumps(errors)+'`', '', f'Full manifest and raw episodes: {path.name}']
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return path


def parser():
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subs = ap.add_subparsers(dest='cmd', required=True)
    for mode in ('run','eval'):
        p = subs.add_parser(mode, allow_abbrev=False)
        p.add_argument('--name', default=time.strftime('phase4-%Y%m%d-%H%M%S'))
        p.add_argument('--split', choices=['val','test'], default='val')
        p.add_argument('--final-test', action='store_true')
        p.add_argument('--allow-repeat-test', action='store_true')
        p.add_argument('--seeds', type=int, default=3)
        p.add_argument('--rgb', choices=['on','off'], default='on')
        p.add_argument('--depth', choices=['on','off'], default='off')
        p.add_argument('--limit-maps', type=int)
        p.add_argument('--eval-ticks', type=int)
        if mode == 'eval':
            p.add_argument('--arm', action='append', required=True)
        else:
            p.add_argument('--steps', type=int, default=1_999_872)
            p.add_argument('--workers', type=int, default=3)
            p.add_argument('--games', type=int, default=4)
            p.add_argument('--seed', type=int, default=0)
            p.add_argument('--train-seed', type=int, action='append')
            p.add_argument('--ticks', type=int)
            p.add_argument('--init', '--source', type=Path, help='shared Phase 3 or earlier warm-start source')
            p.add_argument('--full-level-share', type=float, default=.8)
            p.add_argument('--train-args', default='')
    return ap


def main(argv=None):
    ap = parser()
    a = ap.parse_args(argv)
    if a.split == 'test' and not a.final_test: ap.error('test split requires --final-test')
    for field in ('seeds','eval_ticks','limit_maps','steps','workers','games','ticks'):
        value = getattr(a,field,None)
        if value is not None and value < 1:
            ap.error(field+' must be positive')
    if Path(a.name).name != a.name or a.name in ('.','..'):
        ap.error('name must be a simple filename')
    configs = arm_configs(a.rgb == 'on', a.depth == 'on')
    errors, results = [], {}
    pool = evaluation_pool(a.split,a.limit_maps)
    manifest = dict(mode=a.cmd, split=a.split, evaluation_pool=pool, seeds_per_map=a.seeds,
                    configs=configs, goal_helpers='off', smoke_overrides=dict(eval_ticks=a.eval_ticks,
                    limit_maps=a.limit_maps, train_ticks=getattr(a,'ticks',None)), arguments=vars(a))
    if a.cmd == 'eval':
        arms = parse_arms(a.arm)
    else:
        if not 0 <= a.full_level_share <= 1:
            ap.error('full-level-share must be in [0,1]')
        extra = extra_args(a.train_args)
        from training.train import build_args
        resolved_common = build_args(['--stage','4','--steps',str(a.steps),
                                      '--workers',str(a.workers),'--games',str(a.games), *extra])
        validate_budget(a.steps,a.workers,a.games,resolved_common.T)
        train_seeds = a.train_seed or [a.seed]
        if len(set(train_seeds)) != len(train_seeds):
            ap.error('duplicate training seed')
        from training.map_pool import load_catalog_pool
        manifest['training_pool'] = load_catalog_pool('train',[4],source='freedoom2')
        if not manifest['training_pool']:
            raise ValueError('empty Stage4 training pool')
        manifest['initializations'] = {}
        arms = {}
        from smriti.brain import BrainConfig, SmritiNet, load_checkpoint
        from training.checkpoints import save_checkpoint
        import torch
        for seed in train_seeds:
            source = a.init
            if source:
                net, _ = load_checkpoint(source)
                if any(getattr(net.cfg,k,False) for k in ('use_manager','use_rgb','use_depth')):
                    raise ValueError('shared initialization must be Phase 3 or earlier')
            else:
                source = REPO / 'runs' / 'ablation' / f'{a.name}-source-{seed}.pt'
                torch.manual_seed(seed)
                save_checkpoint(source, SmritiNet(BrainConfig(use_mindmap=True,use_notebook=True)))
            manifest['initializations'][str(seed)] = dict(path=str(source), sha256=digest(source))
            for arm, cfg in configs.items():
                label = f'{a.name}-{arm}-s{seed}'
                cmd = [sys.executable,'-m','training.train','--stage','4','--name',label,
                       '--steps',str(a.steps),'--seed',str(seed),'--workers',str(a.workers),
                       '--games',str(a.games),'--full-level-share',str(a.full_level_share),
                       '--warm-start',str(source)]
                for key,value in cfg.items():
                    cmd += ['--'+key,value]
                if a.ticks is not None:
                    cmd += ['--ticks',str(a.ticks)]
                cmd += extra
                from training.train import build_args
                from training.ppo import PPOConfig
                resolved = build_args(cmd[3:])
                manifest.setdefault('resolved_training',{})[label] = vars(resolved)
                manifest.setdefault('resolved_ppo',{})[label] = asdict(PPOConfig(
                    T=resolved.T, lr=resolved.lr, ent_coef=resolved.ent, epochs=resolved.epochs,
                    minibatches=resolved.minibatches, exit_aux_coef=resolved.exit_aux))
                manifest.setdefault('commands',{})[label] = cmd
                logpath = REPO / 'runs' / f'{label}.log'
                logpath.parent.mkdir(parents=True,exist_ok=True)
                with logpath.open('w') as log:
                    proc = subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                if proc.returncode:
                    errors.append(dict(arm=arm,train_seed=seed,stage='training',returncode=proc.returncode,log=str(logpath)))
                else:
                    arms[(arm,seed)] = REPO / 'runs' / label / 'latest.pt'
    for key,weights in arms.items():
        arm, seed = key if isinstance(key,tuple) else (key,None)
        label = f'{arm}-s{seed}' if seed is not None else arm
        try:
            if seed is not None:
                from smriti.brain import load_checkpoint
                _, checkpoint_extra = load_checkpoint(weights, 'cpu')
                manifest.setdefault('checkpoint_budgets', {})[label] = dict(
                    checkpoint=str(weights), sha256=digest(weights),
                    requested_decisions=a.steps, actual_decisions=checkpoint_extra.get('steps'))
                validate_checkpoint_budget(checkpoint_extra, a.steps)
            result = evaluate_arm(weights,pool,a.seeds,configs[arm],a.eval_ticks,
                                  final_test=a.final_test, allow_repeat_test=a.allow_repeat_test)
            results[label] = result
            errors.extend(episode_errors(label,result['rows']))
            if seed is not None:
                result['train_seed'] = seed
                result['requested_decisions'] = a.steps
                result['actual_decisions'] = result['training'].get('steps')
        except Exception as exc:
            errors.append(dict(arm=label,stage='evaluation',error=f'{type(exc).__name__}: {exc}'))
    # Compare each matched training seed separately; never pool unmatched training runs.
    if a.cmd == 'run':
        manifest['paired_by_train_seed'] = {str(seed):summarize({arm:results[f'{arm}-s{seed}']
            for arm in configs if f'{arm}-s{seed}' in results}) for seed in train_seeds}
    path = write_report(a.name,manifest,results,errors)
    print(path)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
