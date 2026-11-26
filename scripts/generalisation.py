"""Strict validation/final-test evaluation and matched tier score comparison.

Use `run --split val` in development. Test execution requires --final-test and a use ledger.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts'),str(REPO/'submission')]
from scripts.test_use import digest, reserve_test_use, finish_test_use


def compare(validation, test, gap_tolerance, min_mean_score, min_episodes_per_tier=30):
    if not math.isfinite(gap_tolerance) or gap_tolerance<0 or not math.isfinite(min_mean_score) or min_mean_score<=0:
        raise ValueError('declare finite nonnegative gap and positive useful-score threshold')
    if min_episodes_per_tier<1: raise ValueError('minimum episodes must be positive')
    for rows in (validation,test):
        if not rows: raise ValueError('empty results')
        keys=[(r['map'],r['seed']) for r in rows]
        if len(set(keys))!=len(keys): raise ValueError('duplicate map/seed results')
        if any(r.get('stopped') or not math.isfinite(r['score']) or r.get('steps',1)<1 for r in rows):
            raise ValueError('failed/stopped/nonfinite results')
    # Input reports preserve map specs; reject accidental train/val self-comparison.
    if {r['map'] for r in validation} & {r['map'] for r in test}: raise ValueError('overlapping map sets')
    counts=[Counter((r['tier'],r['seed']) for r in rows) for rows in (validation,test)]
    if counts[0]!=counts[1]: raise ValueError('unmatched tier/seed composition')
    from training.scorer import TIER_WEIGHT
    def summary(rows):
        scores=[float(r['score'])*TIER_WEIGHT[r['tier']] for r in rows]
        mean=sum(scores)/len(scores)
        return dict(episodes=len(rows),mean_weighted_score=mean,exit_rate=sum(bool(r['exit']) for r in rows)/len(rows),
                    tiers={str(t):dict(episodes=sum(r['tier']==t for r in rows),
                        mean_score=sum(r['score'] for r in rows if r['tier']==t)/sum(r['tier']==t for r in rows))
                        for t in sorted({r['tier'] for r in rows})})
    v,t=summary(validation),summary(test)
    gap=abs(t['mean_weighted_score']-v['mean_weighted_score'])/max(abs(v['mean_weighted_score']),1.)
    enough=all(g['episodes']>=min_episodes_per_tier for g in v['tiers'].values())
    useful=min(v['mean_weighted_score'],t['mean_weighted_score'])>=min_mean_score
    return dict(validation=v,test=t,relative_score_gap=gap,gap_tolerance=gap_tolerance,
                minimum_mean_score=min_mean_score,minimum_episodes_per_tier=min_episodes_per_tier,
                gap_within_tolerance=gap<=gap_tolerance,
                generalisation_demonstrated=gap<=gap_tolerance and enough and useful,
                interpretation='Threshold-based evidence only; inspect raw episodes and seed variation. Historical test freshness unknown.')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='mode',required=True)
    run=sub.add_parser('run');run.add_argument('--weights',type=Path,required=True)
    run.add_argument('--split',choices=['val','test'],default='val');run.add_argument('--seeds',type=int,default=3)
    run.add_argument('--maps-per-tier',type=int);run.add_argument('--ticks',type=int)
    run.add_argument('--final-test',action='store_true');run.add_argument('--allow-repeat-test',action='store_true')
    cmp=sub.add_parser('compare');cmp.add_argument('--validation',type=Path,required=True);cmp.add_argument('--test',type=Path,required=True)
    cmp.add_argument('--gap-tolerance',type=float,required=True);cmp.add_argument('--min-mean-score',type=float,required=True)
    cmp.add_argument('--min-episodes-per-tier',type=int,default=30)
    for p in (run,cmp): p.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv)
    if a.mode=='compare':
        v,t=(json.loads(p.read_text()) for p in (a.validation,a.test))
        if not v.get('checkpoint_sha256') or v['checkpoint_sha256']!=t.get('checkpoint_sha256') or v.get('config')!=t.get('config'):
            raise ValueError('checkpoint/config drift')
        if v.get('split')!='val' or t.get('split')!='test' or v.get('ticks_override')!=t.get('ticks_override'):
            raise ValueError('require validation and test reports with matching tick overrides')
        report=compare(v['rows'],t['rows'],a.gap_tolerance,a.min_mean_score,a.min_episodes_per_tier)
        report.update(checkpoint_sha256=v['checkpoint_sha256'],raw_validation=v,raw_test=t)
    else:
        if a.seeds<1 or (a.maps_per_tier is not None and a.maps_per_tier<1) or (a.ticks is not None and a.ticks<1):
            ap.error('counts/ticks must be positive')
        from training.map_pool import load_catalog_pool
        from scripts.phase4_ablation import strict_agent
        import evaluate
        pools={s:load_catalog_pool(s,[1,2,3,4]) for s in ('val','test')}
        selected=[]
        for tier in (1,2,3,4):
            groups={s:[r for r in rows if r['tier']==tier] for s,rows in pools.items()}
            n=min(len(rows) for rows in groups.values())
            if a.maps_per_tier is not None: n=min(n,a.maps_per_tier)
            if not n: raise ValueError(f'empty matched tier {tier}')
            selected.extend(groups[a.split][:n])
        agent,cfg,_=strict_agent(a.weights)
        specs=[r['id'] for r in selected]
        token=reserve_test_use(specs,a.weights,final_test=a.final_test,allow_repeat=a.allow_repeat_test)
        report=dict(split=a.split,checkpoint_sha256=digest(a.weights),config=cfg,seeds=a.seeds,
                    ticks_override=a.ticks,test_use_id=token,rows=[],errors=[])
        try:
            for r in selected:
                for seed in range(a.seeds):
                    meta=r if a.ticks is None else dict(r,episode_ticks=a.ticks)
                    row=evaluate.play(agent,r['id'],meta,seed,cfg['use_rgb'] or cfg.get('use_depth',False))
                    report['rows'].append(row)
                    if row.get('stopped'): report['errors'].append(row['stopped'])
            finish_test_use(token,status='failed' if report['errors'] else 'completed')
        except Exception as exc:
            report['errors'].append(f'{type(exc).__name__}: {exc}');finish_test_use(token,status='partial')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(a.output)
    return 1 if report.get('errors') else 0

if __name__=='__main__': raise SystemExit(main())
