"""Real-engine reliability measurements; behavior failures remain failures.

Defaults: 20 generated maps, 4200 ticks, mean <20ms, RSS growth <=64MiB where measurable.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts'),str(REPO/'submission')]
from scripts.phase4_ablation import strict_agent
from scripts.test_use import digest


def valid_action(a):
    a=np.asarray(a)
    return a.shape==(9,) and a.dtype.kind in 'iub' and bool(np.isin(a,(0,1)).all())


def chaos_observations(space):
    from smriti.safety import zeros_like_space
    def mutate(x,value):
        if isinstance(x,dict): return {k:mutate(v,value) for k,v in x.items()}
        return np.full(x.shape,value,dtype=np.float64)
    template=zeros_like_space(space)
    return [None,{}, {'vitals':[]}, mutate(template,float('nan')),
            mutate(template,float('inf')),mutate(template,1e30)]


def measure_chaos(agent):
    rows=[]
    for i,obs in enumerate(chaos_observations(agent.observation_space)):
        agent.reset();agent.reset()
        valid=valid_action(agent.act(obs))
        rows.append(dict(case=i,valid_action=valid,fallbacks=agent.guard.fallbacks,
                         fallback_allowed=True))
    return rows


def scenario_maps(seed):
    from training.mapgen import GridMap, generate
    dense=generate('arena',seed)
    rows=[list(r) for r in dense.rows]
    floor=[(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch=='.']
    for r,c in floor[:20]: rows[r][c]='z'
    dense=GridMap([''.join(r) for r in rows],player_angle=dense.player_angle)
    empty=generate('rooms',seed)
    empty=GridMap([''.join('.' if c in 'zsidrby' else c for c in row) for row in empty.rows],player_angle=180)
    closet=GridMap.parse('''###########
#P........#
#########.#
#X........#
###########''',player_angle=180)
    return dict(dense=dense,empty=empty,closet=closet)


def rss_bytes():
    try:
        line=next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:'))
        return int(line.split()[1])*1024
    except (OSError,StopIteration): return None


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--maps',type=int,default=20)
    ap.add_argument('--ticks',type=int,default=4200);ap.add_argument('--seed',type=int,default=71)
    ap.add_argument('--target-ms',type=float,default=20.);ap.add_argument('--max-rss-growth-mib',type=float,default=64.)
    ap.add_argument('--max-slowdown',type=float,default=3.);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(argv)
    if a.maps<3 or a.ticks<1 or not 0<=a.seed<100000: ap.error('maps>=3, positive ticks and training seed required')
    if any(not math.isfinite(v) or v<=0 for v in (a.target_ms,a.max_rss_growth_mib,a.max_slowdown)):
        ap.error('resource thresholds must be finite and positive')
    agent,cfg,_=strict_agent(a.weights)
    from doom_env import make_env
    from training.mapgen import write_map
    from training.mapgen.randomize import generate_randomized
    failures=[];rows=[]
    report=dict(checkpoint_sha256=digest(a.weights),config=cfg,arguments=vars(a),rows=rows,failures=failures,
                rss_supported=rss_bytes() is not None,official_pinned_cpu=False,chaos=[])
    report['chaos']=measure_chaos(agent)
    for row in report['chaos']:
        if not row['valid_action']: failures.append(f"chaos {row['case']}: invalid action")
    with tempfile.TemporaryDirectory(prefix='phase5-stress-') as tmp:
        cases=list(scenario_maps(a.seed).items())
        for i in range(a.maps-len(cases)):
            kind=('arena','maze','rooms','keydoor','keynear')[i%5]
            cases.append((f'{kind}-{i}',generate_randomized(kind,(a.seed+i)%100000)))
        base_rss=None
        for i,(name,grid) in enumerate(cases):
            env=None
            try:
                path=write_map(grid,Path(tmp)/f'{i}.wad')
                env=make_env(str(path),'MAP01',episode_ticks=a.ticks,render_buffers=cfg['use_rgb'] or cfg.get('use_depth',False))
                obs,_=env.reset(seed=a.seed+i);agent.reset();agent.reset()
                initial=env.privileged_state();times=[];distance=0.;done=False
                while not done:
                    t=time.perf_counter();action=agent.act(obs);times.append((time.perf_counter()-t)*1000)
                    if not valid_action(action): raise ValueError('invalid action')
                    obs,_,term,trunc,_=env.step(action);done=term or trunc
                    state=env.privileged_state()
                    distance=max(distance,math.hypot(state['x']-initial['x'],state['y']-initial['y']))
                current=rss_bytes()
                if base_rss is None: base_rss=current  # first actual episode warms engine/model caches
                mean=float(np.mean(times));n=len(times);window=max(1,n//4)
                slowdown=float(np.mean(times[-window:]))/max(float(np.mean(times[:window])),1e-9)
                row=dict(map=name,seed=a.seed+i,steps=n,mean_ms=mean,p99_ms=float(np.percentile(times,99)),
                         slowdown=slowdown,rss_bytes=current,max_displacement=distance,fallbacks=agent.guard.fallbacks)
                rows.append(row)
                if mean>=a.target_ms: failures.append(f'{name}: latency exceeds target')
                if slowdown>a.max_slowdown: failures.append(f'{name}: latency slowdown')
                if current is not None and base_rss is not None and current-base_rss>a.max_rss_growth_mib*1024**2:
                    failures.append(f'{name}: RSS growth exceeds threshold')
                if agent.guard.fallbacks: failures.append(f'{name}: policy fallback')
                if name=='closet' and distance<128: failures.append('closet: failed to escape (128-unit displacement)')
                if name=='empty' and distance<128: failures.append('empty: failed to explore (128-unit displacement)')
            except Exception as exc:
                failures.append(f'{name}: {type(exc).__name__}: {exc}')
            finally:
                if env is not None: env.close()
    report['reliability_passed']=not failures and len(rows)==a.maps
    report['full_resource_checks_passed']=report['reliability_passed'] and report['rss_supported']
    report['interpretation']='Malformed inputs may use recorded safe fallbacks; normal engine runs require zero fallbacks. Measured behavior and thresholds only; unsupported RSS is not a pass; score stability requires held-out multi-seed evaluation.'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,default=str)+'\n');print(a.output)
    return 1 if failures else 0

if __name__=='__main__': raise SystemExit(main())
