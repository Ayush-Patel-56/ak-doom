"""Regression coverage for independent Phase 5 review findings; no test-map engine runs."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import pytest
from training.map_pool import load_catalog_pool
from scripts.test_use import reserve_test_use

REPO=Path(__file__).resolve().parents[3]


def test_copied_and_linked_test_wads_are_guarded_and_keep_metadata(tmp_path):
    from scripts.test_use import catalog_metadata
    row=next(r for r in load_catalog_pool('test',[1,2,3,4]) if r['tier']==3)
    original=Path(row['resolved_wad'])
    copy=tmp_path/'copy.wad';shutil.copyfile(original,copy)
    hard=tmp_path/'hard.wad'
    try:
        os.link(original,hard)
    except OSError:  # hard links can't cross drives/filesystems (e.g. Windows F: -> C:); identity is by content
        os.link(copy,hard)
    paths=[copy,hard]
    sym=tmp_path/'sym.wad'
    try:
        sym.symlink_to(original);paths.append(sym)
    except OSError:  # Windows needs Developer Mode or admin rights for symlinks
        pass
    for path in paths:
        spec=f"{path}:{row['map_name'].lower()}"
        with pytest.raises(ValueError,match='final-test'):
            reserve_test_use([spec],REPO/'submission/weights/champion.pt',ledger=tmp_path/'ledger')
        assert catalog_metadata([spec])[spec]['tier']==3
    assert reserve_test_use([f'{copy}:MAP99'],REPO/'submission/weights/champion.pt',ledger=tmp_path/'ledger') is None


def test_legacy_ablation_cannot_play_test_without_opt_in(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    from scripts import ablation
    import evaluate
    def unexpected(*args,**kwargs):pytest.fail('held-out engine play must not occur')
    monkeypatch.setattr(evaluate,'play',unexpected)
    with pytest.raises(ValueError,match='final-test'):
        ablation.evaluate_arm(REPO/'submission/weights/champion.pt','test',1)


def test_incomplete_generalisation_reports_rejected():
    from scripts.generalisation import validate_report
    rows=[dict(map=f'm{tier}',tier=tier,seed=0,score=100.,exit=True,stopped=None) for tier in (1,2,3,4)]
    report=dict(split='val',rows=rows,errors=[],seeds=1,expected_episodes=4,
                map_pool=[dict(id=r['map'],tier=r['tier']) for r in rows])
    validate_report(report,'val')
    for bad in (dict(report,errors=['engine crash']),dict(report,rows=rows[:-1]),
                dict(report,map_pool=report['map_pool'][:-1],rows=rows[:-1],expected_episodes=3)):
        with pytest.raises(ValueError):validate_report(bad,'val')


def test_benchmark_direct_cli_with_clean_pythonpath(tmp_path):
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    p=subprocess.run([sys.executable,str(REPO/'scripts/benchmark_latency.py'),
                      '--weights',str(tmp_path/'missing.pt'),'--ticks','4'],env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'FileNotFoundError' in p.stderr
    assert 'ModuleNotFoundError' not in p.stderr


def test_strict_evidence_rejects_untrained_checkpoint(tmp_path):
    from smriti.brain import BrainConfig,SmritiNet
    from training.checkpoints import save_checkpoint
    from scripts.phase4_ablation import strict_agent
    path=save_checkpoint(tmp_path/'untrained.pt',SmritiNet(BrainConfig()))
    with pytest.raises(ValueError,match='trained'):strict_agent(path)


def test_stage5_changed_wad_rejects_resume_and_episodes_are_persisted(tmp_path):
    import uuid
    from training.train import main
    path=tmp_path/'full.wad';shutil.copyfile(REPO/'maps/two_rooms.wad',path)
    catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps({'maps':[dict(id=f'{path}:MAP01',tier=4,split='train',source='custom',usable=True,episode_ticks=2100)]}))
    name='s5-provenance-'+uuid.uuid4().hex[:8];out=REPO/'runs'/name
    args=['--stage','5','--memory','off','--catalog',str(catalog),'--full-level-share','1',
          '--workers','1','--games','1','--T','4','--epochs','1','--minibatches','1','--threads','1',
          '--ticks','8','--name',name]
    try:
        assert main(args+['--steps','4'])==0
        log=out/'episodes.jsonl'
        assert log.exists()
        rows=[json.loads(s) for s in log.read_text().splitlines()]
        assert len(rows)==2 and all(r['episode_ticks']==8 and r['wad']==str(path) for r in rows)
        assert len({r['engine_seed'] for r in rows})==2
        assert all(r['map_metadata']['split']=='train' for r in rows)
        with path.open('ab') as f:f.write(b'changed-content')
        with pytest.raises(ValueError,match='curriculum'):
            main(args+['--steps','8','--resume'])
    finally:shutil.rmtree(out,ignore_errors=True)


def test_chaos_fallbacks_are_visible(spaces):
    from scripts.stress_agent import measure_chaos
    from scripts.agent_loader import load_agent_class
    agent=load_agent_class()(*spaces,'cpu')
    def broken(obs): raise RuntimeError('deliberate test policy failure')
    agent.policy=broken
    # reset implementation invokes policy.reset, so provide its reset contract.
    class Broken:
        reset=staticmethod(lambda:None)
        __call__=staticmethod(broken)
    agent.policy=Broken()
    rows=measure_chaos(agent)
    assert rows and all(r['valid_action'] and r['fallbacks']==1 for r in rows)


def test_strict_loader_times_only_constructor_and_forwards_requested_device(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import phase4_ablation as abl
    import agent_loader
    from smriti.brain import BrainConfig,SmritiNet
    from training.checkpoints import save_checkpoint
    net=SmritiNet(BrainConfig());path=save_checkpoint(tmp_path/'fixture.pt',net,{'steps':1})
    requested=[]
    def ctor(obs,actions,device,weights_path):
        requested.append(device)
        return SimpleNamespace(policy_name='brain:fixture',policy=SimpleNamespace(net=net),device=device)
    monkeypatch.setattr(agent_loader,'load_agent_class',lambda:ctor)
    clock=iter([10.,12.]);monkeypatch.setattr(abl,'time',SimpleNamespace(perf_counter=lambda:next(clock)))
    agent,_,_=abl.strict_agent(path,device='cuda')
    assert requested==['cuda'] and agent.scored_init_s==2.


def test_missing_tiers_cannot_qualify_even_in_direct_comparison():
    from scripts.generalisation import compare
    a=[dict(map='val3',tier=3,seed=0,score=100.,exit=True,stopped=None)]
    b=[dict(a[0],map='test3')]
    assert not compare(a,b,.1,1.,min_episodes_per_tier=1)['generalisation_demonstrated']
