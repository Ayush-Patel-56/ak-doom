"""Protocol fixtures are arithmetic/configuration checks, never measured outcomes."""
from pathlib import Path
import pytest
from scripts import phase4_ablation as abl


def test_matched_treatments_and_phase3_reference():
    arms = abl.arm_configs(rgb=True, depth=True)
    assert set(arms) == {'full', 'no-manager', 'no-curiosity', 'phase3'}
    for cfg in arms.values():
        assert cfg['memory'] == 'on' and cfg['goal-helpers'] == 'off'
    assert arms['full'] | {'manager': 'off'} == arms['no-manager']
    assert arms['full'] | {'rnd': 'off'} == arms['no-curiosity']
    assert all(arms['phase3'][key] == 'off' for key in ('manager','rnd','rgb','depth'))


@pytest.mark.parametrize('extra', ['--manager=off', '--steps 2', '--seed=3', '--stage 3', '--warm-start x', '--resume', '--depth off', '--helper-weights {}', '--kinds hall'])
def test_controlled_overrides_rejected(extra):
    with pytest.raises(ValueError):
        abl.extra_args(extra)


def test_extra_args_shell_quoting():
    assert abl.extra_args('--device "cpu" --lr 0.001') == ['--device','cpu','--lr','0.001']


def test_duplicate_and_missing_checkpoints(tmp_path):
    p = tmp_path / 'x.pt'
    p.touch()
    with pytest.raises(ValueError, match='duplicate'):
        abl.parse_arms([f'full={p}', f'full={p}'])
    with pytest.raises(FileNotFoundError):
        abl.parse_arms([f'full={tmp_path / "missing"}'])


def test_paired_math_and_counts():
    a = [{'map':'a','seed':0,'tier':3,'exit':True}, {'map':'b','seed':0,'tier':4,'exit':False}]
    b = [dict(a[0], exit=False), dict(a[1], exit=True)]
    report = abl.summarize({'full': {'rows': a}, 'phase3': {'rows': b}})
    assert report['arms']['full']['tiers']['3'] == {'episodes':1,'exits':1,'exit_rate':1.0}
    assert report['paired']['full-minus-phase3']['episodes'] == 2
    assert report['paired']['full-minus-phase3']['exit_rate_delta'] == 0


def test_checkpoint_fails_closed(tmp_path):
    p = tmp_path / 'bad.pt'
    p.write_text('not a checkpoint')
    with pytest.raises(Exception):
        abl.strict_agent(p)


@pytest.mark.parametrize('rgb,depth', [(True,False),(False,True)])
def test_real_engine_evaluation_renders_rgb_and_rejects_wrong_arm(tmp_path, monkeypatch,rgb,depth):
    import sys
    sys.path[:0] = [str(abl.REPO / 'submission'), str(abl.REPO / 'scripts')]
    from smriti.brain import BrainConfig, SmritiNet
    from training.checkpoints import save_checkpoint
    import evaluate as league
    p = tmp_path / 'rgb.pt'
    save_checkpoint(p, SmritiNet(BrainConfig(use_mindmap=True, use_notebook=True,
                                           use_manager=True, use_rgb=rgb, use_depth=depth)),
                    {'training':{'rnd':'on','helpers':'off'}})
    pool = abl.evaluation_pool('val', 2)
    observed = []
    real_play = league.play
    def play(agent, spec, meta, seed, render_buffers):
        observed.append(render_buffers)
        # Exercise real ViZDoom, checkpoint inference, and report rows.
        return real_play(agent, spec, meta, seed, render_buffers)
    monkeypatch.setattr(league, 'play', play)
    result = abl.evaluate_arm(p,pool,1,abl.arm_configs(rgb,depth)['full'],eval_ticks=7)
    assert len(result['rows']) == 2
    assert observed == [True,True]
    assert {r['tier'] for r in result['rows']} == {3,4}
    with pytest.raises(ValueError, match='manager'):
        abl.evaluate_arm(p,pool,1,abl.arm_configs(rgb,depth)['no-manager'],eval_ticks=7)


def test_budget_must_match_complete_rollouts():
    with pytest.raises(ValueError, match='divisible'):
        abl.validate_budget(9, 2, 1, 4)
    abl.validate_budget(16, 2, 1, 4)


def test_evaluation_pool_requires_both_tiers():
    with pytest.raises(ValueError, match='at least 2'):
        abl.evaluation_pool('val', 1)


def test_missing_evaluation_tier_rejected(monkeypatch):
    from training import map_pool
    monkeypatch.setattr(map_pool, 'load_catalog_pool', lambda *args: [
        dict(resolved_wad='/tmp/a.wad', map_name='MAP01', tier=3)])
    with pytest.raises(ValueError, match='Tier 3 and Tier 4'):
        abl.evaluation_pool('val')


def test_duplicate_and_unpaired_rows_rejected():
    row = dict(map='a', seed=0, tier=3, exit=False)
    with pytest.raises(ValueError, match='duplicate'):
        abl.summarize({'full': {'rows':[row,row]}})
    with pytest.raises(ValueError, match='unpaired'):
        abl.summarize({'full': {'rows':[row]}, 'phase3': {'rows':[]}})


def test_stopped_episode_is_failure():
    assert abl.episode_errors('full', [{'map':'a','seed':0,'stopped':'act timeout'}])
    assert not abl.episode_errors('full', [{'map':'a','seed':0,'stopped':None}])


def test_actual_training_budget_checked():
    with pytest.raises(ValueError, match='checkpoint decisions'):
        abl.validate_checkpoint_budget({'steps':12}, 8)
    with pytest.raises(ValueError, match='checkpoint decisions'):
        abl.validate_checkpoint_budget({}, 8)
    abl.validate_checkpoint_budget({'steps':8}, 8)


def test_invalid_rollout_budget_rejected_before_trainer_spawn(monkeypatch):
    def unexpected_spawn(*args, **kwargs):
        pytest.fail('invalid budget spawned a trainer')
    monkeypatch.setattr(abl.subprocess, 'run', unexpected_spawn)
    with pytest.raises(ValueError, match='divisible'):
        abl.main(['run', '--steps','9','--workers','1','--games','1',
                  '--train-args=--T=4', '--limit-maps','2'])


def test_default_run_budget_is_complete_rollouts():
    from training.train import build_args
    args = abl.parser().parse_args(['run'])
    trainer = build_args([])
    abl.validate_budget(args.steps,args.workers,args.games,trainer.T)


def test_wrong_checkpoint_budget_never_evaluated_or_summarized(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from smriti.brain import BrainConfig, SmritiNet
    from training.checkpoints import save_checkpoint
    # Protocol-only fixture: real checkpoint metadata with a deliberately invalid budget.
    net = SmritiNet(BrainConfig(use_mindmap=True,use_notebook=True))
    source = save_checkpoint(tmp_path / 'source.pt', net)
    pool = abl.evaluation_pool('val',2)
    monkeypatch.setattr(abl, 'REPO', tmp_path)
    monkeypatch.setattr(abl, 'evaluation_pool', lambda *args:pool)
    def training_process(cmd, **kwargs):
        label = cmd[cmd.index('--name')+1]
        save_checkpoint(tmp_path / 'runs' / label / 'latest.pt', net, {'steps':12})
        return SimpleNamespace(returncode=0)
    def unexpected_evaluation(*args, **kwargs):
        pytest.fail('budget-rejected checkpoint reached evaluator')
    monkeypatch.setattr(abl.subprocess, 'run', training_process)
    monkeypatch.setattr(abl, 'evaluate_arm', unexpected_evaluation)
    assert abl.main(['run','--name','bad-budget','--steps','8','--workers','1',
                     '--games','1','--train-args=--T=4','--init',str(source)]) == 1
    report = json.loads((tmp_path / 'runs/ablation/bad-budget.json').read_text())
    assert report['results'] == {}
    assert report['summary']['arms'] == {}
    assert report['summary']['paired_by_train_seed']['0']['paired'] == {}
    assert len(report['errors']) == 4
    assert all('checkpoint decisions' in error['error'] for error in report['errors'])
