import json
from pathlib import Path
import pytest


def test_test_use_guard_alias_repeats_and_failed_reservation(tmp_path):
    from scripts.test_use import reserve_test_use, finish_test_use
    from training.map_pool import load_catalog_pool
    entry=load_catalog_pool('test',[1,2,3,4])[0]
    spec=f"{entry['resolved_wad']}:{entry['map_name'].lower()}"
    ckpt=tmp_path/'model.pt'; ckpt.write_bytes(b'identity only')
    ledger=tmp_path/'ledger'
    with pytest.raises(ValueError,match='final-test'):
        reserve_test_use([spec],ckpt,ledger=ledger)
    token=reserve_test_use([spec],ckpt,ledger=ledger,final_test=True)
    assert token and json.loads((ledger/f'{token}.json').read_text())['status']=='reserved'
    with pytest.raises(ValueError,match='already'):
        reserve_test_use([spec],ckpt,ledger=ledger,final_test=True)
    finish_test_use(token,ledger=ledger,status='failed')
    second=reserve_test_use([spec],ckpt,ledger=ledger,final_test=True,allow_repeat=True)
    assert second != token
    assert reserve_test_use(['maps/two_rooms.wad:MAP01'],ckpt,ledger=ledger) is None


def test_generalisation_rejects_unmatched_or_failed_and_reports_low_score():
    from scripts.generalisation import compare
    val=[dict(map='a',tier=3,seed=0,score=0.,exit=False,stopped=None)]
    test=[dict(map='b',tier=3,seed=0,score=0.,exit=False,stopped=None)]
    result=compare(val,test,gap_tolerance=.1,min_mean_score=1.)
    assert result['gap_within_tolerance'] and not result['generalisation_demonstrated']
    with pytest.raises(ValueError): compare([],test,.1,1.)
    with pytest.raises(ValueError): compare(val,[dict(test[0],tier=4)],.1,1.)
    with pytest.raises(ValueError): compare(val,[dict(test[0],stopped='error')],.1,1.)
    with pytest.raises(ValueError): compare(val,val,.1,1.)


def test_existing_evaluation_commands_expose_final_test_guard():
    import inspect
    from scripts import evaluate, phase4_ablation
    assert 'final_test' in inspect.signature(phase4_ablation.evaluate_arm).parameters
    assert phase4_ablation.parser().parse_args(['eval','--arm','full=x','--final-test']).final_test
    assert 'argv' in inspect.signature(evaluate.main).parameters
