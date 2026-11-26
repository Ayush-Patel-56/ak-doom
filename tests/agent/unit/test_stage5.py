import json
import numpy as np
import pytest
from training.mapgen import playable, write_map


def test_stage5_pool_seed_range_caps_and_identity(tmp_path, monkeypatch):
    from training.curriculum import MapSampler
    import training.curriculum as c
    monkeypatch.setattr(c, 'GENERATED', tmp_path)
    a, b = MapSampler(5,19), MapSampler(5,19)
    assert a.pool_manifest == b.pool_manifest
    assert a.pool_manifest['generated_seed_range'] == [0,100000]
    assert a.pool_manifest['full_level_share'] == .5
    assert all(r['id'] for r in a.pool_manifest['full_pool'])
    kinds = [a.pick_kind() for _ in range(500)]
    assert 200 < kinds.count('full') < 300
    assert set(kinds) == {'full','arena','maze','rooms','keydoor','keynear'}
    gen = MapSampler(5,19,kinds=['arena'])
    wad, name, kind = next(gen)
    assert 's5-' in wad and 2100 <= gen.episode_ticks(wad,name) <= 4200
    assert gen.map_metadata(wad,name)['randomisation']['seed'] < 100000
    assert gen.pool_manifest['randomisation']['profile_hash'] in wad


def test_stage5_train_cli_and_legacy_defaults():
    from training.train import build_args
    from training.curriculum import MapSampler
    assert build_args(['--stage','5']).stage == 5
    assert MapSampler(4,0).pool_manifest['full_level_share'] == .8
    assert MapSampler(3,0).pool_manifest['seeds_per_kind'] == 200

