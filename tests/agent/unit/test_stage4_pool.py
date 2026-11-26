"""Catalog isolation and real Stage 4 sampling contracts."""
import json
from pathlib import Path

import pytest

from training.curriculum import MapSampler


def catalog(tmp_path, entries):
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps({'maps': entries}))
    return path


def entry(**updates):
    row = dict(id='maps/two_rooms.wad:MAP01', source='freedoom2', split='train',
               tier=4, usable=True, episode_ticks=4200)
    return row | updates


def test_stage4_seeded_mix_and_manifest():
    a, b = MapSampler(4, 73), MapSampler(4, 73)
    kinds = [a.pick_kind() for _ in range(1000)]
    assert kinds == [b.pick_kind() for _ in range(1000)]
    assert 740 < kinds.count('full') < 860
    assert {'keydoor', 'maze', 'rooms', 'arena'} <= set(kinds)
    assert json.loads(json.dumps(a.pool_manifest))['full_level_share'] == .8


def test_catalog_selection_and_copy():
    from training.map_pool import load_catalog_pool
    pool = load_catalog_pool('train', [4], source='freedoom2')
    assert pool and all(r['split'] == 'train' and r['tier'] == 4 for r in pool)
    assert all(Path(r['resolved_wad']).is_file() for r in pool)
    pool[0]['tier'] = -1
    assert load_catalog_pool('train', [4], source='freedoom2')[0]['tier'] == 4


@pytest.mark.parametrize('updates', [{'split': 'typo'}, {'tier': True}, {'tier': 5},
    {'usable': 'true'}, {'episode_ticks': 4201}, {'episode_ticks': 2099},
    {'episode_ticks': True}, {'id': ''}, {'source': ''}])
def test_invalid_metadata(tmp_path, updates):
    from training.map_pool import load_catalog_pool
    with pytest.raises(ValueError):
        load_catalog_pool('train', [4], catalog_path=catalog(tmp_path, [entry(**updates)]))


def test_alias_leakage_even_outside_selected_tier(tmp_path):
    from training.map_pool import load_catalog_pool
    alias = str(Path('maps/two_rooms.wad').resolve()) + ':map01'
    with pytest.raises(ValueError, match='split'):
        load_catalog_pool('train', [4], catalog_path=catalog(tmp_path,
            [entry(), entry(id=alias, split='val', tier=3)]))


def test_missing_assets_and_empty_pool(tmp_path):
    from training.map_pool import load_catalog_pool
    path = catalog(tmp_path, [entry(id=str(tmp_path / 'missing.wad') + ':MAP01')])
    with pytest.raises(FileNotFoundError):
        load_catalog_pool('train', [4], catalog_path=path)
    assert load_catalog_pool('train', [4], catalog_path=path, require_assets=False)
    with pytest.raises(ValueError, match='empty'):
        MapSampler(4, 1, catalog_path=catalog(tmp_path, []))


def test_full_caps_and_generated_override(tmp_path):
    full = MapSampler(4, 7, kinds=['full'])
    wad, name, kind = next(full)
    assert kind == 'full' and full.episode_ticks(wad, name) == 4200
    replay = MapSampler(4, 7, kinds=['arena'], catalog_path=tmp_path / 'absent.json')
    wad, name, kind = next(replay)
    assert kind == 'arena' and replay.episode_ticks(wad, name) == 2100


def test_sampled_full_level_loads_in_real_engine():
    import vizdoom as vzd
    sampler = MapSampler(4, 19, full_level_share=1)
    wad, name, _ = next(sampler)
    game = vzd.DoomGame()
    try:
        game.set_doom_game_path(wad)
        game.set_doom_map(name)
        game.set_window_visible(False)
        game.set_doom_skill(3)
        game.set_episode_timeout(sampler.episode_ticks(wad, name))
        game.init()
        game.new_episode()
        assert game.get_state() is not None
    finally:
        game.close()


def test_manifest_portable_and_full_metadata():
    sampler = MapSampler(4, 3, kinds=['full'])
    wad, name, _ = next(sampler)
    metadata = sampler.map_metadata(wad, name)
    assert metadata['tier'] == 4 and metadata['episode_ticks'] == 4200
    manifest = sampler.pool_manifest
    assert all(r['id'].startswith('@vizdoom/') for r in manifest['full_pool'])
    assert str(Path.cwd()) not in json.dumps(manifest)
    assert manifest['catalog_sha256']


def test_duplicate_identity_within_split_rejected(tmp_path):
    from training.map_pool import load_catalog_pool
    alias = str(Path('maps/two_rooms.wad').resolve()) + ':map01'
    with pytest.raises(ValueError, match='duplicate'):
        load_catalog_pool('train', [4], catalog_path=catalog(tmp_path,
            [entry(), entry(id=alias)]))


def test_hardlink_alias_cross_split_rejected(tmp_path):
    from training.map_pool import load_catalog_pool
    original = Path('maps/two_rooms.wad').resolve()
    alias = tmp_path / 'alias.wad'
    alias.hardlink_to(original)
    with pytest.raises(ValueError, match='split'):
        load_catalog_pool('train', [4], catalog_path=catalog(tmp_path,
            [entry(), entry(id=str(alias) + ':map01', split='val')]))
