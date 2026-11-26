import json
import numpy as np
import pytest
from training.mapgen import playable, write_map


def test_randomised_reproducible_playable_and_varied():
    from training.mapgen.randomize import generate_randomized, PROFILE_HASH
    for kind in ('arena','maze','rooms','keydoor','keynear'):
        signatures = set()
        for seed in range(8):
            a = generate_randomized(kind, seed)
            assert a == generate_randomized(kind, seed)
            assert playable(a, kind)
            assert a.notes['randomisation']['profile_hash'] == PROFILE_HASH
            signatures.add(json.dumps(a.notes['appearance'], sort_keys=True))
        assert len(signatures) > 1
    with pytest.raises(ValueError):
        generate_randomized('arena', 100000)


def test_randomised_real_engine(tmp_path):
    from training.mapgen.randomize import generate_randomized
    from doom_env import make_env
    path = write_map(generate_randomized('keydoor', 41), tmp_path/'random.wad')
    env = make_env(str(path), 'MAP01', episode_ticks=32, render_buffers=True)
    try:
        obs, _ = env.reset(seed=5)
        assert np.any(obs['rgb']) and np.any(obs['depth'])
        env.step(np.zeros(9, np.int8))
    finally:
        env.close()


def test_randomisation_assets_exist_in_iwad():
    import struct
    from doom_env.wad import read_wad
    from scripts.map_spec import vizdoom_dir
    from training.mapgen.randomize import PROFILE
    lumps=read_wad(vizdoom_dir()/'freedoom2.wad')
    names={x.name for x in lumps}
    assert set(PROFILE['floors']+PROFILE['ceilings']) <= names
    textures=set()
    for lump in lumps:
        if lump.name in ('TEXTURE1','TEXTURE2'):
            count=struct.unpack_from('<i',lump.data)[0]
            for i in range(count):
                off=struct.unpack_from('<i',lump.data,4+4*i)[0]
                textures.add(lump.data[off:off+8].rstrip(b'\0').decode())
    assert set(PROFILE['walls']) <= textures
