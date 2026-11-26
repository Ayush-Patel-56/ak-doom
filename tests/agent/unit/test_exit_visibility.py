"""Visibility labels checked against actual generated maps rendered by ViZDoom."""
import numpy as np
import pytest
from doom_env import make_env
from training.mapgen.udmf import GridMap, write_map
from training.exit_visibility import ExitVisibility


@pytest.mark.parametrize('rows,angle,expected', [
    ('########\n#P..XX.#\n########', 0, (1., True)),
    ('########\n#P..XX.#\n########', 180, (0., True)),
    ('########\n#P.#.X.#\n#......#\n########', 0, (0., True)),
    ('########\n#P.DXX.#\n########', 0, (0., True)),
])
def test_real_rendered_views(tmp_path, rows, angle, expected):
    env = make_env(str(write_map(GridMap.parse(rows, player_angle=angle), tmp_path/'view.wad')),
                   'MAP01', render_buffers=True)
    try:
        labeler = ExitVisibility.from_env(env)  # construction before reset is supported
        obs, _ = env.reset(seed=0)
        assert obs['rgb'].shape[-1] == 3
        assert np.std(obs['rgb']) > 1  # real nonempty rendered observation
        assert labeler.label() == expected
    finally:
        env.close()


def test_visible_exit_with_real_item_sprite(tmp_path):
    env = make_env(str(write_map(GridMap.parse('########\n#P.aXX.#\n########'), tmp_path/'sprite.wad')),
                   'MAP01', render_buffers=True)
    try:
        labeler = ExitVisibility.from_env(env)
        env.reset(seed=0)
        assert labeler.label() == (1., True)
    finally:
        env.close()


def test_live_door_height_changes_visibility(tmp_path):
    env = make_env(str(write_map(GridMap.parse('########\n#P.DXX.#\n########'), tmp_path/'door.wad')),
                   'MAP01', render_buffers=True)
    try:
        labeler = ExitVisibility.from_env(env)
        env.reset(seed=0)
        assert labeler.label() == (0., True)
        # Engine movement and USE actions, no invented sector state.
        forward = np.zeros(9, np.int8); forward[0] = 1
        for _ in range(12):
            env.step(forward)
        use = np.zeros(9, np.int8); use[7] = 1
        env.step(use)
        for _ in range(35):
            env.step(np.zeros(9, np.int8))
        assert labeler.label() == (1., True)
    finally:
        env.close()


def test_unknown_without_exit_action_even_with_exit_texture(tmp_path):
    from doom_env.wad import Lump, write_wad
    from training.mapgen.udmf import to_udmf
    # Keep the same visible floor texture but remove its actual Exit_Normal action.
    text = to_udmf(GridMap.parse('########\n#P..XX.#\n########')).replace('special = 243;', 'special = 0;')
    wad = tmp_path/'unsupported.wad'
    write_wad(wad, [Lump('MAP01', b''), Lump('TEXTMAP', text.encode()), Lump('ENDMAP', b'')])
    env = make_env(str(wad), 'MAP01', render_buffers=True)
    try:
        labeler = ExitVisibility.from_env(env)
        env.reset(seed=0)
        assert labeler.label() == (0., False)
    finally:
        env.close()


def test_exit_texture_is_not_identity(tmp_path):
    from doom_env.wad import Lump, write_wad
    from training.mapgen.udmf import to_udmf
    text = to_udmf(GridMap.parse('########\n#P..XX.#\n########')).replace('GATE3', 'FLOOR0_1')
    wad = tmp_path/'texture.wad'
    write_wad(wad, [Lump('MAP01', b''), Lump('TEXTMAP', text.encode()), Lump('ENDMAP', b'')])
    env = make_env(str(wad), 'MAP01', render_buffers=True)
    try:
        labeler = ExitVisibility.from_env(env)
        env.reset(seed=0)
        assert labeler.label() == (1., True)
    finally:
        env.close()
