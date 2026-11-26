"""Training-only, conservative exit-floor visibility for generated flat UDMF maps.

Identity comes from the back sector of actual Exit_Normal (243) crossing lines,
never a texture. Pixel-centre floor rays use the live camera and exact WAD line
geometry; portal openings use live engine sector heights (including moving doors).
Only axis-aligned 64-unit generated geometry, flat zero floors, 128-unit rooms,
zero pitch/roll, and the wrapper's 160x120 software renderer are supported.
Unsupported or ambiguous observations return an unknown mask, never a negative.
Live renderer label pixels exclude sprites, including the weapon, from positives.
A two-pixel interior margin makes positive evidence robust to raster rounding;
ambiguous tiny visible footprints are masked. Geometry proves negative labels.
Construction enables labels by restarting the configured engine before first reset.
"""
from __future__ import annotations

import numpy as np


class ExitVisibility:
    def __init__(self, env):
        self.env = env
        self.supported = False
        self.exits = set()

    @classmethod
    def from_env(cls, env):
        obj = cls(env)
        try:
            from doom_env.mapinfo import parse_udmf
            from doom_env.wad import read_wad, map_lumps
            text = map_lumps(read_wad(env.wad), env.map_name)['TEXTMAP'].decode()
            if 'namespace = "zdoom";' not in text:
                return obj
            blocks = parse_udmf(text)
            vertices, sectors = blocks['vertex'], blocks['sector']
            sides, lines = blocks['sidedef'], blocks['linedef']
            if any(s.get('heightfloor', 0) != 0 or s.get('heightceiling') not in (0, 128)
                   or set(s) - {'heightfloor', 'heightceiling', 'texturefloor', 'textureceiling', 'lightlevel'}
                   for s in sectors):
                return obj
            for line in lines:
                a, b = vertices[line['v1']], vertices[line['v2']]
                if any(float(v[k]) % 64 for v in (a, b) for k in ('x', 'y')):
                    return obj
                if (a['x'] != b['x'] and a['y'] != b['y']) or line.get('special', 0) not in (0, 12, 13, 243):
                    return obj
                if 'sideback' in line and any(sides[line[k]].get('texturemiddle', '-') != '-'
                                                  for k in ('sidefront', 'sideback')):
                    return obj
                if line.get('special') == 243:
                    if not line.get('playercross') or 'sideback' not in line:
                        return obj
                    obj.exits.add(sides[line['sideback']]['sector'])
            if not obj.exits:
                return obj
            obj.geometry = env.mapinfo.lines
            obj.boundaries = env.mapinfo.sector_lines()
            obj.supported = obj.geometry is not None and bool(env.render_buffers)
            if obj.supported:
                # ViZDoom buffer configuration takes effect only at init. This
                # training-only constructor must be called before env.reset().
                if env._geometry is not None:
                    obj.supported = env._game.get_state().labels_buffer is not None
                else:
                    env._game.close()
                    env._game.set_labels_buffer_enabled(True)
                    env._game.init()
        except (AttributeError, KeyError, ValueError, OSError, IndexError):
            pass
        return obj

    def label(self):
        if not self.supported:
            return 0., False
        try:
            import vizdoom as vd
            game = self.env._game
            state = game.get_state()
            if state is None or state.screen_buffer is None or state.labels_buffer is None:
                return 0., False
            def camera(name):
                return float(game.get_game_variable(getattr(vd.GameVariable, 'CAMERA_' + name)))
            x, y, z = (camera('POSITION_' + axis) for axis in 'XYZ')
            angle, pitch, roll, fov = (camera(n) for n in ('ANGLE', 'PITCH', 'ROLL', 'FOV'))
            if abs(pitch) > 1e-5 or abs(roll) > 1e-5 or not (0 < fov < 180 and z > 0):
                return 0., False
            h, w = state.labels_buffer.shape
            if (h, w) != (120, 160):
                return 0., False
            focal = w / (2 * np.tan(np.deg2rad(fov) / 2))
            # Screen x increases to camera-right; Doom yaw increases anticlockwise.
            u, v = np.meshgrid((np.arange(w) + .5 - w/2) / focal,
                               (np.arange(h) + .5 - h/2) / (focal * 1.2))
            valid = v > 0
            a = np.deg2rad(angle)
            direction = np.stack((np.cos(a) + u*np.sin(a), np.sin(a) - u*np.cos(a)), -1)
            distance = np.divide(z, v, out=np.zeros_like(v), where=valid)
            delta = direction * distance[..., None]
            point = np.array([x, y]) + delta
            inside = np.zeros((h, w), bool)
            for sec in self.exits:
                polygon = self.boundaries[sec]
                crossing = np.zeros((h, w), bool)
                for x1, y1, x2, y2 in polygon:
                    if y1 == y2:
                        continue
                    crossing ^= ((y1 > point[..., 1]) != (y2 > point[..., 1])) & (
                        point[..., 0] < (x2-x1)*(point[..., 1]-y1)/(y2-y1)+x1)
                inside |= crossing
            visible = valid & inside
            g = self.geometry
            heights = {s.id: (s.floor_height, s.ceiling_height) for s in state.sectors}
            for p1, p2, front, back in zip(g.p1, g.p2, g.front, g.back):
                if not np.any(visible):
                    break
                edge = p2-p1
                offset = p1 - np.array([x, y])
                den = delta[..., 0]*edge[1] - delta[..., 1]*edge[0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = (offset[0]*edge[1]-offset[1]*edge[0])/den
                    s = (offset[0]*delta[..., 1]-offset[1]*delta[..., 0])/den
                intersects = (t > 1e-7) & (t < 1-1e-7) & (s >= 0) & (s <= 1)
                if back < 0:
                    visible &= ~intersects
                else:
                    low = max(heights[int(front)][0], heights[int(back)][0])
                    high = min(heights[int(front)][1], heights[int(back)][1])
                    ray_z = z*(1-t)
                    visible &= ~(intersects & ((ray_z <= low) | (ray_z >= high)))
            if not np.any(visible):
                return 0., True
            # Require a small interior footprint; sprite labels include the
            # weapon (255); renderer flats have value 1. Erosion avoids boundary rasterization ambiguity.
            interior = visible.copy()
            for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                interior &= np.roll(visible, (dy, dx), (0, 1))
            interior[:2] = interior[-2:] = False
            interior[:, :2] = interior[:, -2:] = False
            if np.any(interior & (state.labels_buffer == 1)):
                return 1., True
            return 0., False
        except (AttributeError, KeyError, ValueError, RuntimeError):
            return 0., False
