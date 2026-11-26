"""Versioned, deterministic training-domain randomisation; no held-out seeds."""
from __future__ import annotations
import hashlib
import json
import random
from dataclasses import asdict
from .layouts import generate, playable
from .udmf import GridMap

PROFILE = dict(version=1, walls=['STARTAN2','STONE2','BROWN1'],
               floors=['FLOOR0_1','FLOOR4_8','FLAT5_4'], ceilings=['CEIL1_1','CEIL3_5','FLAT1'],
               light=[112,224], monster_density=[.01,.08], item_density=[.01,.05],
               spawn_candidates=64, ticks=[2100,4200])
PROFILE_HASH = hashlib.sha256(json.dumps(PROFILE,sort_keys=True).encode()).hexdigest()[:12]


def generate_randomized(kind: str, seed: int) -> GridMap:
    if type(seed) is not int or not 0 <= seed < 100000:
        raise ValueError('randomised training seeds must be in [0,100000)')
    grid = generate(kind, seed)
    rng = random.Random(seed * 1000003 + 57)
    rows = [list(row) for row in grid.rows]
    start = next(grid.cells('P'))
    floors = list(grid.cells('.'))
    rng.shuffle(floors)
    # Relocate only if the complete puzzle remains valid, including no key bypass.
    for r,c in floors[:PROFILE['spawn_candidates']]:
        rows[start[0]][start[1]]='.'; rows[r][c]='P'
        candidate=GridMap([''.join(row) for row in rows], notes=dict(grid.notes))
        if playable(candidate,kind):
            grid=candidate
            break
        rows[r][c]='.'; rows[start[0]][start[1]]='P'
    rows=[list(row) for row in grid.rows]
    for r,row in enumerate(rows):
        for c,ch in enumerate(row):
            if ch in 'zsidhaAS': rows[r][c]='.'
    free=[(r,c) for r,row in enumerate(rows) for c,ch in enumerate(row) if ch=='.']
    rng.shuffle(free)
    enemies=min(len(free),int(len(free)*rng.uniform(*PROFILE['monster_density'])))
    items=min(len(free)-enemies,int(len(free)*rng.uniform(*PROFILE['item_density'])))
    for r,c in free[:enemies]: rows[r][c]=rng.choice('zzzsid')
    for r,c in free[enemies:enemies+items]: rows[r][c]=rng.choice('haAS')
    appearance={k:rng.choice(PROFILE[k]) for k in ('walls','floors','ceilings')}
    appearance['light']=rng.randint(*PROFILE['light'])
    notes=dict(grid.notes, appearance=appearance,
               randomisation=dict(version=PROFILE['version'],profile_hash=PROFILE_HASH,kind=kind,seed=seed,
                                  episode_ticks=rng.randint(*PROFILE['ticks'])))
    result=GridMap([''.join(row) for row in rows],rng.randrange(360),notes=notes)
    if not playable(result,kind):
        raise ValueError('randomisation invalidated map')
    return result
