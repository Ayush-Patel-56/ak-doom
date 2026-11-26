"""Explicit, persisted reservation of known held-out test-map evaluation.

This records future use, not a guarantee that historical test data was never viewed.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from training.map_pool import DEFAULT_CATALOG, load_catalog_pool, map_identity

DEFAULT_LEDGER = Path(__file__).resolve().parents[1]/'runs'/'final-test-ledger'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def physical_identity(spec):
    wad,name=map_identity(spec)
    stat=Path(wad).stat()
    return [stat.st_dev,stat.st_ino,name]


def reserve_test_use(specs, weights, *, final_test=False, allow_repeat=False,
                     ledger=DEFAULT_LEDGER, catalog_path=DEFAULT_CATALOG):
    known={tuple(physical_identity(r['id'])):r['id'] for r in
           load_catalog_pool('test',[1,2,3,4],catalog_path=catalog_path)}
    selected={known[tuple(physical_identity(spec))] for spec in specs
              if tuple(physical_identity(spec)) in known}
    if not selected: return None
    if not final_test: raise ValueError('known test maps require --final-test and a recorded reservation')
    checkpoint=digest(weights)
    folder=Path(ledger); folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:
        raise RuntimeError('test-use ledger locked by another evaluation; inspect before retrying') from exc
    try:
        prior=[json.loads(p.read_text()) for p in folder.glob('*.json')]
        overlaps=[r['id'] for r in prior if selected.intersection(r['maps'])]
        if overlaps and not allow_repeat:
            raise ValueError('test maps already reserved/evaluated; --allow-repeat-test required')
        token=uuid.uuid4().hex
        row=dict(id=token,created_unix=time.time(),checkpoint_sha256=checkpoint,maps=sorted(selected),
                 status='reserved',prior_overlapping_uses=overlaps,historical_freshness='unknown')
        (folder/f'{token}.json').write_text(json.dumps(row,indent=2)+'\n')
        return token
    finally:
        os.close(fd); lock.unlink()


def finish_test_use(token, *, status, ledger=DEFAULT_LEDGER):
    if token is None: return
    if status not in ('completed','failed','partial'): raise ValueError('invalid final-test status')
    path=Path(ledger)/f'{token}.json'
    row=json.loads(path.read_text());row.update(status=status,finished_unix=time.time())
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)
