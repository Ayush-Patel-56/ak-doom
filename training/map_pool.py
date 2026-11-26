"""Validated, split-isolated catalog pools with portable asset resolution."""
from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.map_spec import parse_map

DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / 'maps' / 'catalog.json'
SPLITS = {'train', 'val', 'test'}


def map_identity(spec: str) -> tuple[str, str]:
    """Canonical asset/map identity, including symlink and portable aliases."""
    wad, name = parse_map(spec)
    return str(Path(wad).resolve()), name.upper()


def load_catalog_pool(split, tiers, source=None, catalog_path=DEFAULT_CATALOG,
                      require_assets=True) -> list[dict]:
    """Copy selected usable entries, rejecting malformed metadata and split leakage.

    All catalog identities are checked before filtering; duplicates are rejected
    even within a split to preserve sampling weights. Aliases in a different
    tier/source cannot hide held-out exposure. Asset existence applies to the
    selected pool, so unrelated optional scenario installations are not required.
    """
    tiers = set(tiers)
    if split not in SPLITS or not tiers or any(type(t) is not int or t not in range(1, 5) for t in tiers):
        raise ValueError('invalid split or tiers')
    data = json.loads(Path(catalog_path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get('maps'), list):
        raise ValueError('catalog must contain a maps list')
    identities = {}
    selected = []
    for row in data['maps']:
        if not isinstance(row, dict):
            raise ValueError('catalog entries must be objects')
        if (not isinstance(row.get('id'), str) or not re.search(r':(?:MAP\d\d|E\dM\d)$', row['id'], re.I)
                or not row['id'].rsplit(':', 1)[0]
                or not isinstance(row.get('source'), str) or not row['source']
                or row.get('split') not in SPLITS
                or type(row.get('tier')) is not int or row['tier'] not in range(1, 5)
                or type(row.get('usable')) is not bool
                or type(row.get('episode_ticks')) is not int or not 2100 <= row['episode_ticks'] <= 4200):
            raise ValueError(f'invalid catalog metadata: {row!r}')
        identity = map_identity(row['id'])
        # Existing files can have distinct realpaths yet refer to one asset
        # (hard links or alternate casing on a case-insensitive filesystem).
        # Map names remain part of the key: levels in one WAD are distinct.
        try:
            stat = Path(identity[0]).stat()
        except FileNotFoundError:
            asset_identity = ('path', identity[0], identity[1])
        else:
            asset_identity = ('inode', stat.st_dev, stat.st_ino, identity[1])
        previous = identities.get(asset_identity)
        if previous is not None and previous != row['split']:
            raise ValueError(f'catalog identity crosses splits: {row["id"]} ({previous}/{row["split"]})')
        if previous is not None:
            raise ValueError(f'duplicate catalog identity within split {previous}: {row["id"]}')
        identities[asset_identity] = row['split']
        if row['usable'] and row['split'] == split and row['tier'] in tiers and (source is None or row['source'] == source):
            if require_assets and not Path(identity[0]).is_file():
                raise FileNotFoundError(f'catalog asset missing: {row["id"]} -> {identity[0]}')
            selected.append(dict(row, resolved_wad=identity[0], map_name=identity[1]))
    return selected
