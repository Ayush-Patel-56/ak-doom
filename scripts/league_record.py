"""What a Mini-League result covers: its map set and the exact weights it tested.

Two Mini-League totals are only comparable on the SAME maps and seeds. The validation set grows over
time (e.g. 20 -> 24 maps when the keynear maps were added), so scripts/evaluate.py records a map-set
fingerprint and the SHA-256 of the weights, and scripts/export_champion.py refuses to compare results
whose fingerprints differ.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def map_set_of(runs: list[dict]) -> list[str]:
    """Sorted, de-duplicated 'map#seed' entries of a Mini-League result."""
    return sorted({f"{r['map']}#{r['seed']}" for r in runs})


def fingerprint(entries: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(entries)).encode()).hexdigest()[:16]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_result(result: dict) -> dict:
    """Map-set info for a Mini-League JSON (works for old files that only have 'runs')."""
    entries = result.get("map_set") or map_set_of(result["runs"])
    return {"map_set_fingerprint": result.get("map_set_fingerprint") or fingerprint(entries),
            "maps": len({e.rsplit("#", 1)[0] for e in entries}), "runs": len(entries),
            "weights_sha256": result.get("weights_sha256")}


def load_result(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
