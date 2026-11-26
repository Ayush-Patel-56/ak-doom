"""Parse a 'path/to.wad:MAP01' argument. Shared by the Phase 0 scripts."""

from __future__ import annotations

import re


def parse_map(spec: str) -> tuple[str, str]:
    """Split "path/to.wad:MAP05" into (wad, map). Safe with Windows drive letters like "F:\\..."."""
    wad, sep, name = spec.rpartition(":")
    if sep and re.fullmatch(r"(MAP\d\d|E\dM\d)", name, re.IGNORECASE):
        return wad, name.upper()
    return spec, "MAP01"
