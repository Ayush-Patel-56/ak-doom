"""Parse map arguments like 'path/to.wad:MAP01'. Shared by the scripts.

WAD paths may use two portable prefixes, so map lists work on every machine:
- "@vizdoom/<file>"  a WAD bundled with the vizdoom package (e.g. "@vizdoom/freedoom2.wad",
                     "@vizdoom/scenarios/my_way_home.wad")
- a relative path    resolved against the current directory first, then the repo root
                     (e.g. "maps/two_rooms.wad")
"""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath

REPO = Path(__file__).resolve().parent.parent
VIZDOOM_PREFIX = "@vizdoom/"


def vizdoom_dir() -> Path:
    import vizdoom

    return Path(vizdoom.__file__).resolve().parent


def resolve_wad(wad: str) -> str:
    """Turn a portable WAD reference into a real file path."""
    if wad.startswith(VIZDOOM_PREFIX):
        return str(vizdoom_dir() / wad[len(VIZDOOM_PREFIX):])
    if PureWindowsPath(wad).drive:
        # "C:\\maps\\x.wad" is absolute on Windows but not on macOS/Linux: never prefix the repo root.
        return wad
    path = Path(wad)
    if path.is_absolute() or path.exists():
        return str(path)
    return str(REPO / path)


def parse_map(spec: str) -> tuple[str, str]:
    """Split "path/to.wad:MAP05" into (resolved wad path, map). Safe with Windows drive letters."""
    wad, sep, name = spec.rpartition(":")
    if sep and re.fullmatch(r"(MAP\d\d|E\dM\d)", name, re.IGNORECASE):
        return resolve_wad(wad), name.upper()
    return resolve_wad(spec), "MAP01"
