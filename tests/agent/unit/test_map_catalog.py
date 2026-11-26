"""Unit tests for scripts/build_map_catalog.py: UDMF maps (the format of the template's own maps)."""

from __future__ import annotations

import sys

from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))
from build_map_catalog import describe, udmf_stats  # noqa: E402

UDMF = """namespace = "zdoom";
thing { x = 0.0; y = 0.0; type = 1; skill3 = true; single = true; }
thing { x = 1.0; y = 0.0; type = 3004; skill3 = true; single = true; }
thing { x = 2.0; y = 0.0; type = 3001; skill1 = true; single = true; }
thing { x = 3.0; y = 0.0; type = 5; skill3 = true; single = true; }
thing { x = 4.0; y = 0.0; type = 9; skill3 = true; single = false; }
linedef { v1 = 0; v2 = 1; sidefront = 0; special = 243; playercross = true; }
linedef { v1 = 1; v2 = 2; sidefront = 0; special = 13; arg3 = 130; playeruse = true; }
"""


def test_udmf_counts_skill3_single_player_things_and_exit_lines():
    s = udmf_stats(UDMF, has_behavior=False)
    assert s == {"monsters": 1, "keys": 1, "exit_lines": 1}


def test_udmf_exit_unknown_when_acs_may_exit():
    no_exit = UDMF.replace("special = 243;", "special = 0;")
    assert udmf_stats(no_exit, has_behavior=True)["exit_lines"] is None
    assert udmf_stats(no_exit, has_behavior=False)["exit_lines"] == 0


def test_template_test_map_is_described_from_its_textmap():
    m = describe("maps/two_rooms.wad", "template", "MAP01")
    assert m["format"].startswith("udmf")
    assert m["exit_lines"] >= 1 and m["has_exit"] is True
    assert m["monsters"] == 1 and m["keys"] == 0

