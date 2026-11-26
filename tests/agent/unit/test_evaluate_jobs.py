"""evaluate.py --jobs N must give exactly the same Mini-League results as --jobs 1."""

from __future__ import annotations

import json
import sys
import uuid

import pytest

from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))

CHAMPION = REPO / "submission" / "weights" / "champion.pt"
MAPS = ["maps/curated/val/t1-arena-100001.wad:MAP01", "maps/curated/val/t2-maze-100003.wad:MAP01"]
SCORE_FIELDS = ("score", "weighted_score", "exit", "keys", "doors", "locked_doors", "areas", "kills", "damage",
                "dead", "steps", "stopped", "buckets")


def _run(jobs: int) -> dict:
    import evaluate

    name = f"test-jobs{jobs}-{uuid.uuid4().hex[:6]}"
    argv = ["--seeds", "2", "--no-buffers", "--jobs", str(jobs), "--name", name]
    for m in MAPS:
        argv += ["--map", m]
    assert evaluate.main(argv) == 0
    out = REPO / "runs" / "eval" / f"{name}.json"
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    finally:
        out.unlink(missing_ok=True)


@pytest.mark.skipif(not CHAMPION.is_file(), reason="needs the trained champion (strict loader)")
def test_parallel_jobs_match_a_sequential_run():
    seq, par = _run(1), _run(2)
    assert seq["total"] == par["total"] and seq["by_tier"] == par["by_tier"]
    assert seq["map_set_fingerprint"] == par["map_set_fingerprint"]
    assert [(r["map"], r["seed"]) for r in seq["runs"]] == [(r["map"], r["seed"]) for r in par["runs"]]
    for a, b in zip(seq["runs"], par["runs"]):
        assert {k: a[k] for k in SCORE_FIELDS} == {k: b[k] for k in SCORE_FIELDS}, (a["map"], a["seed"])
