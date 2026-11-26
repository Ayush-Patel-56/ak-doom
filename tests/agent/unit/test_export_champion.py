"""export_champion.py must only compare Mini-League totals measured on the SAME maps and weights.

Regression: champion #1 was scored on the 20-map val set; after 4 keynear maps were added (24 maps) a
worse candidate's bigger 24-map total would have beaten the stale 20-map number.
"""

from __future__ import annotations

import json
import sys

import pytest
import torch

from conftest import REPO

sys.path.insert(0, str(REPO / "scripts"))
import export_champion as ec  # noqa: E402
from league_record import file_sha256, fingerprint, map_set_of  # noqa: E402
from smriti.brain import BrainConfig, SmritiNet  # noqa: E402
from training.checkpoints import save_checkpoint  # noqa: E402

MAPS_20 = [f"maps/curated/val/m{i}.wad:MAP01" for i in range(20)]
MAPS_24 = MAPS_20 + [f"maps/curated/val/keynear{i}.wad:MAP01" for i in range(4)]


@pytest.fixture()
def weights_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CHAMPION", tmp_path / "champion.pt")
    monkeypatch.setattr(ec, "CHAMPION_INFO", tmp_path / "champion.json")
    return tmp_path


def brain(path, seed):
    torch.manual_seed(seed)
    return save_checkpoint(path, SmritiNet(BrainConfig()))


def result(path, weights, maps, total, stopped=None, with_sha=True):
    runs = [{"map": m, "seed": s, "stopped": stopped} for m in maps for s in (0, 1)]
    data = {"total": total, "by_tier": {"1": total}, "runs": runs}
    if with_sha:
        data["weights_sha256"] = file_sha256(weights)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_fingerprint_depends_on_maps_and_seeds_not_order():
    a = map_set_of([{"map": "x", "seed": 0}, {"map": "y", "seed": 1}])
    b = map_set_of([{"map": "y", "seed": 1}, {"map": "x", "seed": 0}])
    c = map_set_of([{"map": "x", "seed": 0}, {"map": "y", "seed": 0}])
    assert fingerprint(a) == fingerprint(b) != fingerprint(c)


def test_a_bigger_total_on_more_maps_does_not_beat_the_champion(weights_dir):
    champ = brain(weights_dir / "champion.pt", 0)
    ec.main(["--rebaseline", str(result(weights_dir / "c20.json", champ, MAPS_20, 2021.0))])
    cand = brain(weights_dir / "cand.pt", 1)
    ev = result(weights_dir / "cand24.json", cand, MAPS_24, 2463.0)  # bigger number, different maps
    assert ec.main([str(cand), "--eval", str(ev)]) == 1
    assert file_sha256(weights_dir / "champion.pt") == file_sha256(champ)  # champion unchanged


def test_same_maps_compare_fairly(weights_dir):
    champ = brain(weights_dir / "champion.pt", 0)
    c24 = result(weights_dir / "c24.json", champ, MAPS_24, 2511.0)
    cand = brain(weights_dir / "cand.pt", 1)
    worse = result(weights_dir / "worse.json", cand, MAPS_24, 2463.0)
    assert ec.main([str(cand), "--eval", str(worse), "--champion-eval", str(c24)]) == 1
    better = result(weights_dir / "better.json", cand, MAPS_24, 2600.0)
    assert ec.main([str(cand), "--eval", str(better), "--champion-eval", str(c24)]) == 0
    assert file_sha256(weights_dir / "champion.pt") == file_sha256(cand)
    info = json.loads((weights_dir / "champion.json").read_text(encoding="utf-8"))
    (fp, stored), = info["results"].items()
    assert stored["total"] == 2600.0 and stored["maps"] == 24 and info["weights_sha256"] == file_sha256(cand)


def test_results_must_come_from_the_right_weights(weights_dir):
    champ = brain(weights_dir / "champion.pt", 0)
    other = brain(weights_dir / "other.pt", 2)
    cand = brain(weights_dir / "cand.pt", 1)
    c24_wrong = result(weights_dir / "c24.json", other, MAPS_24, 100.0)  # not the champion's weights
    ev = result(weights_dir / "cand.json", cand, MAPS_24, 2600.0)
    assert ec.main([str(cand), "--eval", str(ev), "--champion-eval", str(c24_wrong)]) == 1
    assert ec.main(["--rebaseline", str(c24_wrong)]) == 1
    no_sha = result(weights_dir / "nosha.json", cand, MAPS_24, 2600.0, with_sha=False)
    assert ec.main([str(cand), "--eval", str(no_sha), "--champion-eval",
                    str(result(weights_dir / "c.json", champ, MAPS_24, 1.0))]) == 1


def test_crashes_refuse_promotion(weights_dir):
    champ = brain(weights_dir / "champion.pt", 0)
    cand = brain(weights_dir / "cand.pt", 1)
    c24 = result(weights_dir / "c24.json", champ, MAPS_24, 1.0)
    crashed = result(weights_dir / "crash.json", cand, MAPS_24, 9999.0, stopped="exception: X")
    assert ec.main([str(cand), "--eval", str(crashed), "--champion-eval", str(c24)]) == 1
