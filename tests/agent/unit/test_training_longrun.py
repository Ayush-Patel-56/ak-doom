"""Long-run robustness of the trainer (training/rollout.py, training/train.py).

Ported from the first 6M-step run, which needed all three: the Windows timer fix (speed), skipping a
bad map (it crashed at 3.62M steps on one) and resuming from latest.pt + train_state.pt.
"""

from __future__ import annotations

import json
import shutil
import uuid

import numpy as np

from conftest import REPO
from training.mapgen import generate, write_map
from training.rollout import Game, fine_timer


def test_fine_timer_is_safe_on_every_os():
    fine_timer()  # Windows: timeBeginPeriod(1); elsewhere a no-op
    fine_timer()


def test_game_skips_a_map_that_fails_to_load(tmp_path):
    good = write_map(generate("arena", 7), tmp_path / "arena-7.wad")
    picks = iter([(str(tmp_path / "missing.wad"), "MAP01", "arena"), (str(good), "MAP01", "arena")])
    game = Game(picks, use_mindmap=False, use_notebook=False, episode_ticks=200, switch_every=8,
                rng=np.random.default_rng(0))
    try:
        feats = game.start()  # the missing map is skipped, the good one loads
        assert game.wad == str(good) and feats["ray_dist"].shape == (64,)
    finally:
        game.close()


def test_resume_continues_from_the_saved_step():
    from training.train import main

    name = f"test-resume-{uuid.uuid4().hex[:8]}"
    out = REPO / "runs" / name
    common = ["--stage", "3", "--memory", "off", "--kinds", "arena", "--workers", "1", "--games", "2",
              "--T", "16", "--epochs", "1", "--minibatches", "1", "--threads", "1", "--ticks", "400", "--name", name]
    try:
        assert main(common + ["--steps", "32"]) == 0  # 1 iteration of 16 steps x 2 games
        assert (out / "train_state.pt").is_file()
        assert main(common + ["--steps", "96", "--resume"]) == 0  # continues: 2 more iterations, not 3
        rows = [json.loads(ln) for ln in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["steps"] for r in rows] == [32, 64, 96]
        assert [r["it"] for r in rows] == [1, 2, 3]
    finally:
        shutil.rmtree(out, ignore_errors=True)
