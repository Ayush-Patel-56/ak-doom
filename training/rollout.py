"""Parallel game runner for PPO: worker processes, each running several doom_env games.

Every game uses the SAME per-decision code path as DoomAgent: EpisodeMemory.observe(executed action,
observation) then smriti.features.extract(...). Rewards come from training/rewards.py (privileged
state, training only). render_buffers=False: Phase 3 does not read RGB/depth.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


MAP_ATTEMPTS = 8  # a bad map must never end a multi-hour run: try other maps first


def fine_timer() -> None:
    """Ask Windows for 1 ms timer resolution (no-op elsewhere).

    Windows wakes sleeping threads only every ~15.6 ms by default. Workers wait for their next command
    between steps and the engine's waits snap to that tick: lockstep stepping was ~10x slower without
    this (measured on a 20-core laptop: 2 workers 100 steps/s in total vs 600 for one).
    """
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)


def _paths():
    for p in (str(REPO / "submission"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)


class Game:
    """One doom_env game + its memory, reward shaper and episode bookkeeping."""

    def __init__(self, sampler, use_mindmap: bool, use_notebook: bool, episode_ticks: int, switch_every: int, rng,
                 use_rgb=False, goal_helpers=False, helper_weights=None, use_depth=False):
        from smriti.memory import EpisodeMemory, Odometer

        self.sampler, self.ticks, self.switch_every, self.rng = sampler, episode_ticks, switch_every, rng
        self.memory = EpisodeMemory(Odometer(), use_mindmap=use_mindmap, use_notebook=use_notebook)
        self.uses_memory = use_mindmap or use_notebook
        self.env, self.episodes_on_map = None, 0
        from training.rewards import RewardShaper

        self.shaper = RewardShaper()
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.labeler = None
        from training.goal_rewards import GoalHelpers
        self.helpers = GoalHelpers(helper_weights) if goal_helpers else None

    def _features(self, obs):
        from smriti.features import extract

        mem = self.memory.features() if self.uses_memory else None
        feats = extract(obs, self.prev_action, mem)
        target, known = self.labeler.label() if self.labeler is not None else (0., False)
        feats['_exit_target'], feats['_exit_known'] = np.array(target, np.float32), np.array(known, bool)
        feats['_extrinsic'], feats['_helper'] = np.array(0., np.float32), np.array(0., np.float32)
        return feats

    def start(self):
        from doom_env import make_env

        if self.env is None or self.episodes_on_map >= self.switch_every:
            if self.env is not None:
                self.env.close()
                self.env = None
            self._open_next_map(make_env)
            self.episodes_on_map = 0
        obs, _ = self.env.reset(seed=int(self.rng.integers(2**31 - 1)))
        self.memory.reset()
        self.memory.observe(None, obs, obs)  # the reset observation is not a transition
        self.shaper.reset(self.env.privileged_state())
        if self.helpers is not None:
            self.helpers.reset(self.env.privileged_state(), obs)
        self.helper_ret = 0.0
        self.prev_action = np.zeros(9, np.float32)
        self.ret, self.len = 0.0, 0
        return self._features(obs)

    def _open_next_map(self, make_env) -> None:
        """Load the sampler's next map; skip (with a warning) any map that fails to build or load."""
        last_error = None
        for _ in range(MAP_ATTEMPTS):
            try:
                wad, map_name, kind = next(self.sampler)
                cap = self.ticks if self.ticks is not None else self.sampler.episode_ticks(wad, map_name)
                self.env = make_env(wad, map_name, episode_ticks=cap,
                                    render_buffers=self.use_rgb or self.use_depth)
                self.current_episode_ticks = cap
                self.wad, self.map, self.kind = wad, map_name, kind
                if self.use_rgb:
                    from training.exit_visibility import ExitVisibility
                    self.labeler = ExitVisibility.from_env(self.env)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[rollout] skipping a map that failed to load: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
        raise RuntimeError(f"{MAP_ATTEMPTS} maps in a row failed to load") from last_error

    def step(self, buttons: np.ndarray, goal=None):
        obs, _, term, trunc, _ = self.env.step(buttons)
        r = self.shaper(self.env.privileged_state(), buttons)
        helper = self.helpers(self.env.privileged_state(), int(goal), None if term or trunc else obs) if self.helpers is not None else 0.
        self.helper_ret += helper
        self.ret += r
        self.len += 1
        self.prev_action = buttons.astype(np.float32)
        done = bool(term or trunc)
        info = None
        if done:
            info = {"map": Path(self.wad).name, "kind": self.kind, "ret": self.ret, "len": self.len,
                    **self.shaper.stats(), 'helper_ret': self.helper_ret,
                    'map_name': self.map, 'episode_ticks': self.current_episode_ticks,
                    'map_metadata': self.sampler.map_metadata(self.wad, self.map)}
            self.episodes_on_map += 1
            feats = self.start()
        else:
            self.memory.observe(buttons, obs, obs)
            feats = self._features(obs)
        feats['_extrinsic'], feats['_helper'] = np.array(r, np.float32), np.array(helper, np.float32)
        return feats, r + helper, done, info

    def close(self):
        if self.env is not None:
            self.env.close()


def _worker(remote, wid, n_games, stage, seed, use_mindmap, use_notebook, episode_ticks, switch_every, kinds, use_rgb, goal_helpers, helper_weights, puzzle_share=None, use_depth=False, full_level_share=None, catalog_path=None):
    _paths()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    fine_timer()
    games = []
    try:
        def stack(items):
            return {k: np.stack([item[k] for item in items]) for k in items[0]}
        from smriti.policy.actions import groups_to_buttons
        from training.curriculum import MapSampler

        sampler = MapSampler(stage, seed * 1009 + wid, kinds=kinds, puzzle_share=puzzle_share,
                             full_level_share=full_level_share,
                             **({'catalog_path': catalog_path} if catalog_path is not None else {}))
        rng = np.random.default_rng(seed * 7919 + wid)
        games = [Game(sampler, use_mindmap, use_notebook, episode_ticks, switch_every, rng, use_rgb, goal_helpers, helper_weights, use_depth) for _ in range(n_games)]
        remote.send(("ok", stack([g.start() for g in games])))
        while True:
            cmd, data = remote.recv()
            if cmd == "close":
                break
            choices, progress, goals = data
            sampler.progress = progress
            buttons = groups_to_buttons(choices)
            out = [g.step(buttons[i], None if goals is None else goals[i]) for i, g in enumerate(games)]
            remote.send(("ok", (stack([o[0] for o in out]), np.array([o[1] for o in out], np.float32),
                                np.array([o[2] for o in out], bool), [o[3] for o in out])))
    except Exception:  # noqa: BLE001
        remote.send(("error", traceback.format_exc()))
    finally:
        for g in games:
            try:
                g.close()
            except Exception:  # noqa: BLE001
                pass


class VecGames:
    def __init__(self, workers: int, games_per_worker: int, stage: int, seed: int, use_mindmap: bool,
                 use_notebook: bool, episode_ticks: int | None = None, switch_every: int = 2, kinds=None,
                 use_rgb=False, goal_helpers=False, helper_weights=None, puzzle_share=None,
                 use_depth=False, full_level_share=None, catalog_path=None):
        ctx = mp.get_context("spawn")  # same on Linux, macOS and Windows
        self.k, self.n = games_per_worker, workers * games_per_worker
        self.remotes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(b, w, games_per_worker, stage, seed, use_mindmap, use_notebook, episode_ticks,
                                  switch_every, kinds, use_rgb, goal_helpers, helper_weights, puzzle_share,
                                  use_depth, full_level_share, catalog_path))
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.first = self._cat([self._recv(r) for r in self.remotes])

    @staticmethod
    def _recv(remote):
        status, payload = remote.recv()
        if status == "error":
            raise RuntimeError("game worker failed:\n" + payload)
        return payload

    @staticmethod
    def _cat(ds):
        return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}

    def step(self, choices: np.ndarray, progress: float, goals=None):
        for i, r in enumerate(self.remotes):
            r.send(("step", (choices[i * self.k:(i + 1) * self.k], progress,
                             None if goals is None else goals[i * self.k:(i + 1) * self.k])))
        outs = [self._recv(r) for r in self.remotes]
        feats = self._cat([o[0] for o in outs])
        return (feats, np.concatenate([o[1] for o in outs]), np.concatenate([o[2] for o in outs]),
                [i for o in outs for i in o[3]])

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:  # noqa: BLE001
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
