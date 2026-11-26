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


def _paths():
    for p in (str(REPO / "submission"), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)


class Game:
    """One doom_env game + its memory, reward shaper and episode bookkeeping."""

    def __init__(self, sampler, use_mindmap: bool, use_notebook: bool, episode_ticks: int, switch_every: int, rng):
        from smriti.memory import EpisodeMemory, Odometer

        self.sampler, self.ticks, self.switch_every, self.rng = sampler, episode_ticks, switch_every, rng
        self.memory = EpisodeMemory(Odometer(), use_mindmap=use_mindmap, use_notebook=use_notebook)
        self.uses_memory = use_mindmap or use_notebook
        self.env, self.episodes_on_map = None, 0
        from training.rewards import RewardShaper

        self.shaper = RewardShaper()

    def _features(self, obs):
        from smriti.features import extract

        mem = self.memory.features() if self.uses_memory else None
        return extract(obs, self.prev_action, mem)

    def start(self):
        from doom_env import make_env

        if self.env is None or self.episodes_on_map >= self.switch_every:
            if self.env is not None:
                self.env.close()
            self.wad, self.map, self.kind = next(self.sampler)
            self.env = make_env(self.wad, self.map, episode_ticks=self.ticks, render_buffers=False)
            self.episodes_on_map = 0
        obs, _ = self.env.reset(seed=int(self.rng.integers(2**31 - 1)))
        self.memory.reset()
        self.memory.observe(None, obs, obs)  # the reset observation is not a transition
        self.shaper.reset(self.env.privileged_state())
        self.prev_action = np.zeros(9, np.float32)
        self.ret, self.len = 0.0, 0
        return self._features(obs)

    def step(self, buttons: np.ndarray):
        obs, _, term, trunc, _ = self.env.step(buttons)
        r = self.shaper(self.env.privileged_state(), buttons)
        self.ret += r
        self.len += 1
        self.prev_action = buttons.astype(np.float32)
        done = bool(term or trunc)
        info = None
        if done:
            info = {"map": Path(self.wad).name, "kind": self.kind, "ret": self.ret, "len": self.len,
                    **self.shaper.stats()}
            self.episodes_on_map += 1
            feats = self.start()
        else:
            self.memory.observe(buttons, obs, obs)
            feats = self._features(obs)
        return feats, r, done, info

    def close(self):
        if self.env is not None:
            self.env.close()


def _worker(remote, wid, n_games, stage, seed, use_mindmap, use_notebook, episode_ticks, switch_every, kinds):
    _paths()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    games = []
    try:
        from smriti.features import stack
        from smriti.policy.actions import groups_to_buttons
        from training.curriculum import MapSampler

        sampler = MapSampler(stage, seed * 1009 + wid, kinds=kinds)
        rng = np.random.default_rng(seed * 7919 + wid)
        games = [Game(sampler, use_mindmap, use_notebook, episode_ticks, switch_every, rng) for _ in range(n_games)]
        remote.send(("ok", stack([g.start() for g in games])))
        while True:
            cmd, data = remote.recv()
            if cmd == "close":
                break
            choices, progress = data
            sampler.progress = progress
            buttons = groups_to_buttons(choices)
            out = [g.step(buttons[i]) for i, g in enumerate(games)]
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
                 use_notebook: bool, episode_ticks: int = 2100, switch_every: int = 2, kinds=None):
        ctx = mp.get_context("spawn")  # same on Linux, macOS and Windows
        self.k, self.n = games_per_worker, workers * games_per_worker
        self.remotes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(b, w, games_per_worker, stage, seed, use_mindmap, use_notebook, episode_ticks,
                                  switch_every, kinds))
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

    def step(self, choices: np.ndarray, progress: float):
        for i, r in enumerate(self.remotes):
            r.send(("step", (choices[i * self.k:(i + 1) * self.k], progress)))
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
