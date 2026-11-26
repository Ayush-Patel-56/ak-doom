"""Parallel game runner: N worker processes, each playing its own DoomEnv.

Each worker computes rewards (training/rewards.py) and episode stats (training/scorer.py) itself,
and sends back only the compact network inputs (smriti.features), so traffic between processes is
small. When an episode ends the worker starts the next one at once (new map every few episodes) and
returns the episode summary in `info`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "submission"), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from smriti.features import extract, stack  # noqa: E402

from training.curriculum import STAGES, MapSampler  # noqa: E402
from training.rewards import RewardCalculator, RewardConfig  # noqa: E402
from training.scorer import ScoreTracker  # noqa: E402


def fine_timer() -> None:
    """Windows wakes sleeping threads only every ~15.6 ms by default. While a worker waits for its next
    command the engine's waits snap to that tick, which made lockstep stepping ~10x slower (measured:
    2 workers 100 steps/s in total vs 600 for one). Asking for 1 ms timer resolution fixes it."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)


def _worker(conn, worker_id: int, stage_id: int, seed: int, reward_cfg: RewardConfig, tmp_dir: str,
            episodes_per_map: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    fine_timer()
    from doom_env import make_env

    rng = random.Random(seed)
    sampler = MapSampler(STAGES[stage_id], rng, Path(tmp_dir) / f"w{worker_id}")
    rewards = RewardCalculator(reward_cfg)
    tracker = ScoreTracker()
    env, label, episodes_left = None, "", 0
    ep_return, ep_len = 0.0, 0

    def start_episode():
        nonlocal env, label, episodes_left, ep_return, ep_len
        if env is None or episodes_left <= 0:
            if env is not None:
                env.close()
            wad, map_name, ticks, label = sampler.next()
            env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=False)
            episodes_left = episodes_per_map
        episodes_left -= 1
        obs, _ = env.reset(seed=rng.randrange(2**31))
        state = env.privileged_state()
        rewards.reset(state)
        tracker.reset(state)
        ep_return, ep_len = 0.0, 0
        return extract(obs)

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(start_episode())
            elif cmd == "step":
                action = np.asarray(data, dtype=np.int8)
                obs, _, terminated, truncated, _ = env.step(action)
                state = env.privileged_state()
                r = rewards.step(state, action, obs)
                tracker.update(state)
                ep_return += r
                ep_len += 1
                done = bool(terminated or truncated)
                info = None
                if done:
                    res = tracker.result()
                    info = {
                        "map": label, "return": ep_return, "length": ep_len, "score": res.score,
                        "exit": res.exit, "keys": res.keys, "doors": res.doors + res.locked_doors,
                        "kills": res.kills, "dead": res.dead, "areas": res.areas,
                        "reward_parts": dict(rewards.totals),
                    }
                    feats = start_episode()
                else:
                    feats = extract(obs, action)
                conn.send((feats, r, done, info))
            elif cmd == "set_stage":
                sampler.set_stage(STAGES[data])
                episodes_left = 0  # switch map at the next episode
            elif cmd == "close":
                break
    except Exception:  # noqa: BLE001 - report the worker's error to the main process
        conn.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


class EnvPool:
    def __init__(self, num_envs: int, stage: int = 1, seed: int = 0, reward_cfg: RewardConfig | None = None,
                 episodes_per_map: int = 2, tmp_dir: str | None = None):
        ctx = mp.get_context("spawn")
        self.tmp = tempfile.TemporaryDirectory(prefix="ak-doom-maps-") if tmp_dir is None else None
        root = tmp_dir or self.tmp.name
        self.conns, self.procs = [], []
        for i in range(num_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, i, stage, seed * 1000 + i, reward_cfg or RewardConfig(),
                                                  root, episodes_per_map), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.num_envs = num_envs

    def _recv(self, conn):
        msg = conn.recv()
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "error":
            raise RuntimeError(f"env worker failed:\n{msg[1]}")
        return msg

    def reset(self) -> dict[str, np.ndarray]:
        for c in self.conns:
            c.send(("reset", None))
        return stack([self._recv(c) for c in self.conns])

    def step(self, actions: np.ndarray):
        for c, a in zip(self.conns, actions):
            c.send(("step", a))
        results = [self._recv(c) for c in self.conns]
        feats = stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], np.float32)
        dones = np.array([r[2] for r in results], bool)
        infos = [r[3] for r in results]
        return feats, rewards, dones, infos

    def set_stage(self, stage: int) -> None:
        for c in self.conns:
            c.send(("set_stage", stage))

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        if self.tmp is not None:
            self.tmp.cleanup()
