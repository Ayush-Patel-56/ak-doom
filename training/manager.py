"""Semi-Markov manager targets from actual discounted extrinsic intervals."""
from __future__ import annotations

import numpy as np
import torch


class ManagerIntervals:
    def __init__(self, n, gamma):
        if n <= 0 or not 0 <= gamma <= 1:
            raise ValueError('n must be positive and gamma in [0, 1]')
        self.n, self.gamma = int(n), float(gamma)
        self.pending = [None] * self.n
        self.completed = []

    def _finish(self, i, bootstrap):
        record = self.pending[i]
        if record is None:
            return
        record['target'] += self.gamma ** record['duration'] * float(bootstrap)
        self.completed.append(record)
        self.pending[i] = None

    def begin(self, aux):
        arrays = {}
        for name in ('hidden', 'boundary', 'goal', 'manager_logp', 'manager_value'):
            value = aux[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            arrays[name] = np.asarray(value)
        h = arrays['hidden']
        if h.ndim != 2 or h.shape[0] != self.n:
            raise ValueError('hidden must have shape (n, hidden_dim)')
        if any(arrays[k].shape != (self.n,) for k in arrays if k != 'hidden'):
            raise ValueError('manager boundary, goal, logp and value must have shape (n,)')
        if arrays['boundary'].dtype != np.bool_ or arrays['goal'].dtype.kind not in 'iu':
            raise ValueError('boundary must be boolean and goal integer')
        if (not all(np.isfinite(v).all() for v in arrays.values())
                or np.any((arrays['goal'] < 0) | (arrays['goal'] >= 5))):
            raise ValueError('manager inputs must be finite with goals in [0, 5)')
        for i in np.flatnonzero(arrays['boundary']):
            self._finish(i, arrays['manager_value'][i])
            self.pending[i] = dict(hidden=h[i].copy(), goal=int(arrays['goal'][i]),
                                   old_logp=float(arrays['manager_logp'][i]),
                                   old_value=float(arrays['manager_value'][i]),
                                   target=0., duration=0)

    def step(self, extrinsic_reward, done):
        reward, done = np.asarray(extrinsic_reward), np.asarray(done)
        if reward.shape != (self.n,) or done.shape != (self.n,) or done.dtype != np.bool_:
            raise ValueError('reward and boolean done must have shape (n,)')
        if not np.isfinite(reward).all():
            raise ValueError('rewards must be finite')
        for i, record in enumerate(self.pending):
            if record is not None:
                record['target'] += self.gamma ** record['duration'] * float(reward[i])
                record['duration'] += 1
                if done[i]:
                    self._finish(i, 0.)

    def drain(self):
        completed, self.completed = self.completed, []
        return completed
