"""Training-only random network distillation over legal flat observation features.

Call bonus before predictor updates. Its normalization uses only preceding calls;
update_stats=False is a read-only evaluation. Episode/reset masking belongs to the
caller, which must never score an auto-reset replacement as a terminal observation.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn


class _Moments:
    def __init__(self, shape):
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    @property
    def variance(self):
        return self.m2 / self.count if self.count > 1 else np.ones_like(self.mean)

    def update(self, values):
        n = len(values)
        if not n:
            return
        mean = values.mean(axis=0, dtype=np.float64)
        delta = mean - self.mean
        total = self.count + n
        self.m2 += ((values - mean) ** 2).sum(axis=0) + delta ** 2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def state_dict(self):
        return dict(count=self.count, mean=self.mean.copy(), m2=self.m2.copy())

    def load_state_dict(self, state):
        self.count = int(state['count'])
        self.mean = np.array(state['mean'], dtype=np.float64, copy=True)
        self.m2 = np.array(state['m2'], dtype=np.float64, copy=True)


class RND:
    """Frozen target and Adam-trained predictor; rewards are nonnegative and capped.

    Features have shape (batch, input_dim). Statistics are updated only by bonus,
    after computing errors and rewards, so no future samples affect normalization.
    update does not change normalization. Reward weighting/decay is the trainer's
    responsibility; clip is the maximum *unweighted* intrinsic reward.
    """

    def __init__(self, input_dim, device='cpu', seed=0, lr=1e-3, clip=5.0):
        if input_dim <= 0 or lr <= 0 or not np.isfinite(clip) or clip <= 0:
            raise ValueError('input_dim, lr and clip must be positive')
        self.input_dim = int(input_dim)
        self.device = torch.device(device)
        self.clip = float(clip)
        # Initialization is CPU-local and restores the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.target = self._network().to(self.device)
            self.predictor = self._network().to(self.device)
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.observation_moments = _Moments((self.input_dim,))
        self.reward_moments = _Moments(())

    def _network(self):
        return nn.Sequential(nn.Linear(self.input_dim, 64), nn.ReLU(), nn.Linear(64, 32))

    def _features(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all():
            raise ValueError(f'features must be finite (batch, {self.input_dim}) array')
        return x

    def _normalized(self, features):
        moments = self.observation_moments
        x = (features - moments.mean) / np.sqrt(np.maximum(moments.variance, 1e-4))
        return torch.as_tensor(np.clip(x, -5, 5), dtype=torch.float32, device=self.device)

    def _errors(self, features):
        x = self._normalized(features)
        with torch.no_grad():
            target = self.target(x)
        return (self.predictor(x) - target).square().mean(dim=-1)

    def bonus(self, features, update_stats=True):
        x = self._features(features)
        if not len(x):
            return np.empty(0, dtype=np.float32)
        with torch.no_grad():
            errors = self._errors(x).cpu().numpy()
        scale = np.sqrt(np.maximum(self.reward_moments.variance, 1e-4))
        rewards = np.clip(errors / scale, 0, self.clip).astype(np.float32)
        if update_stats:
            self.observation_moments.update(x)
            self.reward_moments.update(errors)
        return rewards

    def update(self, features):
        x = self._features(features)
        if not len(x):
            return 0.0
        loss = self._errors(x).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def state_dict(self):
        return deepcopy(dict(input_dim=self.input_dim, clip=self.clip,
                             target=self.target.state_dict(), predictor=self.predictor.state_dict(),
                             optimizer=self.optimizer.state_dict(),
                             observation_moments=self.observation_moments.state_dict(),
                             reward_moments=self.reward_moments.state_dict()))

    def load_state_dict(self, state):
        if state['input_dim'] != self.input_dim:
            raise ValueError('RND checkpoint input dimension does not match')
        self.target.load_state_dict(state['target'])
        self.predictor.load_state_dict(state['predictor'])
        self.optimizer.load_state_dict(deepcopy(state['optimizer']))
        self.clip = float(state['clip'])
        self.observation_moments.load_state_dict(state['observation_moments'])
        self.reward_moments.load_state_dict(state['reward_moments'])
