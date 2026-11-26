"""Saving brain checkpoints (training only: the agent itself never writes files)."""

from __future__ import annotations

from pathlib import Path

import torch

from smriti.brain import SmritiNet, checkpoint_dict


def save_checkpoint(path: str | Path, net: SmritiNet, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_dict(net, extra), tmp)
    tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint
    return path
