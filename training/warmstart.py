"""Warm-start a memory brain from a trained brain without memory (e.g. the Phase 2 champion).

The memory brain only ADDS inputs at the end of two layers:
- fuse:  Linear([rays, entities, vitals | mind map])          -> the mind-map columns are new
- gru:   GRUCell([fused input | notebook read], hidden state)  -> the notebook columns are new

So we copy every weight that exists in both brains and set the NEW input columns to zero. At step 0
the memory brain then plays exactly like the source brain (memory is wired in but has no effect yet),
and training only has to learn how to USE memory instead of re-learning to move, fight and explore.
The new encoders (MapEncoder, NotebookReader) keep their normal random initialisation; the gradient
reaches them as soon as the zero columns start to move.
"""

from __future__ import annotations

import torch

from smriti.brain import SmritiNet


@torch.no_grad()
def warm_start_from(target: SmritiNet, source: SmritiNet) -> dict[str, list[str]]:
    """Copy `source` weights into `target` (in place). Returns which parameters were copied / widened / new."""
    if target.cfg.hidden != source.cfg.hidden:
        raise ValueError("warm start needs the same hidden size")
    src = source.state_dict()
    report: dict[str, list[str]] = {"copied": [], "widened": [], "new": []}
    for name, param in target.state_dict().items():
        if name not in src:
            report["new"].append(name)
            continue
        old = src[name]
        if old.shape == param.shape:
            param.copy_(old)
            report["copied"].append(name)
        elif name in ("fuse.0.weight", "gru.weight_ih") and old.shape[0] == param.shape[0] \
                and old.shape[1] < param.shape[1]:
            param.zero_()
            param[:, : old.shape[1]] = old  # old inputs come first; extra (memory) columns stay zero
            report["widened"].append(name)
        else:
            raise ValueError(f"{name}: cannot warm-start {tuple(old.shape)} -> {tuple(param.shape)}")
    return report
