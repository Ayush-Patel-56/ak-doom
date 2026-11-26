"""Warm-start a memory brain from a trained brain without memory (e.g. the Phase 2 champion).

New encoder inputs are transferred by named blocks in two layers:
- fuse:  Linear([rays, entities, vitals | mind map])          -> the mind-map columns are new
- gru:   GRUCell([fused input | notebook read], hidden state)  -> the notebook columns are new

We copy shared weights and set NEW encoder input columns to zero. RGB columns are
matched by name when a mind map is inserted before them. At step 0
the memory brain then plays exactly like the source brain (memory is wired in but has no effect yet),
and training only has to learn how to USE memory instead of re-learning to move, fight and explore.
The new encoders and hierarchy modules keep their normal random initialisation;
adding a manager changes the worker architecture, so exact action preservation is
guaranteed only when the hierarchy configuration is unchanged. For encoder additions, the gradient
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
    for flag in ('use_mindmap', 'use_notebook', 'use_rgb', 'use_depth', 'use_manager'):
        if getattr(source.cfg, flag) and not getattr(target.cfg, flag):
            raise ValueError(f'warm start cannot remove {flag}')

    def fuse_blocks(cfg):
        blocks = [('rays', cfg.ray_out), ('entities', cfg.entity_dim), ('vitals', cfg.vitals_out)]
        if cfg.use_mindmap:
            blocks.append(('map', cfg.map_out))
        if cfg.use_rgb:
            blocks.append(('rgb', cfg.rgb_out))
        if cfg.use_depth:
            blocks.append(('depth', cfg.depth_out))
        result, offset = {}, 0
        for name, width in blocks:
            result[name] = (offset, width)
            offset += width
        return result

    old_blocks, new_blocks = fuse_blocks(source.cfg), fuse_blocks(target.cfg)
    if any(name not in new_blocks or new_blocks[name][1] != width
           for name, (_, width) in old_blocks.items()):
        raise ValueError('warm start needs matching existing encoder widths')
    src = source.state_dict()
    report: dict[str, list[str]] = {"copied": [], "widened": [], "new": []}
    for name, param in target.state_dict().items():
        if name not in src:
            report["new"].append(name)
            continue
        old = src[name]
        if name == 'fuse.0.weight' and old_blocks != new_blocks:
            # RGB follows the map input. Adding a map therefore inserts columns
            # before RGB; a plain prefix copy would silently attach RGB weights to map features.
            param.zero_()
            for block, (start, width) in old_blocks.items():
                destination = new_blocks[block][0]
                param[:, destination:destination + width] = old[:, start:start + width]
            report['widened'].append(name)
        elif old.shape == param.shape:
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
