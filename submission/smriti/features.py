"""Observation -> network input. Used by BOTH the agent and training, so they can never disagree.

Phase 2 inputs (docs/03-SYSTEM-DESIGN.md §3.1): rays, visible entities, vitals and the previous
action. Depth and RGB come later. Scaling to network-friendly ranges happens inside the network
(smriti/brain.py) so this step stays a cheap, lossless repack into numpy arrays.
"""

from __future__ import annotations

import numpy as np

NUM_RAYS = 64
NUM_HIT_TYPES = 6
MAX_ENTITIES = 32
ENTITY_FEATURES = 5
NUM_ENTITY_TYPES = 12
NUM_VITALS = 17
NUM_BUTTONS = 9

# name -> (shape without batch dims, dtype)
FEATURE_SPEC: dict[str, tuple[tuple[int, ...], type]] = {
    "ray_dist": ((NUM_RAYS,), np.float32),
    "ray_hit": ((NUM_RAYS,), np.int64),
    "ent_feat": ((MAX_ENTITIES, ENTITY_FEATURES), np.float32),
    "ent_type": ((MAX_ENTITIES,), np.int64),
    "ent_mask": ((MAX_ENTITIES,), np.bool_),
    "vitals": ((NUM_VITALS,), np.float32),
    "prev_action": ((NUM_BUTTONS,), np.float32),
}


def extract(obs: dict, prev_action: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """One observation (already cleaned by SafetyGuard) -> flat dict of arrays, no batch dim."""
    rays, ents = obs["rays"], obs["visible_entities"]
    return {
        "ray_dist": np.asarray(rays["distance"], np.float32),
        "ray_hit": np.clip(np.asarray(rays["hit_type"], np.int64), 0, NUM_HIT_TYPES - 1),
        "ent_feat": np.asarray(ents["features"], np.float32),
        "ent_type": np.clip(np.asarray(ents["entity_type"], np.int64), 0, NUM_ENTITY_TYPES - 1),
        "ent_mask": np.asarray(ents["mask"]).astype(np.bool_),
        "vitals": np.asarray(obs["vitals"], np.float32),
        "prev_action": (np.zeros(NUM_BUTTONS, np.float32) if prev_action is None
                        else np.asarray(prev_action, np.float32)),
    }


def stack(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """List of extract() outputs -> dict of arrays with a leading batch dimension."""
    return {k: np.stack([it[k] for it in items]) for k in FEATURE_SPEC}


def empty(batch_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Zero-filled feature arrays with the given leading dims (used for rollout buffers)."""
    return {k: np.zeros(batch_shape + shape, dtype) for k, (shape, dtype) in FEATURE_SPEC.items()}
