"""Action groups: the policy picks 6 small choices, which become the official 9 buttons.

Picking per group (instead of 9 independent on/off buttons) makes silly combinations impossible,
such as forward + backward or turn left + turn right at the same time (docs/03-SYSTEM-DESIGN.md §3.4).

Buttons (docs/CONTRACT.md §3): 0 MOVE_FORWARD, 1 MOVE_BACKWARD, 2 MOVE_LEFT, 3 MOVE_RIGHT,
4 TURN_LEFT, 5 TURN_RIGHT, 6 ATTACK, 7 USE, 8 SELECT_NEXT_WEAPON.
"""

from __future__ import annotations

import numpy as np

# (group name, buttons pressed by each choice). Choice 0 is always "nothing".
GROUPS: tuple[tuple[str, tuple[tuple[int, ...], ...]], ...] = (
    ("move", ((), (0,), (1,))),  # none / forward / backward
    ("strafe", ((), (2,), (3,))),  # none / left / right
    ("turn", ((), (4,), (5,))),  # none / left / right
    ("attack", ((), (6,))),
    ("use", ((), (7,))),
    ("next_weapon", ((), (8,))),
)
GROUP_SIZES: tuple[int, ...] = tuple(len(choices) for _, choices in GROUPS)
NUM_GROUPS = len(GROUPS)
NUM_LOGITS = sum(GROUP_SIZES)
NUM_BUTTONS = 9

# Lookup table: for group g and choice c, which buttons are pressed. Shape (NUM_GROUPS, max_choices, 9).
_TABLE = np.zeros((NUM_GROUPS, max(GROUP_SIZES), NUM_BUTTONS), dtype=np.int8)
for g, (_, choices) in enumerate(GROUPS):
    for c, buttons in enumerate(choices):
        _TABLE[g, c, list(buttons)] = 1


def groups_to_buttons(choices: np.ndarray) -> np.ndarray:
    """(..., 6) integer choices -> (..., 9) int8 buttons."""
    choices = np.asarray(choices, dtype=np.int64)
    out = np.zeros(choices.shape[:-1] + (NUM_BUTTONS,), dtype=np.int8)
    for g in range(NUM_GROUPS):
        out |= _TABLE[g][choices[..., g]]
    return out


def buttons_to_groups(buttons: np.ndarray) -> np.ndarray:
    """(..., 9) buttons -> (..., 6) choices. Contradictory pairs (e.g. forward + backward) become 'none'."""
    b = np.asarray(buttons).astype(bool)
    out = np.zeros(b.shape[:-1] + (NUM_GROUPS,), dtype=np.int64)
    for g, (_, choices) in enumerate(GROUPS):
        for c, pressed in enumerate(choices):
            if not pressed:
                continue
            others = [i for cc, p in enumerate(choices) if cc != c for i in p]
            hit = np.all(b[..., list(pressed)], axis=-1)
            if others:
                hit &= ~np.any(b[..., others], axis=-1)
            out[..., g] = np.where(hit, c, out[..., g])
    return out
