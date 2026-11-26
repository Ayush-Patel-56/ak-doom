"""Print one full observation from the official wrapper in a readable form.

Phase 0 tool: shows exactly what the agent receives, so encoder design matches reality.

    uv run python scripts/inspect_observation.py
    uv run python scripts/inspect_observation.py --map maps/two_rooms.wad:MAP01 --steps 20 --save runs/obs.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from doom_env import make_env
from doom_env import spaces as S

from map_spec import parse_map

REPO = Path(__file__).resolve().parent.parent
HIT_SYMBOL = {0: ".", 1: "#", 2: "D", 3: "L", 4: "E", 5: "i"}  # none, wall, door, locked, enemy, item


def describe(obs: dict) -> str:
    lines = ["=== Observation keys, shapes and dtypes ==="]

    def walk(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
            else:
                arr = np.asarray(v)
                lines.append(f"  {prefix + k:<34} shape={str(arr.shape):<12} dtype={arr.dtype}")

    walk(obs)

    lines.append("\n=== vitals ===")
    for name, value in zip(S.VITALS_FIELDS, obs["vitals"]):
        lines.append(f"  {name:<22} {value:9.3f}")

    rays = obs["rays"]
    lines.append("\n=== rays (index 0 = ahead, counter-clockwise; . none  # wall  D door  L locked  E enemy  i item) ===")
    ring = "".join(HIT_SYMBOL[int(h)] for h in rays["hit_type"])
    lines.append(f"  hit ring : {ring}")
    lines.append("  markers  : ^ahead          <left           v behind         >right")
    dist = rays["distance"] * S.RAY_RANGE
    for label, idx in (("ahead", 0), ("left", 16), ("behind", 32), ("right", 48)):
        lines.append(
            f"  {label:<7} ray {idx:>2}: {S.RAY_HIT_TYPES[int(rays['hit_type'][idx])]:<12} {dist[idx]:7.1f} units"
        )

    ents = obs["visible_entities"]
    n = int(np.sum(ents["mask"]))
    lines.append(f"\n=== visible_entities ({n} of {S.MAX_ENTITIES} slots used, 90 deg field of view) ===")
    for i in range(n):
        f = ents["features"][i]
        angle = np.degrees(np.arctan2(f[1], f[2]))  # display only: never used for decisions
        lines.append(
            f"  {S.ENTITY_TYPES[int(ents['entity_type'][i])]:<17} dist {f[0] * 1024:7.1f}  "
            f"angle {angle:+6.1f} deg  rel_vel ({f[3]:+.2f}, {f[4]:+.2f})"
        )

    lines.append("\n=== images ===")
    for key in ("rgb", "depth"):
        arr = np.asarray(obs[key])
        lines.append(f"  {key:<6} min {arr.min():3d}  max {arr.max():3d}  mean {arr.mean():6.1f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=str(REPO / "maps" / "two_rooms.wad") + ":MAP01", help="wad[:MAPxx]")
    ap.add_argument("--steps", type=int, default=10, help="forward steps to take before printing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=Path, help="optional .npz path to save the raw observation")
    args = ap.parse_args()

    wad, map_name = parse_map(args.map)
    env = make_env(wad, map_name)
    try:
        obs, _ = env.reset(seed=args.seed)
        forward = np.zeros(len(S.BUTTONS), dtype=np.int8)
        forward[S.BUTTONS.index("MOVE_FORWARD")] = 1
        for _ in range(args.steps):
            obs, _, terminated, truncated, _ = env.step(forward)
            if terminated or truncated:
                break
        print(f"Map {wad}:{map_name} after {args.steps} forward steps\n")
        print(describe(obs))
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            flat = {}
            for k, v in obs.items():
                if isinstance(v, dict):
                    flat.update({f"{k}.{kk}": np.asarray(vv) for kk, vv in v.items()})
                else:
                    flat[k] = np.asarray(v)
            np.savez_compressed(args.save, **flat)
            print(f"\nSaved raw observation to {args.save}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
