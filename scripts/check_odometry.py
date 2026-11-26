"""Compare shipped odometry with real training ground truth, without refitting.

Example: uv run python scripts/check_odometry.py --map maps/two_rooms.wad --steps 600
Exit 1 when a configured drift budget is exceeded; JSON contains the actual measurements.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from measure_odometry import Calibration, Odometer, REPO, make_env, measure_drift
from map_spec import parse_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', action='append', dest='maps', help='repeatable wad[:MAPxx]')
    parser.add_argument('--calibration', type=Path, help='default: packaged submission calibration')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=733)
    parser.add_argument('--max-heading-error-deg', type=float, default=.06)
    parser.add_argument('--max-position-drift-fraction', type=float,
                        help='optional maximum trajectory-wide position error / distance travelled')
    parser.add_argument('--out', type=Path, help='write measured JSON report')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    for limit in (args.max_heading_error_deg, args.max_position_drift_fraction):
        if limit is not None and (not math.isfinite(limit) or limit < 0):
            parser.error('error budgets must be finite and nonnegative')
    calibration = Calibration.load(args.calibration) if args.calibration else Odometer().calibration
    episodes = []
    for spec in args.maps or [str(REPO / 'maps/two_rooms.wad')]:
        wad, map_name = parse_map(spec)
        env = make_env(wad, map_name, render_buffers=False, episode_ticks=(args.steps + 20) * 4)
        try:
            result = measure_drift(env, args.seed, args.steps, calibration)
        finally:
            env.close()
        distance = result['distance_travelled_units']
        drift = result['max_position_error_units']
        fraction = drift / distance if distance > 0 else (0.0 if drift == 0 else None)
        passed = result['max_heading_error_deg'] <= args.max_heading_error_deg
        if args.max_position_drift_fraction is not None:
            passed = passed and fraction is not None and fraction <= args.max_position_drift_fraction
        episodes.append({'map': spec, **result, 'max_position_drift_fraction': fraction, 'passed': passed})
    report = {'passed': all(e['passed'] for e in episodes),
              'max_heading_error_deg': args.max_heading_error_deg,
              'max_position_drift_fraction': args.max_position_drift_fraction,
              'episodes': episodes}
    output = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding='utf-8')
    print(output, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
