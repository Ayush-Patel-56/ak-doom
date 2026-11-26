# Experiment Log

One line per training run. Newest at the top. Keep it short.

| Run name | Stage | What we changed | Result (Mini-League) | Learned / next |
|----------|-------|-----------------|----------------------|----------------|
| `phase0_odometry` | 0 | Calibrated odometer on `two_rooms.wad` (`configs/odometry.json`) | Heading error < 0.06°, drift 0.3% of 8,364 units | Turn = 7.03125° / 12.3046875° / 14.0625° (160 / 280 / 320 fine angles); distance = 2.86·v + 1.56·v_prev; first step after reset is a no-op, and the steady ramp needs ≥2 no-ops. Re-check on bigger maps. |
| *example:* `stage1_baseline_1127` | 1 | First PPO + ray/entity/vitals | T1: 0.42 · 0 crashes · 11 ms | Shoots walls a lot → add "shooting at nothing" penalty |
