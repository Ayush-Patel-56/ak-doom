# Experiment Log

One line per training run. Newest at the top. Keep it short.

| Run name | Stage | What we changed | Result (Mini-League) | Learned / next |
|----------|-------|-----------------|----------------------|----------------|
| `phase0_odometry` | 0 | Calibrated odometer on `two_rooms.wad` (`configs/odometry.json`) | Heading error < 0.06°, drift 0.3% of 8,364 units | Turn = 7.03° / 12.30° / 14.06° per held step; distance = 2.86·v + 1.56·v_prev; first step after reset is ignored. Re-check on big maps. |
| *example:* `stage1_baseline_1127` | 1 | First PPO + ray/entity/vitals | T1: 0.42 · 0 crashes · 11 ms | Shoots walls a lot → add "shooting at nothing" penalty |
