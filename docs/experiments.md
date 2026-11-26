# Experiment Log

One line per training run. Newest at the top. Keep it short.

| Run name | Stage | What we changed | Result (Mini-League) | Learned / next |
|----------|-------|-----------------|----------------------|----------------|
| `baseline-random-val-v2` | 2 | Random agent on the **new 20-map val** (4 Freedoom + 16 generated: 4 arena, 4 maze, 4 rooms, 4 keydoor), 2 seeds | **Weighted total −312.37** (T1 −29.8, T2 −140.2, T3 −136.2, T4 −6.2); 0 crashes | Dies on most generated maps (monsters at skill 3); 0 exits, 0 keys, 1 door. Replaces 4.28 as the number to beat (different map set). |
| `phase1_docker_validate` | 1 | `validate.py --docker --image devforge-doom` on `main` (`3c318e4`) | **PASSED**: pinned to 1 CPU, init 0.1 s on `cuda`, p50 0.4 ms / max 1 ms per step, peak RAM 44 MB | Only warning: GPU memory can't be measured on Windows. Re-run after every brain change. |
| `baseline-random-val` | 1 | Template random agent behind the safety guard; Mini-League on `val` (4 Freedoom 2 maps, 2 seeds) | **Weighted total 4.28** (T3 10.5, T4 −6.22); 0 crashes; ~0.1 ms/decision | Never opens doors or finds keys; explores 5–10 areas; dies on MAP26. This is the score Phase 2 must beat. |
| `phase0_odometry` | 0 | Calibrated odometer on `two_rooms.wad` (`configs/odometry.json`) | Heading error < 0.06°, drift 0.3% of 8,364 units | Turn = 7.03125° / 12.3046875° / 14.0625° (160 / 280 / 320 fine angles); distance = 2.86·v + 1.56·v_prev; first step after reset is a no-op, and the steady ramp needs ≥2 no-ops. Re-check on bigger maps. |
| *example:* `stage1_baseline_1127` | 1 | First PPO + ray/entity/vitals | T1: 0.42 · 0 crashes · 11 ms | Shoots walls a lot → add "shooting at nothing" penalty |
