# Experiment Log

One line per training run. Newest at the top. Keep it short.

| Run name | Stage | What we changed | Result (Mini-League) | Learned / next |
|----------|-------|-----------------|----------------------|----------------|
| `baseline-random-val-v3` | 2 | Random agent on the **regenerated** 20-map val (Kunal's mapgen fix: every layout solvable, one start + one exit, monsters placed through doors), 2 seeds | **Weighted total −201.16** (T1 −29.8, T2 −36.2, T3 −128.9, T4 −6.2); 0 crashes | 12 of 16 generated val maps changed, so this **replaces −312.37** as the number to beat. Still 0 exits and 0 keys. |
| `baseline-random-val-v2` | 2 | Random agent on the **new 20-map val** (4 Freedoom + 16 generated: 4 arena, 4 maze, 4 rooms, 4 keydoor), 2 seeds | **Weighted total −312.37** (T1 −29.8, T2 −140.2, T3 −136.2, T4 −6.2); 0 crashes | Dies on most generated maps (monsters at skill 3); 0 exits, 0 keys, 1 door. Replaced 4.28; superseded by v3 after the generator fix. |
| `phase2a_brain_docker` | 2 | Untrained SmritiNet (557,632 params) as temporary champion, `validate.py --docker` | PASSED: init 19.3 s, p50 2.2 ms / max 8 ms per step, 965 MB RAM | First forward pass took 5 s before the `__init__` warm-up fix. Brain is fast enough. |
| `phase1_docker_validate` | 1 | `validate.py --docker --image devforge-doom` on `main` (`3c318e4`) | **PASSED**: pinned to 1 CPU, init 0.1 s on `cuda`, p50 0.4 ms / max 1 ms per step, peak RAM 44 MB | Only warning: GPU memory can't be measured on Windows. Re-run after every brain change. |
| `baseline-random-val` | 1 | Template random agent behind the safety guard; Mini-League on `val` (4 Freedoom 2 maps, 2 seeds) | **Weighted total 4.28** (T3 10.5, T4 −6.22); 0 crashes; ~0.1 ms/decision | Never opens doors or finds keys; explores 5–10 areas; dies on MAP26. This is the score Phase 2 must beat. |
| `phase0_odometry` | 0 | Calibrated odometer on `two_rooms.wad` (`configs/odometry.json`) | Heading error < 0.06°, drift 0.3% of 8,364 units | Turn = 7.03125° / 12.3046875° / 14.0625° (160 / 280 / 320 fine angles); distance = 2.86·v + 1.56·v_prev; first step after reset is a no-op, and the steady ramp needs ≥2 no-ops. Re-check on bigger maps. |
| *example:* `stage1_baseline_1127` | 1 | First PPO + ray/entity/vitals | T1: 0.42 · 0 crashes · 11 ms | Shoots walls a lot → add "shooting at nothing" penalty |
