# Episode-relative odometer

## Intent

Implement the odometer in System Design §3.3.1 as the first Phase 3 memory component. Estimate position and heading relative to spawn from legal observations and the actions actually sent to the engine. Compare estimates with training-only ground truth using real ViZDoom episodes. Deliver a feature-branch PR into Kunal628-hue/ak-doom:main; do not update main directly.

## Approach

Use the existing measured dead-reckoning model: rotate observed forward/right velocity into the spawn frame and integrate the calibrated current/previous velocity combination. Track turning from the actual turn buttons, including simultaneous buttons, direction changes, release, and engine startup. This follows the repository's existing design and requires no neural-network checkpoint changes.

Alternatives are ray matching or a learned motion estimator. Both add substantial scope and require their own training or matching validation; they remain deferred unless real-engine drift measurements show that the existing design is insufficient.

## Runtime boundary

- Add a NumPy-only odometer under submission/smriti/memory with explicit reset and transition-update methods, plus an immutable pose result.
- Calibration must be generated from real engine measurements, validated for the pinned engine/frame skip, and available inside submission. No fabricated fallback coefficients or fitted evaluation ground truth.
- Estimate the first transition immediately after reset, without inserting scripted startup actions or assuming the calibration warmup occurred.
- Update from the preceding final action returned by SafetyGuard and the observation resulting from that action. The initial observation is not a completed transition. Reset clears position, heading, velocity history, and turn history.
- Keep the existing policy, action selection, checkpoint format, official wrapper, and validator unchanged. Expose pose for subsequent memory components without adding it to the current network input.
- Invalid calibration and malformed/nonfinite transition inputs must produce explicit errors rather than silently inventing movement. Cover their interaction with the agent's safety boundary.

## Calibration and training validation

Extend/refactor scripts/measure_odometry.py so calibration and drift measurement exercise the production estimator. Measure startup turning instead of imposing no-op actions on the agent. Keep privileged_state() entirely in training, calibration, and integration tests.

Validation compares estimated pose with true engine pose transformed into the spawn frame. Cover forward/backward movement, both strafes, movement during turning, held/reversed/released/opposing turns, heading wraparound, wall contact, and episode resets. Use fresh deterministic engine trajectories and multiple maps, including a nonzero spawn angle. Separate fitting trajectories from validation trajectories. Record actual position/heading errors and distance travelled; do not use an example trajectory's historical errors as test results.

Tests must catch wrong rotation signs, transition alignment, startup assumptions, stale episode state, wrong action bookkeeping, invalid calibration, and dependency on privileged runtime inputs. Exact heading and bounded position drift are separate checks: collision-related dead-reckoning drift is expected and must be reported honestly.

## Scope and delivery

No mind map, landmark memory, training-policy changes, or checkpoint migration in this PR. Preserve the existing local setup notes and .dockerignore outside the feature commits. Run focused tests, available regression tests, the official validator, and a latency check. Report platform-dependent exclusions explicitly. Push only the feature branch to the fork and create a PR with real validation evidence; do not merge it.
