## Summary

Implements Phase 5 in two parts, with separate focused commits.

### Part 1 — Generalisation and hardening
- Add Stage 5 mixing usable training-split catalog tiers with all five generated map kinds. Expand generated sampling to the complete training seed range while preserving validation/test isolation.
- Add reproducible, versioned randomisation of safe spawns/headings, enemy/item density and composition, textures and lighting. Validate puzzle solvability and preserve skill 3. Record profile hashes, actual episode seeds, map identities and tick caps.
- Detect changed full-level WAD contents on resume, in addition to changed catalog/configuration provenance.
- Add a real-engine stress runner and harden extreme observation conversion against floating-point overflow and integer wraparound.
- Guard project evaluation, legacy/new ablation and benchmark entry points against accidental known test-map use, including aliases and byte-identical copies. Require explicit final-test opt-in and persist reservation/repeat/completion records.
- Add strict matched-tier validation/test reports. Reject incomplete, errored, duplicate or configuration-mismatched comparisons and prevent low scores or missing tiers from qualifying.
- Harden latency measurement: reject missing/untrained checkpoints and random fallback; report requested/effective devices and time only the scored constructor.

### Part 2 — Optional training extensions
- Add frozen recurrent teacher distillation on actual student-visited observations, with independent recurrent state, goal cadence and sampling RNG. Record teacher checkpoint identity and validate it on resume; enable required memory/image modalities for both networks.
- Add action-conditioned prediction of measured legal ray/vitals changes. Mask terminal/reset transitions and keep the auxiliary head unused during inference.
- Preserve old checkpoint defaults and explicit warm-start behavior. Distillation and dynamics losses default off.

## Validation and measured results

- **296 tests passed**, two known macOS-incompatible tests deselected (`deep_json`, `cpu_and_ram_limits_are_emulated`). The final suite ran with local Unix socket access required by scorer/validator tests.
- Real combined training: 512 decisions, then successful resume to 528, from the existing trained champion; frozen teacher uses that same actual trained source. The initial 25 completed episodes covered every generated kind and nine full-level episodes. Training used an explicit 80-tick smoke override.
- Validation-only sample: eight actual episodes, two seeds on one map per tier, 210-tick cap. Zero exits, no errors/stopped runs. No final-test engine evaluation was performed.
- Stress: 20 actual map episodes with 4200-tick caps, 7186 total decisions, no recorded reliability failures. Malformed-input fallback counts are included. RSS growth is unsupported on this Mac and is not marked passed.
- Final candidate mean inference latency: 4.278 ms on two_rooms and 3.891 ms on Freedoom MAP08; below the 20 ms target on this Mac. Measurements are not an official pinned Linux scorer benchmark.
- Final candidate passed the official validator in an isolated snapshot, with warnings for snapshot Git metadata and unavailable Mac CPU/RAM checks.
- Independent review findings have regression coverage for test-map copies and old routes, incomplete reports, strict provenance, episode logging, WAD content changes, benchmark entry/timing/device handling and visible chaos fallbacks.

Raw measurements, checkpoint hashes, training metrics, episode records and limitations are committed in `configs/phase5-smoke.json`. Usage is in `training/README.md`.

## Remaining acceptance criteria

The short smoke verifies execution, not policy superiority or generalisation. Phase 4 exit-rate improvement and Phase 5 final-test score-gap/resource acceptance remain open. The teacher implemented here uses legal observations; a privileged full-map teacher remains a separate extension.

The official `validate.py` is preserved unchanged and does not enforce the project experiment ledger. Direct manual use of that supplied validator on test maps is outside the runner guard. Historical test freshness cannot be guaranteed.

Champion weights remain unchanged. This PR targets the original repository's `main`; the fork only hosts the feature branch.
