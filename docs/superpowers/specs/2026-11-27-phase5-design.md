# Phase 5: generalisation and hardening

## Intent and delivery

Build two parts on a feature branch, with multiple focused commits and one PR targeting
Ayush-Patel-56/ak-doom:main. Never push main. Preserve existing local docs/06 edits and .dockerignore.
Part 1 is core generalisation/hardening; Part 2 adds optional teacher–student and auxiliary training.
The user selected this two-part approach. Detailed engineering choices below are proposed by the builder.

## Existing system and constraints

Stage 4 has catalog-validated train-only Freedoom2 maps and generated replay, saved curriculum provenance,
legacy checkpoint compatibility, real RGB/depth buffers and a strict comparison runner. Latest upstream
includes PR14 and the export same-map-set fix. Python/PyTorch/ViZDoom remain the stack; no new runtime
service, frontend or Docker service is required. Existing runner and official agent interfaces stay intact.
No fake metrics, fabricated labels, hard-coded action rules or smoke champion promotion. Fixed numerical
hyperparameters are explicit configuration, recorded in manifests, and are not presented as measured data.

## Approaches

1. Extend the existing generators, recurrent PPO and evaluation tools in two parts (selected delivery;
recommended implementation). Reuses established inference and checkpoint contracts; more integration
work than standalone scripts, but features participate in actual training.
2. Add an independent training framework. Easier experimentation in isolation, but duplicates recurrent
state, memory and checkpoint handling and weakens reproducibility; rejected.
3. Core only, defer stretch goals. Smaller initial scope, but leaves requested stretch features undelivered;
the user chose two parts instead.

## Part 1: mixed curriculum and hardening

Add Stage 5 using all usable catalog training tiers plus generated arena, maze, rooms and key puzzles.
Default full-level share 0.5; generated share uniform across five kinds. Keep Stage 1–4 behavior unchanged.
Use the complete training seed range [0,100000) for Stage 5 instead of a fixed 200-map subset; seed namespaces
for held-out generated maps remain excluded. Save configuration, generator version and effective pool in
curriculum/checkpoint provenance. Resume must reject incompatible Stage 5 provenance.

A versioned randomisation profile controls seeded spawn relocation/heading, monster composition and density,
item density, textures and lighting. Generated levels retain validated door/key reachability. Choose alternate
starts only where key-and-exit solvability remains true; fail bounded generation rather than silently issuing
an invalid map. Preserve 64-unit flat geometry and locked-door identities so existing label assumptions remain
valid. Asset choices must be verified against installed Freedoom resources. Do not mutate catalog WADs.
Stage 5 uses skill 3 throughout, retaining official difficulty while varying content. Generated episode caps
vary deterministically within 2100–4200; catalog levels retain catalog caps. Cache names include profile hash,
generator version and seed to prevent stale-map reuse. Record per-episode map/profile/seed/cap identity.

Add stress/chaos coverage for malformed/missing/NaN/extreme observations, repeated reset, cross-episode
memory clearing, long inference, dense enemies, empty maps and constrained spawns. A separate real-engine
runner supports the full 20-map × 4200-tick scenario, records RSS/time/fallbacks/errors and valid actions,
and returns nonzero for declared acceptance failures. Behavioral checks such as escaping a closet or
consistent scores across seeds are measured policy outcomes, not assertions that an untrained policy passes.
Report these failures honestly. Resource limits are explicit configurable acceptance criteria, not arbitrary
post-hoc thresholds. Distinguish platform-unsupported measurements from passes.

Benchmark representative RGB/depth/memory/manager configurations with strict checkpoint loading and real
buffers. Measure before optimizing. Only retain an inference optimization if numerical/action equivalence
and latency improve; TorchScript is an option, not a compulsory change. Do not skip trained encoders at runtime.
The target is <20 ms mean on the evaluation CPU; Mac results are platform-specific evidence.

Generalisation tooling compares identical metric definitions and matched tier composition across validation
and test, includes raw scores/exits/seed variation/sample counts and a declared gap threshold. It rejects
checkpoint/config drift and empty/stopped results. Development/smokes use train and validation only. Test
execution is explicitly opted into with a persisted checkpoint hash and test-use ledger. A shared
map-identity guard covers the new command, existing evaluate.py and phase4_ablation.py test splits,
and custom map arguments resolving to known catalog test identities (including aliases). The guard
records use before engine execution, distinguishes failed/partial runs, and requires an explicit
repeat-use override when the ledger already contains a reservation for that checkpoint/test pool;
repeated use is disclosed, never described as first use. Existing Phase 4 scripts exposed test evaluation,
so historical test freshness cannot be guaranteed. Do not label Phase 5 scientifically complete from low
scores, small samples, unavailable official timing or a small gap alone. Leave success criteria open if
actual evidence does not establish useful policy performance and generalisation.

## Part 2: teacher–student and auxiliary learning

Implement optional recurrent policy distillation from an actual loaded, frozen trained checkpoint.
A larger teacher can use the existing legal observation encoders and hidden size; the student architecture
may differ. Both consume actual student-visited training observations with independent recurrent/memory
state and proper episode resets. The teacher uses its own manager cadence; soft action-group categorical
KL loss transfers behavior without copying goals or privileged features into the actor. Teacher rollout
runs no gradients; checkpoint hash, architecture, coefficient and temperature are saved. Validate finite
positive temperature, nonnegative coefficient and compatible action spaces; reject missing/random teachers.
PPO fine-tuning continues, and coefficient zero reproduces normal training. Resume restores configuration
and rejects teacher identity changes. The initial implementation uses a legal-observation teacher; a new
privileged full-map teacher is outside this PR because it needs a separate observation contract and a
separately trained teacher. Update roadmap documentation to state this distinction explicitly.

Add optional next-step dynamics auxiliary prediction: from recurrent state and the executed action,
predict changes in legal normalized vitals and ray proximity. Targets come from actual consecutive
observations, with terminal/reset transitions masked; no invented labels or privileged inference inputs.
Loss coefficients and target normalization are explicit, validated and checkpointed. Auxiliary heads are
unused by inference. Preserve old architecture defaults and allow explicit warm starts; do not silently
change loaded checkpoints. Verify gradients reach intended student encoders, teacher parameters remain
unchanged, recurrent resets are correct, and actual small training updates affect parameters.

## Commit structure and verification

1. Design and implementation plans.
2. Versioned map randomisation plus generator/engine tests.
3. Stage 5 pool, rollout and resume integration.
4. Stress runner and generalisation reporting; locked final-test entry point.
5. Measured inference checks and any justified speed changes.
6. Frozen recurrent teacher distillation with checkpoint provenance.
7. Transition auxiliary loss with terminal masking and compatibility tests.
8. Real smoke evidence, docs and final review fixes as needed.

Each implementation commit gets relevant tests first and verification before commit. Run broad regressions,
real engine smokes, strict candidate validator and latency measurements. Record exact exclusions and hardware.
Keep champion weights unchanged. PR description separates delivered functionality, measured outcomes and
remaining scientific criteria. The feature branch may be pushed to the fork solely as the upstream PR source.

## Builder review

Independent builder review identified one ambiguity: whether older evaluation routes could bypass the
final-test ledger. Resolved above with a shared identity-based guard covering existing entry points.
No other material planning questions remained.
