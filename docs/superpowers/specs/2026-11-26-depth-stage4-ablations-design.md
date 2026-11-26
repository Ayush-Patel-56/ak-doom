# Depth, full-level curriculum, and controlled Phase 4 comparisons

## Requested outcome

Add the remaining three Phase 4 items: an optional depth encoder gated by measured inference speed, Curriculum Stage 4 on real full levels, and reproducible manager/curiosity ablations. Deliver a reviewed PR targeting Ayush-Patel-56/ak-doom:main, with actual measurements and a clear description. Only the feature branch may be pushed; do not merge a PR or push either main.

## Approach

Extend the existing recurrent network, curriculum sampler, and evaluation pipeline. Use the checked-in map catalog and installed ViZDoom/Freedoom assets rather than adding an external level-generator dependency. Extend the existing ablation CLI while keeping its Phase 3 memory comparison available.

An external generator would add installation and licensing work without being necessary for full-level training. A separate experiment framework would duplicate existing training/evaluation paths. Disabling trained depth dynamically at inference would change the learned input distribution, so depth remains an explicit checkpoint architecture choice instead.

## Optional depth encoder

- Add a small learned single-channel CNN on real 60x80 downsampled depth from the official uint8 observation. Normalize the provided buffer values; do not reinterpret them as calibrated metric distances.
- Fuse depth with the existing legal sensor/image embeddings. Default the new architecture switch off for older checkpoints. Enable actual render buffers whenever either image encoder requires them.
- Extend explicit warm-start transfer by named input blocks so existing map/RGB columns stay aligned and newly added depth columns start at zero.
- Exercise depth-dependent outputs/gradients, legacy checkpoint loading, training/resume and inference with real engine observations.
- Benchmark the depth-enabled candidate using the existing one-thread harness on representative maps. Keep depth optional and report measured mean/p99 and official-budget checks. Do not promote it as a default champion feature unless evidence supports the 20 ms average decision target. Report macOS/pinning limitations honestly.

## Curriculum Stage 4

- Select usable full Freedoom levels from catalog entries explicitly assigned to the training split. Never train on validation/test entries, including alternate aliases of the same WAD/map.
- Resolve installed asset paths through the existing portable map-spec logic. Validate missing assets, invalid metadata and empty training pools explicitly; do not silently substitute a held-out map or a small puzzle.
- Use a configurable full-level share with earlier-stage replay, defaulting to the project's documented 80/20 replay convention. Keep sampling seeded and record the pool/mix used by each experiment.
- Use normal engine skill 3, no modified doors/ammo/spawns, and catalog episode caps in the official 2100–4200 range unless an explicit smoke-test tick override is requested and reported. Generated replay maps retain disjoint training seed ranges.
- RGB exit supervision remains masked for unsupported full-level map geometry. Do not invent exit-visibility labels to fill that gap.

## Ablation protocol

- Add matched arms for hierarchy+RND control, minus manager, and minus curiosity. Keep memory, RGB/depth flags, training map exposure, PPO settings, decision budgets, initialization source and paired seeds consistent except for the named treatment.
- Disable goal helpers in all primary manager-comparison arms, because removing the manager otherwise also changes goal-specific reward shaping. Record that choice explicitly; do not call a compound reward/architecture change a clean manager ablation.
- Include a clearly identified Phase 3 architecture reference for the requested Phase 3-versus-Phase 4 comparison. Distinguish that broader architecture comparison from a one-component ablation.
- Persist resolved configurations and initialization/checkpoint identities, per-episode raw results, stopped/crashed runs, and per-tier Tier 3/Tier 4 exit rates. Evaluation must use real render buffers for checkpoints requiring pixels and must fail explicitly if the requested trained checkpoint cannot load, rather than reporting random fallback as that arm.
- Use validation maps for development comparisons. Keep test maps out of smoke tuning; require an explicit test-split choice. Reject duplicate arm names, invalid budgets, missing checkpoints, empty evaluation pools and extra CLI arguments that override controlled treatment/budget/seed settings.
- Report paired differences and sufficient episode counts alongside exit rates. Treat short smoke runs as pipeline verification only. A positive exit-rate difference in a tiny sample is not proof of a robust improvement.

## Success criteria and evidence

Implementation success requires real-engine full-level training, depth-enabled checkpoint reload/play, complete matched smoke comparisons on held-out Tier 3/4 maps, saved raw reports, regression tests, and inference latency checks. Keep measured outcomes separate from the roadmap's scientific completion criterion.

The roadmap is complete only when adequately trained Phase 4 models improve held-out Tier 3/4 exit rates over the Phase 3 reference under a stated, matched protocol. If measured runs do not show improvement, report that result and leave the effectiveness criterion open. Never fabricate wins, thresholds met, map samples, labels or trained weights.

## Branching and delivery

PR #13 remains open at design time. Work starts on a separate branch based on its tested feature state. Before delivery, reconcile with upstream so the new PR does not accidentally duplicate already-merged work; clearly disclose any remaining dependency on PR #13. Preserve unrelated local setup edits outside feature commits. Obtain an independent review, push only the feature branch, and open the PR against the original repository's main with verification evidence, performance results, and limitations.
