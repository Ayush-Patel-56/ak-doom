# Depth, Stage 4 and ablations implementation plan

**Approved spec:** ../specs/2026-11-26-depth-stage4-ablations-design.md

**Goal:** Optional real-depth encoding, training-only full-level curriculum, and controlled held-out manager/RND comparisons.

**Architecture:** Extend the existing checkpoint configuration and named encoder transfer; canonicalize catalog map identities before selecting split-specific pools; use a manifest-driven experiment runner with strict trained-checkpoint loading and real image buffers.

## Tasks

- [x] Depth: add feature/config/CNN and warm-start named block, trainer/rollout flags and render-buffer activation. Test real depth influence/gradients, legacy loads, explicit warm starts and real full-level training/reload.
- [x] Pool/curriculum (independent): training/map_pool.py exposes load_catalog_pool(split, tiers, source=None, catalog_path=..., require_assets=True) and map_identity(spec); curriculum.MapSampler accepts stage4/full_level_share/catalog_path and exposes episode_ticks(wad,map) plus pool manifest. Preserve existing three-tuple sampler interface. Use only usable train-split Tier4 Freedoom levels, catalog caps, replay and deterministic seeds; reject alias leakage/invalid metadata/missing assets.
- [x] Ablations (independent): scripts/phase4_ablation.py builds matched full/no-manager/no-curiosity/phase3 arms and runs existing training/evaluation. Shared helper setting OFF avoids confounding manager with goal rewards. Record typed configs, initialization/checkpoint identities, raw Tier3/4 outcomes, paired differences, sample counts and failures. Strict checkpoint load and real pixel buffers; reject arm/budget/override errors. Keep scripts/ablation.py Phase3 API compatible, adding a route to Phase4 comparisons if useful.
- [x] Integrate: Stage4 CLI/sampler/catalog tick caps and reproducibility metadata, real smoke comparisons and depth latency; tests first for each boundary.
- [x] Verify: full available regressions, candidate validator and benchmark, independent review. Document actual outcomes without claiming improvement from smoke samples. Reconcile dependency on PR13 before pushing only the new branch and opening a described PR against original main.

## Review focus

Catalog aliases crossing splits; real image buffers during evaluation; random fallback mistaken for an arm; equal training exposure/initial shared parameters; goal helper confounding; resume pool/tick/config changes; named depth warm-start mapping; measured latency and insufficient evaluation sample sizes.

## Execution ledger

- Branch starts at tested PR13 merge commit443ae25. PR13 was still open at design approval; do not publish duplicate feature diffs without disclosing/reconciling that dependency.
- Independent pool/curriculum and experiment runner tasks will be delegated without shared edits; depth and integration stay with the primary agent.

- Final regressions: 260 passed, two known macOS-incompatible tests deselected. Independent review found three issues; regression tests verified fixes for default budget divisibility, missing Stage 4 resume provenance and rejected-budget summary inclusion.
- PR13 merged upstream as 5d438f1; this branch now starts from that merge. Four matched arms trained 512 decisions each; all 16 shortened held-out episodes had zero exits. Depth candidate mean latency 2.227–2.253 ms on this Mac. Raw evidence is committed in configs/depth-stage4-smoke.json; scientific improvement remains open.
