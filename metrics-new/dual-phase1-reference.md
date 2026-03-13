# DualCF Phase1 Reference

This file defines the current DualCF baseline used in
`metrics-new/duet-rwku-metric-tables.md`.

All rows named `DualCF (Phase1)` refer to this implementation state.

Metrics source:

- `metrics-new/ep2_dual/saves-dual-cos-sim`

## Algorithm summary

- DualCF builds forget-side artifacts with `question`, original `answer`, and a
  counterfactual `alternate` answer.
- The forget artifact is enriched with two routing signals:
  - `difficulty_score`
  - `attribution_score`
- Training uses three practical components:
  - cross-entropy on the counterfactual answer
  - NPO-style pressure on the original forget answer
  - retain loss on a retain batch
- In the current phase:
  - higher `difficulty_score` increases negative pressure on the original answer
  - higher `attribution_score` reduces destructive forget pressure and increases
    retain-side weight
- The current production-shaped reference uses:
  - merged DUET `city_forget_5`
  - RWKU `forget_level2`
  - `2` training epochs
  - LoRA parity between attribution scoring and training:
    `r=32`, `alpha=64`, `dropout=0.0`

## Phase1 commit chain

1. `7c4200f` (2026-03-09) `Add DualCF unlearning integration`
   Initial trainer, dataset/collator plumbing, configs, launch scripts, and
   offline artifact tools.
2. `7d1822c` (2026-03-09) `docs: tighten DualCF test and artifact plan`
   Tightened the validation plan and artifact/test expectations.
3. `e23a7c9` (2026-03-09) `Fix DualCF verification issues and document testing`
   Added verification-driven fixes and documented the corrected testing flow.
4. `3b51ed4` (2026-03-09) `Add DualCF merged verification outputs and docs`
   Added merged DUET verification outputs and related documentation.
5. `e4382dd` (2026-03-14) `Update DualCF tooling and metrics`
   Production-aligned tooling, runbook updates, artifact observability, LoRA
   parity, and the current metric-table state.

## Comparison rule

- Treat `DualCF (Phase1)` as the frozen baseline.
- Future Dual variants should use explicit row names such as
  `DualCF (Phase2)` so the tables remain directly comparable.
