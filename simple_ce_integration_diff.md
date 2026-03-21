# SimpleCE Integration Diff

## 2026-03-21 DUET/RWKU Integration

Updated files:
- `src/trainer/unlearn/simple_ce.py`
- `src/trainer/__init__.py`
- `configs/trainer/SimpleCE.yaml`
- `configs/experiment/unlearn/duet/simple_ce_lora.yaml`
- `configs/experiment/unlearn/rwku/simple_ce_lora.yaml`
- `scripts/duet/simple_ce_duet.sh`
- `scripts/rwku/simple_ce_rwku.sh`
- `scripts/duet/run_dualcf_ablation_v2.sh`
- `scripts/rwku/run_dualcf_ablation_v2.sh`
- `scripts/dualcf/run_campaign_one_lr.sh`
- `check_saves.py`
- `src/tools/build_structured_saves.py`
- `src/tools/build_results_combine_tables.py`

What changed:
- Added a new `SimpleCE(GradDiff)` trainer that consumes the existing DualCF/DPO counterfactual batch structure and optimizes `-CE(forget.original) + cf_weight * CE(forget.alternate) + retain_weight * CE(retain)`.
- Added a trainer config plus DUET/RWKU experiment configs that reuse the current `*_QA_forget_dual_cf` datasets, so SimpleCE trains on the same counterfactual artifacts already used by DualCF and DPO.
- Added direct DUET and RWKU launch scripts with the same local-JSON / HF dataset override flow, half-epoch checkpoint scheduling, eval, and optional checkpoint eval used by the current counterfactual baselines.
- Registered `simple_ce` in the shared DUET/RWKU ablation wrapper and in the one-LR campaign wrapper so it runs through the existing campaign machinery without a separate ad hoc path.
- Updated save checking and structured-results tooling so `simple_ce` runs are recognized and included in downstream summary tables.
- `SimpleCE` now exposes both explicit loss weights with defaults:
  - `cf_weight=1.0`
  - `retain_weight=1.0`
- The SimpleCE launchers now accept `RETAIN_WEIGHTS`, while still falling back to older `ALPHAS` overrides for backward compatibility.

Example commands:
- DUET direct:
  `CF_DATASET_DATA_FILES=/abs/path/to/duet_simple_ce.jsonl bash scripts/duet/simple_ce_duet.sh`
- RWKU direct:
  `CF_DATASET_DATA_FILES=/abs/path/to/rwku_simple_ce.jsonl bash scripts/rwku/simple_ce_rwku.sh`
- DUET wrapper:
  `METHOD_VARIANT=simple_ce FORGET_LABEL=rare CF_DATASET_DATA_FILES=/abs/path/to/duet_simple_ce.jsonl bash scripts/duet/run_dualcf_ablation_v2.sh`
- RWKU wrapper:
  `METHOD_VARIANT=simple_ce CF_DATASET_DATA_FILES=/abs/path/to/rwku_simple_ce.jsonl bash scripts/rwku/run_dualcf_ablation_v2.sh`
