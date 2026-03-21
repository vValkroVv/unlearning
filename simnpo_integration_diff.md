# SimNPO Integration Diff

## 2026-03-21 DUET/RWKU Integration

Updated files:
- `configs/experiment/unlearn/duet/simnpo_lora.yaml`
- `configs/experiment/unlearn/rwku/simnpo_lora.yaml`
- `scripts/duet/simnpo_duet.sh`
- `scripts/rwku/simnpo_rwku.sh`
- `scripts/duet/run_dualcf_ablation_v2.sh`
- `scripts/rwku/run_dualcf_ablation_v2.sh`
- `scripts/dualcf/run_campaign_one_lr.sh`
- `check_saves.py`
- `src/tools/build_structured_saves.py`
- `src/tools/build_results_combine_tables.py`

What changed:
- Added dedicated DUET and RWKU SimNPO experiment configs that expose the active LoRA defaults and `delta / beta / alpha / gamma` method arguments.
- Added direct DUET and RWKU launch scripts that follow the current baseline pattern: LoRA sweep loops, half-epoch checkpointing, final eval, and optional checkpoint eval.
- Registered `simnpo` in the shared DUET/RWKU ablation wrapper and in the one-LR campaign wrapper so it can run through the same production campaign path as `ga`, `npo`, `npo_sam`, and `loku`.
- Updated save checking and structured-results parsing so `simnpo` runs are detected and included in downstream tables instead of being skipped as unknown method names.
- Added `SimNPO` rows to the combined results table builder so the method appears alongside the other old-baseline runs.

Example commands:
- DUET direct:
  `bash scripts/duet/simnpo_duet.sh`
- RWKU direct:
  `bash scripts/rwku/simnpo_rwku.sh`
- DUET wrapper:
  `METHOD_VARIANT=simnpo FORGET_LABEL=rare bash scripts/duet/run_dualcf_ablation_v2.sh`
- RWKU wrapper:
  `METHOD_VARIANT=simnpo bash scripts/rwku/run_dualcf_ablation_v2.sh`
