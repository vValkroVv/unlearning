# Unilogit Integration Diff

Target: current working tree

## Unilogit baseline integration (2026-05-10)

This update adds `unilogit` as a first-class old-baseline method in the DUET /
RWKU campaign stack. It follows the existing SimNPO launcher surface and does
not consume DualCF counterfactual artifacts inside the trainer.

Changed files for this patch:

- `src/trainer/unlearn/unilogit.py`
- `src/trainer/__init__.py`
- `configs/trainer/Unilogit.yaml`
- `configs/experiment/unlearn/duet/unilogit_lora.yaml`
- `configs/experiment/unlearn/rwku/unilogit_lora.yaml`
- `scripts/duet/unilogit_duet.sh`
- `scripts/rwku/unilogit_rwku.sh`
- `scripts/duet/run_dualcf_ablation_v2.sh`
- `scripts/rwku/run_dualcf_ablation_v2.sh`
- `scripts/dualcf/run_campaign_one_lr.sh`
- `check_saves.py`
- `src/tools/build_structured_saves.py`
- `src/tools/build_results_combine_tables.py`
- `src/tools/export_unlearning_sanity_checks.py`
- `src/tools/analyze_wrong_generations.py`
- `prod-run-dual-gpu.md`
- `unilogit_integration_diff.md`

Behavior change summary:

- `Unilogit` is registered as a trainer with:
  - `forget_coef`
  - `kl_direction={model_to_target,target_to_model}`
  - `alpha`
  - `gamma`
  - `retain_loss_type`
- the forget objective builds a detached modified uniform-target distribution
  from the current logits, filters ignored labels before target assignment, and
  passes `attention_mask` through the model call for padded QA batches
- DUET and RWKU direct launchers expose `FORGET_COEFS`, `KL_DIRECTIONS`,
  `ALPHAS`, and `GAMMAS`, and keep the same train -> endpoint eval ->
  checkpoint eval -> utility eval -> cleanup cadence as the existing SimNPO /
  NPO launchers
- the campaign wrapper accepts `METHOD_VARIANTS=unilogit` and includes it in
  the default method list between `simnpo` and `npo_sam`
- save checking, structured-save parsing, wrong-generation parsing,
  sanity-report parsing, and combined-table row specs now recognize
  `_unilogit_lora_` run names
- `prod-run-dual-gpu.md` now includes Unilogit commands after the GeneralCF
  family, including default, forget-only, and KL-direction campaign sweeps

Validation status:

- passed static validation:
  - `python -m py_compile src/trainer/unlearn/unilogit.py src/trainer/__init__.py src/tools/build_structured_saves.py src/tools/build_results_combine_tables.py src/tools/export_unlearning_sanity_checks.py src/tools/analyze_wrong_generations.py check_saves.py`
  - `bash -n scripts/duet/unilogit_duet.sh`
  - `bash -n scripts/rwku/unilogit_rwku.sh`
  - `bash -n scripts/duet/run_dualcf_ablation_v2.sh`
  - `bash -n scripts/rwku/run_dualcf_ablation_v2.sh`
  - `bash -n scripts/dualcf/run_campaign_one_lr.sh`
  - YAML load for `configs/trainer/Unilogit.yaml`,
    `configs/experiment/unlearn/duet/unilogit_lora.yaml`, and
    `configs/experiment/unlearn/rwku/unilogit_lora.yaml`
- passed a tensor-level loss smoke for both KL directions with a dummy language
  model and local import stubs; a normal package import check was not possible
  in this local environment because `deepspeed` is not installed
- passed a structured-save parser smoke that maps a representative
  `_unilogit_lora_` DUET run name to method key `unilogit`
- no GPU train smoke was run in this patch
