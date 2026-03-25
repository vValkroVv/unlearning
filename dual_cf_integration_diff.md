# DualCF Integration Diff

Base commit: `3e15a8ba7682cf316469a6ffc417c62d33aa22b1` (before DualCF integration)
Target: current working tree

## What was added

DualCF is integrated as a new routed unlearning method that keeps the repo's
existing `forget.original` / `forget.alternate` DPO-style batch structure, but
optimizes the practical decomposition

\[
CE(y^{cf}\mid x) + \lambda_i^{neg} L_{NPO}(x, y^{orig}) + \alpha_{eff} L_{ret},
\]

with:

- per-sample difficulty routing on the forget side,
- per-sample attribution routing on the forget side,
- batch-level `alpha_eff` as the retain-side proxy that fits the current
  `GradDiff` trainer flow.

## Changed files

### Core trainer path

- `src/trainer/utils.py`
- `src/trainer/unlearn/dual_cf.py`
- `src/trainer/__init__.py`
- `configs/trainer/DualCF.yaml`

### Dataset / collator plumbing

- `src/data/qa.py`
- `src/data/collators.py`
- `src/data/__init__.py`
- `src/data/utils.py`
- `configs/data/datasets/DUET_QA_forget_dual_cf.yaml`
- `configs/data/datasets/POPQA_QA_forget_dual_cf.yaml`
- `configs/data/datasets/RWKU_QA_forget_dual_cf.yaml`

### Experiment configs

- `configs/experiment/unlearn/duet/dual_cf_lora.yaml`
- `configs/experiment/unlearn/duet/ada_pop_lora.yaml`
- `configs/experiment/unlearn/popqa/dual_cf_lora.yaml`
- `configs/experiment/unlearn/rwku/dual_cf_lora.yaml`
- `configs/experiment/unlearn/rwku/ada_pop_lora.yaml`

### Launch scripts

- `scripts/duet/dual_cf_duet.sh`
- `scripts/duet/ada_pop_duet.sh`
- `scripts/popqa/dual_cf_popqa.sh`
- `scripts/rwku/dual_cf_rwku.sh`
- `scripts/duet/ga_duet.sh`
- `scripts/duet/npo_duet.sh`
- `scripts/duet/npo_sam_duet.sh`
- `scripts/duet/loku_duet.sh`
- `scripts/rwku/ada_pop_rwku.sh`
- `scripts/rwku/ga_rwku.sh`
- `scripts/rwku/npo_rwku.sh`
- `scripts/rwku/npo_sam_rwku.sh`
- `scripts/rwku/loku_rwku.sh`

### Documentation / runbooks

- `prod-run-dual-vast.md`

### End-to-end runner scripts

- `dual-scripts-run/run_llama_dual_cf_e2e.sh`
- `dual-scripts-run/run_qwen_dual_cf_e2e.sh`
- `dual-scripts-run/run_gemma_dual_cf_e2e.sh`

### Offline artifact tools

- `src/tools/dual_cf_artifact_utils.py`
- `src/tools/make_counterfactuals.py`
- `src/tools/score_difficulty.py`
- `src/tools/score_attribution.py`

## Core implementation notes

### `src/trainer/utils.py`

Added per-sample helpers without changing legacy DPO/NPO helpers:

```python
def _token_counts_from_labels(labels: torch.Tensor) -> torch.Tensor:
    counts = (labels[..., 1:] != -100).sum(dim=-1)
    return counts.clamp_min(1)

def compute_nll_per_sample(model, inputs, normalize_by_tokens: bool = True):
    ...

def compute_npo_per_sample(model, ref_model, lose_inputs, beta: float = 1.0, ...):
    ...
```

This preserves routed per-sample signals and avoids length bias from raw sequence
sums.

### `src/trainer/unlearn/dual_cf.py`

New `DualCF(GradDiff)` trainer:

- uses `compute_nll_per_sample()` on `forget.alternate`,
- uses `compute_npo_per_sample()` on `forget.original`,
- computes soft routing gates
  - `s = sigmoid((difficulty - tau_d) / temp_d)`
  - `r = sigmoid((attribution - tau_a) / temp_a)`
- applies
  - `lambda_neg = lambda_neg_max * s * (1 - r)`
  - `forget_scale = 1 - (1 - risk_forget_scale) * r`
  - `alpha_eff = alpha * (lambda_ret_lo + (lambda_ret_hi - lambda_ret_lo) * r.mean())`
- logs `dualcf_*` diagnostics for smoke tests and sweeps.

Important repo-fit constraint:

- retain weighting is implemented as batch-level `alpha_eff`, not literal
  per-sample `lambda_i^{ret}`, because `GradDiff` consumes one scalar retain loss
  from a separate retain batch.

### Dataset / collator path

`QAwithAlternateMetadataDataset` returns:

```python
{
    "original": tokenized_original,
    "alternate": tokenized_alternate,
    "difficulty_score": float(...),
    "attribution_score": float(...),
    "index": int(...),
}
```

`DataCollatorForSupervisedDataset` now recursively stacks numeric scalar metadata,
so routed scores survive batching without a method-specific collator.

`add_dataset_index()` now no-ops if an artifact already contains an `index`
column, which is required for premerged DualCF datasets.

## Artifact schema

The expected forget-side artifact row is:

```json
{
  "index": 17,
  "question": "...",
  "answer": "...",
  "alternate": "...",
  "difficulty_score": 0.73,
  "attribution_score": 0.18
}
```

The experiment configs support either:

- a Hugging Face dataset path / repo in `cf_dataset_path`, or
- a local JSON/JSONL path by setting:
  - `cf_dataset_path=json`
  - `cf_dataset_data_files=/abs/path/to/file.jsonl`

## Launch configuration

Each benchmark now has a `dual_cf_lora.yaml` experiment config plus a launcher
script with env-controlled routing knobs:

- `BETAS`
- `TAU_DS`
- `TAU_AS`
- `TEMP_DS`
- `TEMP_AS`
- `LAMBDA_NEG_MAXS`
- `LAMBDA_RET_LOS`
- `LAMBDA_RET_HIS`
- `CF_WEIGHTS`
- `RISK_FORGET_SCALES`
- `DISABLE_DIFFICULTY_ROUTES`
- `DISABLE_ATTRIBUTION_ROUTES`

Operational update:

- experiment configs now default to local JSON artifact mode with
  - `cf_dataset_path=json`
  - `cf_dataset_split=train`
- in local JSON mode, actual DualCF forget training is controlled by
  `cf_dataset_path`, `cf_dataset_data_files`, and especially `cf_dataset_split`;
  `forget_split` remains the benchmark/eval identity and does not shrink the
  training artifact by itself
- launcher scripts now fail fast if:
  - `CF_DATASET_PATH` still uses a placeholder path
  - local JSON mode is selected without `CF_DATASET_DATA_FILES`
- launcher scripts now support controlled smoke-test execution with:
  - `MAX_STEPS`
  - `FORGET_SPLIT_OVERRIDE`
  - `RETAIN_SPLIT_OVERRIDE`
  - `FORGET_LABEL_OVERRIDE`
- DUET and POPQA switch back to benchmark-native split names only when
  `CF_DATASET_PATH` is overridden away from `json`
- RWKU local JSON mode defaults to `cf_dataset_name=null` and
  `cf_dataset_split=train`

## Post-integration fixes

- `scripts/duet/dual_cf_duet.sh` and `scripts/rwku/dual_cf_rwku.sh` no longer
  unconditionally append DualCF-only Hydra overrides when `TRAINER=DPO`.
  `DPO` now only receives `beta`, `alpha`, `gamma`, and `retain_loss_type`,
  which matches `configs/trainer/DPO.yaml` and avoids Hydra struct errors such
  as `Key 'tau_d' is not in struct`.
- `scripts/duet/eval_checkpoints_duet.sh` and
  `scripts/rwku/eval_checkpoints_rwku.sh` now treat `run_dir/base_model` as a
  standalone LoKU FILA base model:
  - they only use it when it has `config.json` plus model weights
  - they stop inheriting the DUET SFT `MODEL_SUBFOLDER` for LoKU checkpoint and
    utility evals
  - otherwise they fall back to the original base model path instead of failing
    on an incomplete `base_model` directory
- `scripts/utility/eval_checkpoints_utility.sh` now mirrors that LoKU behavior
  for checkpoint/final adapter utility evals: if `LORA_BASE_MODEL_PATH` points
  at a standalone saved `base_model`, it clears only the model subfolder while
  keeping the tokenizer subfolder intact.
- `scripts/dualcf/run_campaign_one_lr.sh` now exports
  more aggressive H100 throughput defaults:
  - `IMPORTANCE_BATCH_SIZE=32`
  - `EVAL_BATCH_SIZE=128`
  - `UTILITY_EVAL_BATCH_SIZE=64`
  instead of inheriting the older lower-throughput values.
- `scripts/duet/loku_duet.sh` and `scripts/rwku/loku_rwku.sh` now use less
  antiquated standalone defaults when unset explicitly:
  - `IMPORTANCE_BATCH_SIZE=32`
  - `EVAL_BATCH_SIZE=128`
  instead of `1` / `8`.
- DUET and RWKU endpoint / checkpoint eval defaults are now normalized across
  the main launcher scripts:
  - `EVAL_BATCH_SIZE=128` for DUET/RWKU training-side evals and checkpoint evals
  - `UTILITY_EVAL_BATCH_SIZE=64` for Utility-1K sweeps
  replacing the previous mixed `8` / `64` defaults.
- The H100 production profile used by the DualCF v2 campaign was then raised
  again for the main ablation tree (`dual`, `ga`, `npo`, `npo_sam`, `loku`):
  - `PER_DEVICE_TRAIN_BS=32`
  - `GRAD_ACCUM=1`
  - `EVAL_BATCH_SIZE=192`
  in both the shared campaign wrapper and the direct DUET/RWKU launcher
  defaults, so direct reruns and campaign runs stay aligned.
- `scripts/dualcf/run_campaign_one_lr.sh` now resolves the DUET SFT base to a
  real local directory before launch and defaults it to the offline filesystem
  path `/data/home/vkropoti/unlearning/SwetieePawsss/DUET_ft_models`, instead
  of relying on a repo-style HF identifier that can fail in offline mode.
- after wrapper-only OOMs in the 4-run campaign path,
  `scripts/dualcf/run_campaign_one_lr.sh` was pulled back to the previously
  stable train defaults:
  - `PER_DEVICE_TRAIN_BS=16`
  - `GRAD_ACCUM=2`
  while leaving the direct DUET/RWKU launcher defaults unchanged
- the offline dataset symlink step
  `ln -sfn /data/home/vkropoti/unlearning/SwetieePawsss /home/vkropoti/diploma/open-unlearning/SwetieePawsss`
  was moved into the common setup block of `prod-run-dual-gpu.md`
- the GPU validation playbook now lives in `plan-test-dual.md`, including:
  - explicit split-matched artifact preparation for DUET rare / popular / merged
  - stronger artifact validation and provenance requirements
  - corrected direct smoke and small-run commands that slice `cf_dataset_split`
    in local JSON mode
  - a 1B DUET launcher smoke command with `USE_SFT_BASE=0`
  - DUET-first, RWKU-second execution order for the main campaign

This supports the intended ablations without additional trainer classes:

- full DualCF
- difficulty-only
- attribution-only

Uniform counterfactual remains available through the existing `DPO` trainer.

## Baseline parity update (2026-03-14)

To make the DUET/RWKU ablation tree directly comparable against DualCF v2, the
baseline launchers now mirror the same run-management behavior:

- respect a shared `OUTPUT_ROOT`, so production runs can stay under
  `/data/home/vkropoti/unlearning/saves/unlearn`
- support `CHECKPOINT_EVERY_HALF_EPOCH=1` with dynamic `save_steps` computed
  from the actual forget split size
- support `SAVE_TOTAL_LIMIT` and `MAX_STEPS` in the same style as the DualCF
  launchers
- enable `trainer.trace_jsonl=true`, which writes `dualcf_trace.jsonl` into each
  run directory for GA / NPO / NPO-SAM / LoKU as well
- keep `DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1` behavior consistent, removing
  top-level run safetensors after endpoint evaluation while leaving
  `checkpoint-*` directories available for trajectory evaluation
- LoKU now keeps `run_dir/base_model` by default during trajectory campaigns,
  and checkpoint evaluators auto-detect that FILA base model for checkpoint
  scoring before optional cleanup

The checkpoint/eval flow is still two-stage for every method:

1. run the launcher or ablation wrapper to train and do the endpoint eval
2. run `scripts/duet/eval_checkpoints_duet.sh` or
   `scripts/rwku/eval_checkpoints_rwku.sh` on the saved run directory to score
   all half-epoch checkpoints and produce `checkpoint_evals/summary.tsv`

## AdaPop production-parity uplift (2026-03-23)

AdaPop was already registered in the repo, but it was still wired through the
older WGA-era DUET / RWKU launchers. The current uplift keeps the same method
family while aligning AdaPop with the production baseline stack.

Changed files:

- `src/trainer/utils.py`
- `src/trainer/unlearn/ada_pop.py`
- `configs/trainer/AdaPop.yaml`
- `configs/experiment/unlearn/duet/ada_pop_lora.yaml`
- `configs/experiment/unlearn/rwku/ada_pop_lora.yaml`
- `scripts/duet/ada_pop_duet.sh`
- `scripts/rwku/ada_pop_rwku.sh`
- `scripts/duet/run_dualcf_ablation_v2.sh`
- `scripts/rwku/run_dualcf_ablation_v2.sh`
- `scripts/dualcf/run_campaign_one_lr.sh`
- `check_saves.py`
- `src/tools/build_structured_saves.py`
- `src/tools/export_unlearning_sanity_checks.py`
- `src/tools/build_results_combine_tables.py`

Updates:

- `beta_from_pop_sum_tensor()` and AdaPop now expose configurable
  `beta_a=58.7` and `beta_b=0.796`, instead of hard-wiring the clipped
  popularity law shape
- AdaPop now has dedicated DUET / RWKU experiment configs instead of reusing
  `wga_lora.yaml`
- `scripts/duet/ada_pop_duet.sh` now matches the newer DUET baseline launchers:
  - respects shared `OUTPUT_ROOT`
  - supports tokenizer path / subfolder overrides for DUET SFT bases
  - supports `EVAL_BATCH_SIZE`, `MAX_STEPS`, `CHECKPOINT_EVERY_HALF_EPOCH`, and
    `SAVE_TOTAL_LIMIT`
  - runs endpoint eval, optional checkpoint eval, and optional top-level
    safetensors cleanup in the same lifecycle as GA / NPO / SimNPO
- `scripts/rwku/ada_pop_rwku.sh` now matches the newer RWKU baseline launchers:
  - respects shared `OUTPUT_ROOT`
  - supports `TOKENIZER_MODEL_PATH`, `EVAL_BATCH_SIZE`, `MAX_STEPS`,
    `CHECKPOINT_EVERY_HALF_EPOCH`, and `SAVE_TOTAL_LIMIT`
  - propagates `HF_TOKEN` into `HUGGINGFACE_HUB_TOKEN` when needed for gated
    environments
  - keeps the existing `eval.duet.batch_size=...` RWKU override style used by
    the other RWKU direct launchers
- the campaign and wrapper path now includes AdaPop by default:
  - `scripts/duet/run_dualcf_ablation_v2.sh`
  - `scripts/rwku/run_dualcf_ablation_v2.sh`
  - `scripts/dualcf/run_campaign_one_lr.sh`
- save checking and downstream summary tooling now recognize `ada_pop` run
  names, so AdaPop no longer disappears from structured saves, sanity exports,
  or combined table generation

## Verification-driven fixes (2026-03-09)

After running the DUET and RWKU smoke/functional checks from
`plan-test-dual.md`, the following repo-fit fixes were added.

### Hugging Face auth propagation

Changed files:

- `src/model/__init__.py`
- `src/model/lora.py`
- `src/data/utils.py`
- `src/tools/dual_cf_artifact_utils.py`

Fix:

- shared model/tokenizer/dataset loaders now forward `HF_TOKEN` /
  `HUGGINGFACE_HUB_TOKEN` / `HF_HUB_TOKEN` into gated Hugging Face loads

Why:

- artifact tools and train/eval runs against gated Meta Llama checkpoints were
  failing until auth was passed explicitly

### Model override compatibility

Changed files:

- `configs/experiment/unlearn/duet/dual_cf_lora.yaml`
- `configs/experiment/unlearn/popqa/dual_cf_lora.yaml`
- `configs/experiment/unlearn/rwku/dual_cf_lora.yaml`

Fix:

- removed hardcoded 8B `model.model_args.pretrained_model_name_or_path` entries

Why:

- CLI model overrides like `model=Llama-3.2-1B-Instruct-lora` were still
  loading the 8B base model, which broke the documented 1B smoke path

### Local JSON split quoting in launchers

Changed files:

- `scripts/duet/dual_cf_duet.sh`
- `scripts/popqa/dual_cf_popqa.sh`
- `scripts/rwku/dual_cf_rwku.sh`

Fix:

- local JSON runs now quote `cf_dataset_split` consistently so Hydra receives the
  intended split literal

## DualCF v2 upgrade (2026-03-14)

This update integrates the reviewer-requested next iteration of DualCF instead
of only documenting the earlier routed baseline.

### New files

Artifact generation / scoring:

- `src/tools/vllm_cf_client.py`
- `src/tools/build_duet_candidate_bank.py`
- `src/tools/clean_counterfactuals.py`
- `src/tools/build_proxy_retain_map.py`
- `src/tools/calibrate_dual_cf_scores.py`
- `src/tools/summarize_checkpoint_metrics.py`

Trainer / callbacks:

- `src/trainer/callbacks/__init__.py`
- `src/trainer/callbacks/jsonl_trace.py`

Configs / scripts:

- `configs/experiment/unlearn/duet/dual_cf_v2_lora.yaml`
- `configs/experiment/unlearn/rwku/dual_cf_v2_lora.yaml`
- `scripts/vllm/start_qwen3_cf_server.sh`
- `scripts/duet/prepare_dual_cf_duet_v2.sh`
- `scripts/rwku/prepare_dual_cf_rwku_v2.sh`
- `scripts/duet/run_dualcf_ablation_v2.sh`
- `scripts/rwku/run_dualcf_ablation_v2.sh`
- `scripts/duet/eval_checkpoints_duet.sh`
- `scripts/rwku/eval_checkpoints_rwku.sh`

### Updated files

- `src/tools/dual_cf_artifact_utils.py`
- `src/tools/make_counterfactuals.py`
- `src/tools/score_difficulty.py`
- `src/tools/score_attribution.py`
- `src/tools/validate_dual_cf_artifact.py`
- `src/trainer/unlearn/dual_cf.py`
- `src/trainer/__init__.py`
- `configs/trainer/DualCF.yaml`
- `scripts/duet/dual_cf_duet.sh`
- `scripts/rwku/dual_cf_rwku.sh`
- `prod-run-dual-gpu.md`

### Counterfactual generator v2

`src/tools/make_counterfactuals.py` now supports two generation backends:

- `hf` for the old local-model path
- `vllm_openai` for a separate vLLM server

Added CLI flags:

- `--generator-backend`
- `--vllm-base-url`
- `--vllm-api-key`
- `--vllm-model`
- `--generator-concurrency`
- `--generator-batch-size`
- `--candidate-bank`
- `--repair-invalid`
- `--reject-gold-substring`
- `--require-short-answer`
- `--max-overlap-ratio`
- `--max-alt-length-chars`

`src/tools/vllm_cf_client.py` uses the OpenAI-compatible vLLM server with
structured JSON outputs so alternates stay short and explanation-style leakage
is suppressed at generation time.

### DUET candidate bank

`src/tools/build_duet_candidate_bank.py` builds relation-consistent candidates by
grouping rows by `property_pid`, excluding the same `object_qid`, and emitting a
per-row `candidate_answers` list. The generator can then choose or minimally
rewrite candidates instead of free-form inventing long explanations.

### Cleaning and strict validation

`src/tools/clean_counterfactuals.py` applies a dedicated cleanup / repair pass:

- strips prefixes like `Alternative answer:`
- keeps the first answer span
- optionally repairs invalid alternates from the candidate bank
- annotates `cf_invalid_reason` and `cf_is_valid`

`src/tools/validate_dual_cf_artifact.py` now supports stricter checks:

- `--reject-gold-substring`
- `--require-short-answer`
- `--max-alt-length-chars`
- `--check-overlap-ratio`
- `--strict`

The validator also accepts `--input-path` as an alias for `--artifact-path` and
checks optional raw score fields when present.

### Richer artifact schema

The v2 artifact tools keep the trainer-facing keys unchanged:

```json
{
  "index": 17,
  "question": "...",
  "answer": "...",
  "alternate": "...",
  "difficulty_score": 0.63,
  "attribution_score": 0.71
}
```

but now also emit raw routing metadata for offline recalibration:

```json
{
  "difficulty_score_raw": 0.58,
  "attribution_score_raw": 0.014,
  "difficulty_components": {
    "confidence": 0.82,
    "popularity": 0.61,
    "stage_prior": 1.0,
    "stability": 0.0
  },
  "attribution_components": {
    "global_align": 0.12,
    "global_align_cosine": 0.07,
    "local_align": 0.25,
    "local_align_cosine": 0.19,
    "proxy_mode": "template_exact",
    "proxy_key": "...",
    "proxy_size": 12
  }
}
```

### Offline percentile calibration

`src/tools/calibrate_dual_cf_scores.py` converts raw routing scores into
artifact-local percentiles and writes calibrated `difficulty_score` /
`attribution_score` before training. This replaces the earlier assumption that
raw `0.5` thresholds mean the same thing across model families and datasets.

### Difficulty scorer v2

`src/tools/score_difficulty.py` now:

- accepts `--input-path`
- writes `difficulty_score_raw` and `difficulty_components`
- supports `--w-stability`
- supports `--stability-mode prompt_perturb`
- keeps a simple no-model path for popularity-only scoring

Important implementation note:

- this script no longer depends on `trainer.utils` import side effects, so it
  can run in artifact-prep environments that do not have the full training stack
  loaded

### Attribution scorer v2

`src/tools/build_proxy_retain_map.py` builds a syntax-aware proxy map using a
delexicalized question template and a fallback token-overlap search.

`src/tools/score_attribution.py` now supports:

- `--retain-proxy-mode global|template_local|hybrid`
- `--retain-proxy-map`
- `--hybrid-rho`

In hybrid mode, the tool computes:

- one global retain gradient reference
- cached local retain gradients per proxy group
- a mixed raw attribution score used later for percentile calibration

### Trainer updates

`src/trainer/unlearn/dual_cf.py` now adds:

- `alpha_eff_stat`
- `alpha_eff_topk_frac`
- `risk_power`
- `neg_power`

The retain-side batch summary is no longer hard-coded to `risk_gate.mean()`:

```python
risk_batch = self._summarize_risk(risk_gate)
lambda_ret_batch = self.lambda_ret_lo + (
    self.lambda_ret_hi - self.lambda_ret_lo
) * risk_batch
```

Supported retain summaries:

- `mean`
- `p75`
- `max`
- `topk_mean`

The trainer now also logs richer route diagnostics:

- `dualcf_s_p50`
- `dualcf_s_p90`
- `dualcf_r_p50`
- `dualcf_r_p90`
- `dualcf_r_hi_frac`
- `dualcf_risk_batch`

### Trace logging

`src/trainer/callbacks/jsonl_trace.py` appends every trainer log event to
`dualcf_trace.jsonl` in the run directory.

`src/trainer/__init__.py` now auto-registers that callback when
`trace_jsonl: true` is set in the trainer config.

### Config defaults

`configs/trainer/DualCF.yaml` now defaults to the calibrated-score regime:

- `tau_d: 0.6`
- `tau_a: 0.6`
- `temp_d: 0.15`
- `temp_a: 0.15`
- `lambda_ret_hi: 3.0`
- `alpha_eff_stat: topk_mean`
- `alpha_eff_topk_frac: 0.25`
- `trace_jsonl: true`

### Training / ablation scripts

`scripts/duet/dual_cf_duet.sh` and `scripts/rwku/dual_cf_rwku.sh` were upgraded
to support:

- v2 experiment configs by default
- `NUM_EPOCHS=5` by default
- half-epoch checkpoint saving via `CHECKPOINT_EVERY_HALF_EPOCH=1`
- dynamic `save_steps` / `logging_steps` computed from artifact size
- new routing knobs:
  - `ALPHA_EFF_STATS`
  - `ALPHA_EFF_TOPK_FRACS`
  - `RISK_POWERS`
  - `NEG_POWERS`
- method reuse through:
  - `TRAINER`
  - `METHOD_NAME`
  - `RUN_LABEL`
  - `OUTPUT_ROOT`

`scripts/duet/run_dualcf_ablation_v2.sh` and
`scripts/rwku/run_dualcf_ablation_v2.sh` add one launcher entry point for:

- full DualCF
- difficulty-only
- attribution-only
- DPO on the same artifact
- GA / AdaPop / NPO / NPO-SAM / LoKU baseline dispatch

### Checkpoint evaluation

`scripts/duet/eval_checkpoints_duet.sh` and
`scripts/rwku/eval_checkpoints_rwku.sh` evaluate all saved checkpoints plus the
final run directory and then write `checkpoint_evals/summary.tsv` through
`src/tools/summarize_checkpoint_metrics.py`.

### Production runbook

`prod-run-dual-gpu.md` now reflects the intended v2 campaign:

- separate vLLM server
- DUET rare/popular/merged preparation
- RWKU hybrid attribution preparation
- 5-epoch training
- half-epoch checkpoint evaluation
- DUET split-first campaign order

- launcher train commands now pass
  `"cf_dataset_split='${cf_dataset_split}'"`

Why:

- Hydra treats bracket slices like `train[:2]` as grammar unless the value is
  quoted as a string

### Offline artifact tool device placement

Changed file:

- `src/tools/dual_cf_artifact_utils.py`

Fix:

- `load_model_bundle()` now clears inherited `model_args.device_map` before
  loading artifact-tool models

Why:

- attribution scoring with LoRA configs inherited `device_map=auto`, but the
  offline tools manage their own device placement and otherwise hit CPU/GPU
  mismatch errors

### Counterfactual generation semantics

Changed file:

- `src/tools/make_counterfactuals.py`

Fix:

- generator mode now prompts explicitly for a short incorrect alternative answer
- the true answer is included in the prompt
- a stricter retry prompt is used if the first generation still matches the
  true answer

Why:

- RWKU validation exposed rows where `alternate == answer`, showing that the
  previous generator flow was effectively asking the model the original QA task
  instead of requesting a counterfactual

### Offline model subfolder overrides

Changed files:

- `src/tools/dual_cf_artifact_utils.py`
- `src/tools/make_counterfactuals.py`
- `src/tools/score_difficulty.py`
- `src/tools/score_attribution.py`

Fix:

- offline artifact tools now accept `--model-subfolder` and
  `--tokenizer-subfolder`
- `load_model_bundle()` forwards those values into the shared model/tokenizer
  config before loading

Why:

- the DUET SFT weights are published inside the `SwetieePawsss/DUET_ft_models`
  repo under a subfolder, so artifact preparation previously had no clean way
  to address that layout without first resolving a local snapshot path

### Attribution progress visibility and quick caps

Changed file:

- `src/tools/score_attribution.py`

Fix:

- added `tqdm` progress bars for both retain-gradient accumulation and
  forget-gradient scoring
- added `--forget-max-steps` as a symmetric quick cap alongside
  `--retain-max-steps`

Why:

- merged 8B attribution runs were otherwise opaque during execution
- a retain-only cap was not enough for fast verification runs when the user
  wanted a bounded forget-side pass as well

### Artifact validation helper

Added file:

- `src/tools/validate_dual_cf_artifact.py`

Purpose:

- reusable JSONL validation for required keys
- duplicate indices
- empty question / answer / alternate fields
- non-finite score values
- `alternate == answer`
- difficulty / attribution range reporting

## Production alignment (2026-03-13)

Changed files:

- `configs/experiment/unlearn/rwku/dual_cf_lora.yaml`
- `scripts/duet/dual_cf_duet.sh`
- `scripts/rwku/dual_cf_rwku.sh`
- `prod-run-dual-gpu.md`

Updates:

- RWKU DualCF now defaults to the current production instruct stack:
  `Llama-3.1-8B-Instruct` and `Llama-3.1-8B-Instruct-lora`
- DUET and RWKU DualCF launchers now default to the same production surface as
  the other active methods: `PER_DEVICE_TRAIN_BS=16`, `GRAD_ACCUM=2`,
  `NUM_EPOCHS=2`, `EVAL_BATCH_SIZE=64`, and
  `LRS="1e-6 5e-6 1e-5 5e-5 1e-4"`
- Added `prod-run-dual-gpu.md` with merged-only `DUET` artifact generation,
  `RWKU` artifact generation, mandatory `validate_dual_cf_artifact.py` before
  training, and six ready-to-run training launches
- The new runbook keeps full attribution scoring explicit via
  `--retain-max-steps 0` and `--forget-max-steps 0`

## Artifact observability (2026-03-13)

Changed files:

- `src/tools/make_counterfactuals.py`
- `src/tools/score_difficulty.py`
- `src/tools/score_attribution.py`
- `prod-run-dual-gpu.md`

Updates:

- added explicit stage prints for dataset loading, model loading, scoring mode,
  output path, and final score ranges
- added `tqdm` progress bars to `make_counterfactuals.py` and
  `score_difficulty.py`; `score_attribution.py` now also reports the final
  write stage in addition to retain / forget gradient progress
- per-row failures now raise with row position / index context so it is easier
  to identify the broken sample or stage when an artifact build fails
- updated the production DualCF artifact runbook for H100 80GB usage:
  `score_difficulty.py --batch-size 32` and
  `score_attribution.py --retain-batch-size 4`

## Artifact LoRA parity (2026-03-13)

Changed files:

- `src/tools/dual_cf_artifact_utils.py`
- `src/tools/score_attribution.py`
- `prod-run-dual-gpu.md`

Updates:

- `score_attribution.py` now accepts explicit LoRA overrides:
  `--lora-r`, `--lora-alpha`, `--lora-dropout`
- the shared offline model loader now applies those overrides before building
  the temporary LoRA model used for attribution scoring
- the production DualCF runbook now pins attribution scoring to the same LoRA
  shape as the previous production runs:
  `r=32`, `alpha=64`, `dropout=0.0`

## End-to-end wrappers (2026-03-13)

Changed files:

- `dual-scripts-run/run_llama_dual_cf_e2e.sh`
- `dual-scripts-run/run_qwen_dual_cf_e2e.sh`
- `dual-scripts-run/run_gemma_dual_cf_e2e.sh`

Updates:

- added three model-specific wrapper scripts that execute the full DualCF flow
  for each family:
  merged DUET artifact build, DUET validation, DUET training, RWKU artifact
  build, RWKU validation, and RWKU training
- each wrapper accepts GPU and epoch parameters either as positional args or as
  env vars:
  `bash .../run_llama_dual_cf_e2e.sh 4 2` or
  `CUDA_VISIBLE_DEVICES=4 NUM_EPOCHS=2 bash .../run_llama_dual_cf_e2e.sh`
- wrappers reuse the same production defaults as `prod-run-dual-gpu.md` and
  keep artifact LoRA parity with training via `r=32`, `alpha=64`,
  `dropout=0.0`
- wrappers now also support hardware-specific batch profiles:
  `H100` and `L40S`
- current built-in estimates are:
  - `H100`: train batch `16`, grad accum `2`, eval batch `64`,
    difficulty batch `32`, attribution retain batch `4`
  - `L40S`: train batch `8`, grad accum `4`, eval batch `32`,
    difficulty batch `16`, attribution retain batch `2`

## Offline tooling

### `src/tools/make_counterfactuals.py`

Builds `alternate` answers by:

- copying an existing alternate column,
- joining from a JSONL sidecar,
- or generating one alternate answer per sample with a model config.

### `src/tools/score_difficulty.py`

Builds `difficulty_score` using cheap offline proxies:

- optional inverted MRD column,
- popularity normalization,
- model confidence normalization,
- optional stage prior.

### `src/tools/score_attribution.py`

Builds `attribution_score` from a proxy retain bank by:

- averaging retain gradients on trainable params,
- scoring forget gradients by dot-product or cosine alignment,
- clipping negatives to zero,
- min-max normalizing the positive risk signal.

Operational constraint:

- artifact preparation remains an offline stage and should not be moved into
  `DualCF.compute_loss()`, because attribution scoring itself requires extra
  backward passes over a retain bank

## Smoke-test command

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  task_name=duet_dualcf_smoke \
  model=Llama-3.2-1B-Instruct-lora \
  forget_split=city_forget_rare_5 \
  retain_split=city_fast_retain_500 \
  cf_dataset_path=json \
  cf_dataset_data_files=/tmp/duet_rare_dualcf.jsonl \
  "cf_dataset_split=train[:2]" \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/duet_dualcf_smoke
```

Expected smoke-test checks:

- `inputs["forget"]["original"]` exists
- `inputs["forget"]["alternate"]` exists
- `difficulty_score` and `attribution_score` reach `DualCF.compute_loss()`
- `dualcf_*` logs appear
- one adapter checkpoint is saved

## Workspace validation update (2026-03-14)

Changed files:

- `src/tools/vllm_cf_client.py`
- `scripts/duet/prepare_dual_cf_duet_v2.sh`
- `scripts/rwku/prepare_dual_cf_rwku_v2.sh`
- `scripts/duet/eval_checkpoints_duet.sh`
- `scripts/rwku/eval_checkpoints_rwku.sh`
- `src/tools/summarize_checkpoint_metrics.py`
- `prod-run-dual-gpu.md`

Updates:

- the vLLM OpenAI client now sends
  `chat_template_kwargs.enable_thinking=false` for Qwen3 chat requests and
  creates a fresh async client per `asyncio.run(...)` call; this fixed repeated
  generation calls hitting closed-event-loop / connection errors
- the DUET and RWKU prep scripts now support the validated two-phase artifact
  flow:
  - `STOP_AFTER_CLEAN_CF=1`
  - `SKIP_CF_GENERATION=1`
  - `DROP_INVALID_AFTER_CLEAN=1`
- the prep scripts also now expose throughput / bounded-test controls without
  changing production defaults:
  - `DIFFICULTY_BATCH_SIZE`
  - `ATTR_RETAIN_BATCH_SIZE`
  - `ATTR_RETAIN_MAX_STEPS`
  - `ATTR_FORGET_MAX_STEPS`
- `REBUILD_CLEAN_CF=1` was added so saved raw Qwen outputs can be re-cleaned
  with the current strict validator after generation, without restarting vLLM
- DUET prep keeps the empty-string `SFT_SUBFOLDER=` override instead of
  silently falling back to the old TripUnLAMB subfolder when the 1B base model
  is used directly
- checkpoint-eval scripts now skip top-level re-evaluation when the launcher
  already deleted endpoint adapter safetensors, and
  `summarize_checkpoint_metrics.py` now still includes the existing endpoint
  `evals/DUET_SUMMARY.json` in `checkpoint_evals/summary.tsv`
- checkpoint-eval scripts now also delete checkpoint adapter weight files
  (`checkpoint-*/adapter_model.safetensors`, with `.bin` fallback) after
  successful trajectory evaluation unless
  `DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL=0` is set
- the workspace-tested small-model validation runbook now lives in
  `prod-run-dual-vast.md`; `prod-run-dual-gpu.md` was restored to the original
  production-oriented version

Validation results:

- Qwen-first generation was run before any Llama scoring:
  DUET rare, DUET popular, DUET merged, and RWKU raw/clean counterfactual files
  were built with `Qwen/Qwen3-1.7B`; the vLLM server was then stopped and the
  GPU was confirmed free before the Llama stages
- DUET rare full offline artifact succeeded under the 1B test profile:
  `artifacts/dualcf/duet/rare_llama32_1b_v2/dualcf_rare_v2.jsonl`
- DUET popular full offline artifact succeeded after the stricter cleaner
  dropped the last invalid row:
  `artifacts/dualcf/duet/popular_llama32_1b_v2/dualcf_popular_v2.jsonl`
- DUET merged clean revalidation succeeded and reduced the clean set from 964 to
  962 rows; the post-vLLM scoring path was revalidated, but the full merged
  attribution sweep was not waited to completion because it is too slow for the
  workspace validation target
- RWKU raw re-cleaning on the saved Qwen output dropped 1330 invalid rows
  (`2879 -> 1549` clean rows); a bounded 64-row attribution/calibration run then
  completed end to end with strict validation:
  `artifacts/dualcf/rwku/llama32_1b_level2_v2_test64/dualcf_forget_level2_v2.jsonl`
- DUET rare one-epoch training succeeded with the small-model profile and saved
  the expected half-epoch checkpoints `checkpoint-16` and `checkpoint-30`
- DUET rare checkpoint evaluation succeeded after the script fix and produced
  `checkpoint_evals/summary.tsv` including both checkpoints plus the existing
  endpoint eval summary
- RWKU bounded one-epoch training on the 64-row validation artifact succeeded,
  saved `checkpoint-2` and `checkpoint-4`, and finished endpoint evaluation on
  the real `forget_level2` / `neighbor_level2` benchmark splits

Concrete outputs:

- DUET rare training summary:
  `saves/unlearn/duet/dual_cf/.../evals/DUET_SUMMARY.json`
  with `forget_qa_rouge=0.07486168741355462`,
  `holdout_qa_rouge=0.5786666666666667`
- DUET rare checkpoint summary:
  `saves/unlearn/duet/dual_cf/.../checkpoint_evals/summary.tsv`
- RWKU bounded training summary:
  `saves/unlearn/rwku/dual_cf_test64/rwku_Llama-3.2-1B-Instruct_forget_level2_dual_cf_lora_r32_lalpha64_ldrop0p0_lr1e-6_beta0p5_alpha1p0_gamma1p0_td0p6_ta0p6_sd0p15_sa0p15_ln1p0_rlo1p0_rhi3p0_cf1p0_rf0p5_aetopk_mean_atk0p25_rp1p0_np1p0_dOn_aOn/evals/DUET_SUMMARY.json`
  with `forget_qa_rouge=0.3808967459540575`,
  `holdout_qa_rouge=0.437477228978223`

Residual note:

- full RWKU checkpoint-trajectory evaluation was not rerun because it would
  repeat full benchmark eval over all saved checkpoints; the script fix itself
  was validated on DUET and is shared with RWKU

## General LLM utility checkpoint evaluation update (2026-03-14)

Changed files:

- `src/evals/lm_eval.py`
- `configs/eval/lm_eval.yaml`
- `configs/eval/lm_eval_utility_1k.yaml`
- `configs/experiment/eval/utility_1k/default.yaml`
- `configs/lm_eval_tasks/utility_1k/utils.py`
- `configs/lm_eval_tasks/utility_1k/_base_mc.yaml`
- `configs/lm_eval_tasks/utility_1k/utility_mmlu_pro_400.yaml`
- `configs/lm_eval_tasks/utility_1k/utility_truthfulqa_bin_200.yaml`
- `configs/lm_eval_tasks/utility_1k/utility_arc_200.yaml`
- `configs/lm_eval_tasks/utility_1k/utility_winogrande_200.yaml`
- `configs/lm_eval_tasks/utility_1k/_utility_1k.yaml`
- `src/tools/build_utility_1k_panel.py`
- `src/tools/checkpoint_summary_utils.py`
- `src/tools/summarize_checkpoint_metrics.py`
- `src/tools/summarize_utility_metrics.py`
- `src/tools/merge_checkpoint_utility_summaries.py`
- `scripts/utility/eval_checkpoints_utility.sh`
- `scripts/duet/eval_checkpoints_duet.sh`
- `scripts/rwku/eval_checkpoints_rwku.sh`
- `prod-run-dual-vast.md`
- `tests/test_utility_pipeline.py`

Updates:

- added a fixed local `Utility-1K` pipeline built from:
  - `MMLU-Pro 400`
  - `TruthfulQA-Binary 200`
  - `ARC-Challenge 200`
  - `WinoGrande-debiased 200`
- `build_utility_1k_panel.py` freezes those subsets into local JSONL files and
  optionally filters rows that mention forget targets / aliases before the panel
  is written
- the lm-eval wrapper now supports:
  - `include_path` for repo-local custom task directories
  - `include_subtask_metrics` so grouped runs like `utility_1k` still expose the
    per-task accuracies needed for weighted utility summaries
  - forwarding the already-loaded tokenizer and effective batch size into
    `HFLM(...)`, which the old wrapper was not doing
- added a dedicated Utility-1K lm-eval config plus repo-local task YAMLs for the
  fixed panel
- added `scripts/utility/eval_checkpoints_utility.sh`, which evaluates:
  - `base_model_orig`
  - optional `base_model_run`
  - every `checkpoint-*`
  - `final` when top-level weights still exist
- DUET and RWKU checkpoint-eval scripts now optionally call the utility sweep
  through `RUN_UTILITY_EVAL=1` before any checkpoint adapter cleanup happens
- the standard checkpoint summary now normalizes the top-level endpoint label to
  `final` and records `step` plus `epoch`
- added:
  - `checkpoint_evals_utility/summary.tsv`
  - `checkpoint_evals_merged/summary.tsv`
  - `checkpoint_evals_merged/trajectory_metrics.json`
- merged trajectory metrics now report:
  - weighted `utility_avg`
  - `utility_delta_vs_base`
  - utility AUC
  - max utility drawdown
  - best-final gap
  - optional `U@F_tau` when `UTILITY_FORGET_TAU` is set
- the VAST runbook now documents:
  - one-time Utility-1K panel build
  - baseline-cache location
  - DUET / RWKU checkpoint sweep commands with utility enabled for all
    algorithm variants

Validation:

- `bash -n scripts/utility/eval_checkpoints_utility.sh`
- `bash -n scripts/duet/eval_checkpoints_duet.sh`
- `bash -n scripts/rwku/eval_checkpoints_rwku.sh`
- `.venv/bin/python -m py_compile` on the new / changed Python files
- `.venv/bin/python -m unittest discover -s tests -v`
  - this ran a real smoke path that:
    - built a local panel from fixture datasets
    - created a tiny local tokenizer, base model, and LoRA adapters
    - executed `scripts/utility/eval_checkpoints_utility.sh`
    - generated utility and merged checkpoint summaries end to end

Additional DUET launcher updates:

- `scripts/duet/_splits.sh`
  - now honors `FORGET_SPLIT_OVERRIDE`, `RETAIN_SPLIT_OVERRIDE`, and
    `FORGET_LABEL_OVERRIDE` consistently for the ablation runner and all older
    DUET baselines
- `scripts/duet/run_dualcf_ablation_v2.sh`
  - now preserves explicit split overrides instead of overwriting them for
    `rare` / `popular`
  - now dispatches sub-scripts through `bash ...` instead of relying on execute
    bits
- `scripts/duet/ga_duet.sh`
- `scripts/duet/npo_duet.sh`
- `scripts/duet/npo_sam_duet.sh`
- `scripts/duet/loku_duet.sh`
  - fixed Hydra overrides for `trace_jsonl` and optional model args in the older
    scripts
  - when `RUN_UTILITY_EVAL=1`, these launchers now automatically run the
    checkpoint sweep plus Utility-1K after the normal endpoint DUET eval, before
    top-level adapter cleanup
- `scripts/duet/eval_checkpoints_duet.sh`
- `scripts/utility/eval_checkpoints_utility.sh`
  - fixed LoKU utility baselining so `base_model_orig` always uses the original
    pretrained base model, while LoKU checkpoints / final still use the FILA
    residual `base_model` as their adapter base

Additional validation:

- real 1-epoch DUET smoke runs through `scripts/duet/run_dualcf_ablation_v2.sh`
  with `RUN_UTILITY_EVAL=1` succeeded for:
  - `METHOD_VARIANT=ga`
  - `METHOD_VARIANT=npo`
  - `METHOD_VARIANT=npo_sam`
  - `METHOD_VARIANT=loku`
- each smoke run produced:
  - `checkpoint_evals/summary.tsv`
  - `checkpoint_evals_utility/summary.tsv`
  - `checkpoint_evals_merged/summary.tsv`
  - `checkpoint_evals_merged/trajectory_metrics.json`

RWKU compatibility follow-up:

- the older RWKU baselines had the same integration hazards as the older DUET
  baselines:
  - `scripts/rwku/run_dualcf_ablation_v2.sh` relied on execute bits instead of
    dispatching sub-scripts through `bash`
  - `scripts/rwku/ga_rwku.sh`
  - `scripts/rwku/npo_rwku.sh`
  - `scripts/rwku/npo_sam_rwku.sh`
  - `scripts/rwku/loku_rwku.sh`
    - used Hydra overrides that could fail on older configs
    - did not automatically run post-hoc checkpoint eval / Utility-1K when
      `RUN_UTILITY_EVAL=1`
- `scripts/rwku/eval_checkpoints_rwku.sh` also had the same LoKU utility-base
  bug that DUET had:
  - `base_model_orig` could incorrectly use the FILA residual `base_model`
    instead of the original pretrained base model
- these RWKU scripts were patched to match the DUET fixes, but they have not yet
  been validated with the same real multi-method smoke sweep that was run for
  DUET

## Offline production runbook realignment (2026-03-15)

Changed files:

- `prod-run-dual-gpu.md`

Updates:

- the offline GPU runbook now mirrors the validated v2 campaign structure from
  `prod-run-dual-vast.md`, but keeps the production offline roots under
  `/home/vkropoti/...` and `/data/home/vkropoti/...`
- the file is now explicitly scoped to `Llama-3.1-8B-Instruct` with:
  - `NUM_EPOCHS=5`
  - half-epoch checkpoints
  - LoRA parity `r=32`, `alpha=64`, `dropout=0.0`
  - calibrated routing defaults
- Utility-1K is now part of the GPU production runbook end to end:
  - one-time panel build
  - explicit checkpoint plus utility sweep commands
  - merged trajectory outputs
- the artifact flow in the GPU runbook now follows the real two-phase v2 path:
  - DUET clean counterfactual generation first
  - then DUET Llama scoring and calibration
  - merged kept after split-first DUET
  - RWKU kept as phase 2
- unlike the VAST validation runbook, the GPU production runbook now uses
  sequential single-shell command blocks instead of pre-splitting work across
  H100 and L40S devices; hardware profiles stay documented, but device sharding
  is deferred to a later pass

## Train -> Eval -> Cleanup alignment (2026-03-15)

Changed files:

- `scripts/duet/dual_cf_duet.sh`
- `prod-run-dual-gpu.md`

Updates:

- `scripts/duet/dual_cf_duet.sh` now matches the RWKU and older DUET baseline
  launchers:
  - it honors `RUN_CHECKPOINT_EVAL`
  - it falls back to `RUN_UTILITY_EVAL` when `RUN_CHECKPOINT_EVAL` is unset
  - it runs `eval_checkpoints_duet.sh` before any top-level safetensor cleanup
- `prod-run-dual-gpu.md` now defaults the production path to:
  - `RUN_CHECKPOINT_EVAL=1`
  - `RUN_UTILITY_EVAL=1`
  - `DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1`
  - `DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL=1`
- with that default combination, both DUET and RWKU now follow the same
  per-run cadence:
  - training
  - endpoint eval
  - checkpoint eval
  - Utility-1K eval
  - checkpoint adapter cleanup
  - top-level run safetensor cleanup

## Production LR shortlist update (2026-03-15)

Changed files:

- `prod-run-dual-gpu.md`

Updates:

- the offline production runbook now defaults `LRS` to the four-value ablation
  shortlist:
  - `5e-6`
  - `1e-5`
  - `5e-5`
  - `1e-4`
- with the current method set in the runbook
  (`full`, `d_only`, `a_only`, `dpo`, `ga`, `npo`, `npo_sam`, `loku`), this
  means:
  - `32` runs per split block
  - `64` runs for DUET `rare + popular`
  - `96` runs for DUET `rare + popular + merged`
  - `128` runs for the full file including RWKU phase 2

## One-LR campaign wrapper (2026-03-15)

Changed files:

- `scripts/dualcf/run_campaign_one_lr.sh`
- `prod-run-dual-gpu.md`

Updates:

- added `scripts/dualcf/run_campaign_one_lr.sh`, a production wrapper that:
  - takes `GPU_ID` and a single `LR`
  - applies the offline Llama 8B defaults from `prod-run-dual-gpu.md`
  - keeps the automatic per-run cadence
    `train -> eval -> Utility-1K -> cleanup`
  - expects artifacts to be prebuilt under `ARTIFACT_ROOT`
- the wrapper supports phase-gated execution:
  - `duet_rare`
  - `duet_popular`
  - `duet_split_first`
  - `duet_merged`
  - `duet_all`
  - `rwku`
  - `all`
- `prod-run-dual-gpu.md` now documents the balanced four-H100 usage pattern:
  one GPU per LR, with the same phase on all four cards

## vLLM device-order guardrail (2026-03-15)

Changed files:

- `scripts/vllm/start_qwen3_cf_server.sh`

Updates:

- the vLLM launcher now exports `CUDA_DEVICE_ORDER=PCI_BUS_ID` by default
- if `VLLM_CUDA_VISIBLE_DEVICES` is set, the launcher now copies it into
  `CUDA_VISIBLE_DEVICES` itself before starting `vllm`
- the launcher now prints the effective CUDA selection so mixed `L40S` / `H100`
  hosts are easier to verify before model load begins

## Offline dataset-cache alignment for artifact prep (2026-03-15)

Changed files:

- `src/tools/dual_cf_artifact_utils.py`
- `scripts/duet/prepare_dual_cf_duet_v2.sh`
- `scripts/rwku/prepare_dual_cf_rwku_v2.sh`

Updates:

- the shared artifact loader now uses the repo’s cache-aware
  `data.utils.load_hf_dataset(...)` wrapper instead of raw
  `datasets.load_dataset(...)`
- this makes the artifact-prep path honor `HF_DATASETS_CACHE` in the same way
  as the rest of the repo
- the DUET and RWKU prep scripts now also support explicit local offline
  mirrors through:
  - `DUET_DATASET_PATH_LOCAL`
  - `RWKU_DATASET_PATH_LOCAL`
- both prep scripts now print the effective dataset path and
  `HF_DATASETS_CACHE`, so offline dataset resolution is easier to debug on GPU
  boxes

## Offline local-mirror alignment with GA runbooks (2026-03-15)

Changed files:

- `src/tools/dual_cf_artifact_utils.py`
- `scripts/duet/prepare_dual_cf_duet_v2.sh`
- `scripts/rwku/prepare_dual_cf_rwku_v2.sh`

Updates:

- artifact prep now normalizes the common dataset-owner typo
  `SweetieePawsss/... -> SwetieePawsss/...`
- the DUET and RWKU prep scripts now mirror the working production GA setup:
  when a local repo mirror like `${repo_root}/SwetieePawsss/DUET` or
  `${repo_root}/SwetieePawsss/exp_r` exists, the scripts resolve to that local
  path directly instead of trying to reach the Hub in offline mode
- this matches the symlink-based offline layout documented in
  `prod-gpu-runs-new.md`

## Dataset alias fallback for offline v2 artifact prep (2026-03-15)

Changed files:

- `src/tools/dual_cf_artifact_utils.py`
- `scripts/duet/prepare_dual_cf_duet_v2.sh`
- `scripts/rwku/prepare_dual_cf_rwku_v2.sh`

Updates:

- the shared artifact loader no longer treats the first owner alias miss as
  final; it now tries local mirrors first and then falls back across the known
  DUET / RWKU owner aliases for the same dataset suffix
- this covers mixed offline environments where training scripts work with one
  cached alias while the v2 artifact prep receives another alias through
  `DATASET_PATH`
- the prep wrappers now resolve any `*/DUET` or `*/exp_r` input to a real local
  directory when present, and warn when they have to rely on loader-side alias
  fallback instead
- removed a stale `_canonicalize_dataset_path(...)` call left behind by the
  alias-fallback refactor; without this fix, `build_duet_candidate_bank.py`
  failed before dataset loading with a `NameError`

## GPU runbook reduced to four prepare commands (2026-03-15)

Changed files:

- `prod-run-dual-gpu.md`

Updates:

- removed the active training blocks from the lower-half operator command sheet
- the runbook now exposes exactly four preparation commands:
  - DUET Phase A
  - DUET Phase B
  - RWKU Phase A
  - RWKU Phase B
- DUET `rare`, `popular`, and `merged` are all handled inside the DUET Phase
  A/B loops

## Exposed vLLM decoding knobs for artifact prep (2026-03-15)

Changed files:

- `scripts/duet/prepare_dual_cf_duet_v2.sh`
- `scripts/rwku/prepare_dual_cf_rwku_v2.sh`
- `prod-run-dual-gpu.md`

Updates:

- DUET and RWKU Phase A prep now accept generation controls through env vars:
  - `GENERATOR_TEMPERATURE`
  - `GENERATOR_TOP_P`
  - `GENERATOR_MAX_NEW_TOKENS`
- both prep scripts forward those values into `src/tools/make_counterfactuals.py`
- the GPU runbook now shows the recommended tighter defaults for generation:
  `0.2 / 0.8 / 32`
- the underlying CLI default in `src/tools/make_counterfactuals.py` was also
  reduced from `64` to `32` max new tokens so the repo default matches the
  prep-script defaults

## Qwen3.5 vLLM launcher aligned with ttft_bench (2026-03-15)

Changed files:

- `scripts/vllm/start_qwen3_cf_server.sh`
- `prod-run-dual-gpu.md`

Updates:

- the vLLM launcher now matches the working `ttft_bench` serving profile more
  closely for Qwen3.5:
  - `MAX_LEN=4096`
  - `OMP_NUM_THREADS=1`
  - `--trust-remote-code`
  - `--kv-cache-dtype fp8`
  - `--calculate-kv-scales`
  - `--enable-chunked-prefill`
  - `--async-scheduling`
  - `--max-num-batched-tokens 16384`
  - `--max-cudagraph-capture-size 32`
- the runbook vLLM section now uses the same safer defaults and includes the
  launcher call again

## Structured-output stability guardrails for vLLM CF generation (2026-03-15)

Changed files:

- `scripts/vllm/start_qwen3_cf_server.sh`
- `src/tools/vllm_cf_client.py`
- `prod-run-dual-gpu.md`

Updates:

- the vLLM launcher now defaults structured outputs to the `guidance` backend
  instead of leaving backend selection on `auto`, which helps avoid xgrammar
  failures on Qwen3.5 JSON-schema requests
- `async-scheduling` is now disabled by default for this serving path because
  structured-output stability matters more than peak throughput here
- the client no longer aborts the entire generation batch when one response is
  empty or malformed JSON; it now falls back to an invalid row payload so the
  cleaner can drop it and the run can continue
- the client now passes the schema through `extra_body.structured_outputs`
  rather than relying on `response_format` translation in the OpenAI-compatible
  server path
- plain-text vLLM generation is now the default client mode
  (`VLLM_USE_STRUCTURED_OUTPUTS=0`); structured outputs remain opt-in for cases
  where the grammar backend is stable

## Targeted retry path for invalid RWKU counterfactual rows (2026-03-15)

Changed files:

- `src/tools/retry_invalid_counterfactuals.py`
- `scripts/rwku/prepare_dual_cf_rwku_v2.sh`
- `prod-run-dual-gpu.md`

Updates:

- added a retry tool that reloads the raw Phase A JSONL, identifies only the
  rows whose alternates are still invalid under the current cleaner rules, and
  reruns those rows through the vLLM generator
- the tool keeps the first valid retry per row, preserves invalid rows that are
  still bad after all retry passes, and writes the merged raw JSONL back out
- RWKU Phase A now supports this path in-line through:
  - `RETRY_INVALID_CF_PASSES`
  - `RETRY_INVALID_CF_CONCURRENCY`
  - `RETRY_INVALID_CF_BATCH_SIZE`
- the runbook now shows a conservative default retry setup for RWKU:
  `2` passes, concurrency `4`, batch size `16`

## Numeric fallback repair for empty counterfactual rows (2026-03-15)

Changed files:

- `src/tools/fix_numeric_empty_counterfactuals.py`

Updates:

- added a narrow rule-based repair tool for raw counterfactual JSONL artifacts
- it only patches rows whose current alternate is invalid for the specific
  reason `empty`
- when the gold answer is a simple numeric form, the tool fills in a nearby
  deterministic value instead of making another model call
- supported forms are:
  - plain integers
  - decimals
  - decade forms like `2000s`
  - ordinals like `19th`
- patched rows are tagged with:
  - `cf_source=numeric_rule_fallback`
  - `cf_answer_type=numeric_rule_fallback`

## Numeric alternates preserved by shared cleaner (2026-03-15)

Changed files:

- `src/tools/dual_cf_artifact_utils.py`

Updates:

- fixed `clean_counterfactual_text()` so it no longer strips leading digits from
  standalone numeric answers
- the cleaner now removes only actual bullet/list prefixes such as:
  - `- foo`
  - `1. foo`
  - `2) foo`
- numeric answers like `2003`, `19`, `19th`, `3.14`, and `2000s` now survive
  cleaning and validation instead of collapsing to empty strings

## Canonical DUET and RWKU dataset refs restored for offline prep (2026-03-15)

Changed files:

- `scripts/duet/prepare_dual_cf_duet_v2.sh`
- `scripts/rwku/prepare_dual_cf_rwku_v2.sh`
- `src/tools/dual_cf_artifact_utils.py`

Updates:

- known `DUET` and `exp_r` dataset refs are now canonicalized back to
  `SwetieePawsss/DUET` and `SwetieePawsss/exp_r` instead of being rewritten to
  `/data/home/...` mirrors that `datasets.load_dataset()` cannot treat as a
  local dataset script repo
- the shared Python loader now only prefers local paths for these datasets when
  they look like a real `datasets.save_to_disk()` artifact
- typo variants such as `SweetieePawsss/...` and `SweetiePawsss/...` are still
  accepted as input, but they are mapped to the canonical owner before loading

## GPU runbook now resets leaked dataset env before DUET prep (2026-03-15)

Changed files:

- `prod-run-dual-gpu.md`

Updates:

- added the one-time `SwetieePawsss -> /data/home/vkropoti/unlearning/SwetieePawsss`
  symlink step explicitly to the offline runbook
- DUET Phase A and Phase B command blocks now clear leaked RWKU and manual
  dataset overrides before running:
  - `unset FORGET_SPLIT`
  - `unset RETAIN_SPLIT`
  - `unset RWKU_DATASET_PATH_LOCAL`
  - `unset DATASET_PATH`
- the DUET blocks now force the canonical offline dataset reference:
  `DUET_DATASET_PATH_LOCAL=SwetieePawsss/DUET`

## 4x H100 LR-split launch blocks documented in GPU runbook (2026-03-15)

Changed files:

- `prod-run-dual-gpu.md`

Updates:

- added a dedicated training section for `4x H100` using
  `scripts/dualcf/run_campaign_one_lr.sh`
- the runbook now documents the fixed LR-to-GPU mapping:
  - GPU `0` -> `5e-6`
  - GPU `1` -> `1e-5`
  - GPU `2` -> `5e-5`
  - GPU `3` -> `1e-4`
- added ready-to-paste background launch blocks with `wait` for:
  - `duet_rare`
  - `duet_popular`
  - `duet_merged`
  - `rwku`
- replaced file redirection with terminal-tagged launch commands using
  `sed -u` prefixes, so logs stay visible in the terminal
- the runbook now also provides a single `all` phase launch block for running
  the full campaign on `4x H100`

## Campaign wrapper now keeps DUET and RWKU tokenizer paths phase-local (2026-03-15)

Changed files:

- `scripts/dualcf/run_campaign_one_lr.sh`
- `prod-run-dual-gpu.md`

Updates:

- fixed a DUET training bug introduced by exporting `TOKENIZER_MODEL_PATH` as
  the base-model path globally in the campaign wrapper
- DUET phases now explicitly use:
  - `TOKENIZER_MODEL_PATH=${DUET_LOCAL_SFT_BASE}`
  - `TOKENIZER_SUBFOLDER=${DUET_SFT_SUBFOLDER}`
- RWKU phases now explicitly reset back to the base-model tokenizer:
  - `unset USE_SFT_BASE`
  - `unset LOCAL_SFT_BASE`
  - `unset SFT_SUBFOLDER`
  - `unset TOKENIZER_SUBFOLDER`
  - `TOKENIZER_MODEL_PATH=${HF_BASE_MODEL_PATH}`
- removed the global `TOKENIZER_MODEL_PATH=${HF_BASE_MODEL_PATH}` export from
  the GPU runbook common setup, because that override also breaks DUET when
  `USE_SFT_BASE=1`

## DUET checkpoint and utility eval now inherit the SFT subfolder (2026-03-15)

Changed files:

- `scripts/duet/eval_checkpoints_duet.sh`
- `scripts/utility/eval_checkpoints_utility.sh`

Updates:

- fixed a DUET post-endpoint-eval bug where checkpoint eval and Utility-1K eval
  tried to load `SwetieePawsss/DUET_ft_models` as a model root without the
  actual model subfolder
- DUET checkpoint eval now reads:
  - `MODEL_SUBFOLDER` or `SFT_SUBFOLDER`
  - `TOKENIZER_SUBFOLDER`
  and forwards them into `src/eval.py`
- Utility-1K checkpoint eval now supports:
  - `BASE_MODEL_SUBFOLDER`
  - `LORA_BASE_MODEL_SUBFOLDER`
  - `BASE_TOKENIZER_SUBFOLDER`
  - `LORA_TOKENIZER_SUBFOLDER`
- `eval_checkpoints_duet.sh` now passes the DUET SFT subfolder through to the
  utility evaluator, so both endpoint follow-up sweeps load the same base
  checkpoint layout as training and endpoint eval

## Checkpoint and utility eval now force auto device placement (2026-03-15)

Changed files:

- `scripts/duet/eval_checkpoints_duet.sh`
- `scripts/rwku/eval_checkpoints_rwku.sh`
- `scripts/utility/eval_checkpoints_utility.sh`

Updates:

- fixed a major performance bug where checkpoint eval and Utility-1K eval did
  not pass `model.model_args.device_map=auto`
- those follow-up evaluators now also set:
  - `++model.model_args.low_cpu_mem_usage=true`
- before this fix, endpoint eval could run on GPU while checkpoint/utility eval
  silently loaded models on CPU, causing:
  - very slow batch times
  - low or zero visible GPU utilization during checkpoint sweeps

## RWKU manual raw-generation repair workflow is now documented (2026-03-15)

Changed files:

- `prod-run-dual-gpu.md`
- `tmp_rwku_fix.txt`
- `tmp_rwku_apply_manual_fixes.py`
- `tmp_rwku_verify_clean.py`

Updates:

- the RWKU Phase A and Phase B runbook blocks now document the exact
  single-shell commands used on the H100 box with:
  - `CUDA_VISIBLE_DEVICES=1`
  - Phase A generation defaults:
    - `GENERATOR_CONCURRENCY=128`
    - `GENERATOR_BATCH_SIZE=512`
    - `RETRY_INVALID_CF_CONCURRENCY=8`
    - `RETRY_INVALID_CF_BATCH_SIZE=32`
  - Phase B scoring defaults:
    - `DIFFICULTY_BATCH_SIZE=64`
    - `ATTR_RETAIN_BATCH_SIZE=8`
- added a short RWKU repair note to the runbook describing the actual recovery
  path used for bad vLLM generations:
  - raw invalid rows: `464`
  - built-in numeric fallback repaired: `414`
  - remaining manual fixes: `50`
- added root-level temporary helpers for that manual RWKU recovery pass:
  - `tmp_rwku_fix.txt` contains the curated replacements
  - `tmp_rwku_apply_manual_fixes.py` patches the clean artifact and validates it
  - `tmp_rwku_verify_clean.py` checks `raw` vs `clean` vs `final` and fails if
    any repaired rows are still missing or invalid

## DUET H100 prep commands and repair path are now documented (2026-03-15)

Changed files:

- `prod-run-dual-gpu.md`

Updates:

- the DUET Phase A and Phase B runbook blocks now document the exact
  single-shell commands used on the H100 box with:
  - `CUDA_VISIBLE_DEVICES=1`
  - Phase A generation defaults:
    - `GENERATOR_CONCURRENCY=128`
    - `GENERATOR_BATCH_SIZE=512`
  - Phase B scoring defaults:
    - `DIFFICULTY_BATCH_SIZE=64`
    - `ATTR_RETAIN_BATCH_SIZE=8`
- added a DUET-specific vLLM repair note so the runbook does not imply the RWKU
  manual-fix workflow applies there
- the documented DUET recovery path is:
  - build `step0_candidate_bank.jsonl`
  - generate with `--candidate-bank`
  - clean with `--candidate-bank --repair-invalid`
  - enforce strict short-answer / gold-substring / overlap checks
  - drop only the rows that remain invalid after candidate-bank repair
- the runbook also now records the clean-only rebuild path for DUET:
  - `SKIP_CF_GENERATION=1`
  - `REBUILD_CLEAN_CF=1`

## Campaign save checker added (2026-03-16)

Changed files:

- `check_saves.py`

Updates:

- added a root-level helper to verify the expected `run_campaign_one_lr.sh`
  save layout under `saves/` or `saves/unlearn`
- the checker matches the current 4-LR, 4-phase, 8-method campaign shape and
  validates:
  - endpoint eval files under `evals/`
  - checkpoint eval summaries under `checkpoint_evals/`
  - Utility-1K summaries under `checkpoint_evals_utility/`
  - merged checkpoint + utility summaries under `checkpoint_evals_merged/`
- the script exits nonzero if any expected run is missing, duplicated, or has
  missing summary artifacts
- updated the default LR set in the checker to match the intended 4-GPU launch
  block:
  - `5e-6`
  - `1e-5`
  - `5e-5`
  - `1e-4`
- fixed LR matching so GA-style runs that end exactly at `_lr...` are detected
  correctly instead of being reported as both `missing` and `extra`
- fixed an RWKU LoKU naming leak in `run_campaign_one_lr.sh` where DUET's last
  `FORGET_LABEL` value could carry into RWKU and produce legacy run names like
  `rwku_..._merged_loku_...` instead of `rwku_..._forget_level2_loku_...`
- updated the checker to treat those already-written legacy RWKU LoKU run names
  as valid matches for the RWKU LoKU slot
- updated the checker to require cosine-sim artifacts alongside every DUET eval
  directory:
  - `COS_SIM_EVAL.json`
  - `COS_SIM_SUMMARY.json`
  for both top-level `evals/` and each `checkpoint_evals/checkpoint-*`

## Cos-sim eval sweep now works on remote saves trees and checkpoint evals (2026-03-16)

Changed files:

- `scripts/calc_cos_sim.py`

Updates:

- the cos-sim helper now takes `--path_to_saves` instead of assuming the repo
  root contains `saves/`
- it resolves both `.../saves` and `.../saves/unlearn`
- it now scans all `DUET_EVAL.json` files under the provided saves tree,
  including:
  - endpoint evals under `evals/`
  - half-epoch / checkpoint evals under `checkpoint_evals/checkpoint-*`
- each discovered eval directory now receives:
  - `COS_SIM_EVAL.json` with per-example cosine similarities
  - `COS_SIM_SUMMARY.json` with aggregated `*_cos_sim` values alongside the
    existing `DUET_SUMMARY.json`
- the helper now resolves the SBERT encoder from the local Hugging Face cache
  under `HF_HOME` first and also accepts `--sbert_model_path` for explicit
  offline model selection on GPU boxes

## Saves packager now supports explicit paths and summary-only mode (2026-03-16)

Changed files:

- `package_saves.sh`

Updates:

- the saves packager no longer assumes repo-root `./saves`
- it now takes:
  - `--path_to_saves`
  - `--out_path`
  - `--save_eval 0|1`
- `--save_eval 0` keeps only summary artifacts:
  - `*_SUMMARY.json`
  - `trajectory_metrics.json`
  - top-level run `.hydra/config.yaml`
- `--save_eval 1` additionally keeps only endpoint benchmark eval JSON files:
  - `evals/*_EVAL.json`
  - excluding cosine sidecars such as `COS_SIM_EVAL.json`
- the packager no longer keeps:
  - `summary.tsv`
  - eval-sidecar `.hydra/*.yaml`
  - checkpoint `*_EVAL.json`
- the script writes both:
  - the cleaned output directory at `--out_path`
  - the zip archive at `--out_path.zip`

## Structured saves now rebuild from packaged JSON summaries when TSVs are absent (2026-03-24)

Changed files:

- `src/tools/build_structured_saves.py`
- `src/tools/export_unlearning_sanity_checks.py`

Updates:

- `build_structured_saves.py` now falls back to packaged JSON summaries when
  `checkpoint_evals_merged/summary.tsv` is absent
- the fallback rebuilds merged checkpoint rows from:
  - `checkpoint_evals/*/DUET_SUMMARY.json`
  - `checkpoint_evals_utility/*/LMEval_SUMMARY.json`
  - `checkpoint_evals_merged/trajectory_metrics.json` when present
- the params export now tolerates missing `.hydra/overrides.yaml`
- the sanity-check exporter now falls back from `evals/.hydra/config.yaml` to
  the run-level `.hydra/config.yaml`, which matches the slimmer packaged-save
  layout

## Saves-clean footprint helper for `.hydra`, JSON, and TSV (2026-03-24)

Changed files:

- `compare_saves_clean_sizes.sh`

Updates:

- added a small shell helper for packaged `saves-clean` trees
- it accepts `--path_to_saves_clean` and defaults to `./saves-clean`
- it reports file counts plus total bytes for:
  - `.hydra/*`
  - `*.json`
  - `*.tsv`
  - `other` files outside those buckets
- it prints which bucket is larger so packaged-save cleanup decisions can be
  made from actual footprint instead of guesswork

## Structured post-run metric tables for packaged saves (2026-03-16)

Changed files:

- `src/tools/build_structured_saves.py`

Updates:

- added a post-processing helper that reads packaged summary-only saves under
  `saves-clean/unlearn` and writes comparison tables under a separate
  `structured-saves/` tree
- the helper creates `structured-saves/params/` with one YAML file per run that
  bundles:
  - split bucket
  - LR
  - method key
  - override list
  - selected config summary fields
  - the full Hydra config payload
- it also writes `structured-saves/params/params_index.tsv` so the full run list
  can be filtered by:
  - split bucket
  - LR
  - method
  - trainer
  - experiment
- per split bucket (`duet_rare`, `duet_popular`, `duet_merged`, `rwku`) and per
  LR (`5e-6`, `1e-5`, `5e-5`, `1e-4`), the helper now writes:
  - `runs_index.tsv`
  - `epoch_reference.tsv`
  - one TSV per metric with methods on rows and half-epoch slots on columns
  - `trajectory_summary.tsv` for merged trajectory-level utility stats
- method row labels are normalized to the campaign's `METHOD_VARIANT` slots:
  - `full`
  - `d_only`
  - `a_only`
  - `dpo`
  - `ga`
  - `npo`
  - `npo_sam`
  - `loku`
- metric tables currently include:
  - forget / holdout ROUGE
  - forget / holdout cosine similarity
  - `utility_avg`
  - `utility_delta_vs_base`
  - Utility-1K split metrics:
    - `mmlu_pro_400_acc`
    - `truthfulqa_bin_200_acc`
    - `arc_200_acc`
    - `winogrande_200_acc`
- DUET runs keep the nominal half-epoch table columns (`0.5`, `1.0`, ...,
  `5.0`) while `epoch_reference.tsv` records the exact saved epoch values such
  as `0.516129...` and `4.838709...`

Example command:

```bash
python src/tools/build_structured_saves.py \
  --input-root metrics-ep5-all/saves-clean/unlearn \
  --output-root metrics-ep5-all/structured-saves \
  --overwrite
```

## 2026-03-21 Combined Old/New Table Helper

Files:

- `src/tools/build_results_combine_tables.py`

Updates:

- added a helper that reads two `structured-saves/` trees and emits one text
  artifact with combined LaTeX tables
- the helper now emits epoch-specific tables instead of mixing epoch 2 and
  epoch 5 in the same table:
  - `4` splits
  - `2` LRs
  - `2` epochs
  - total: `16` tables / slides
- the helper is meant for old-vs-new DualCF comparison runs where the new run
  overlaps the old method keys (`full`, `d_only`, `a_only`, `dpo`)
- overlapping methods are disambiguated in the output rows as:
  - `Full-old`, `d-only-old`, `a-only-old`, `DPO-old`
  - `Full-new`, `d-only-new`, `a-only-new`, `DPO-new`
- the remaining old-only methods stay in the same table:
  - `GA`
  - `NPO`
  - `NPO-SAM`
  - `LoKU`
- each generated table covers one split/LR pair and includes epoch-2 and
  epoch-5 columns for:
  - forget ROUGE
  - holdout ROUGE
  - forget cosine similarity
  - holdout cosine similarity
  - utility average
  - MMLU-Pro
  - TruthfulQA
  - Winogrande
  - ARC
- the same helper can also emit a Beamer `.tex` deck with one slide per
  split/LR/epoch table, using the same row ordering and metric columns
- when both `--simnpo-root` and `--simplece-old-root` are provided, the helper
  keeps the `SimNPO` row and labels the two SimpleCE sources separately as
  `SimpleCE_new` and `SimpleCE_old` in both the combined tables and the
  SimpleCE-only tables

Example command:

```bash
python src/tools/build_results_combine_tables.py \
  --old-root metrics-ep5-all-v2/structured-saves \
  --new-root metrics-ep5-dualfc-new_cf/structured-saves \
  --simnpo-root metrics-ep5-simnpo_simplece/structured-saves \
  --simplece-old-root metrics-ep5-simplece-oldcf/structured-saves \
  --output-file results-combine/combined_tables.txt \
  --output-slides-tex results-combine/combined_tables_slides.tex \
  --output-simplece-file results-combine/simplece_tables.txt \
  --output-simplece-slides-tex results-combine/simplece_tables_slides.tex
```

## 2026-03-25 Explicit Epoch-2 Checkpoints, Resume Safety, and Seed Plumbing

Files:

- `src/trainer/__init__.py`
- `src/trainer/callbacks/__init__.py`
- `src/trainer/callbacks/save_on_epochs.py`
- `scripts/dualcf/run_campaign_one_lr.sh`
- `scripts/duet/dual_cf_duet.sh`
- `scripts/duet/ga_duet.sh`
- `scripts/duet/npo_duet.sh`
- `scripts/duet/npo_sam_duet.sh`
- `scripts/duet/simnpo_duet.sh`
- `scripts/duet/simple_ce_duet.sh`
- `scripts/duet/ada_pop_duet.sh`
- `scripts/duet/loku_duet.sh`
- `scripts/rwku/dual_cf_rwku.sh`
- `scripts/rwku/ga_rwku.sh`
- `scripts/rwku/npo_rwku.sh`
- `scripts/rwku/npo_sam_rwku.sh`
- `scripts/rwku/simnpo_rwku.sh`
- `scripts/rwku/simple_ce_rwku.sh`
- `scripts/rwku/ada_pop_rwku.sh`
- `scripts/rwku/loku_rwku.sh`
- `scripts/duet/eval_checkpoints_duet.sh`
- `scripts/rwku/eval_checkpoints_rwku.sh`
- `scripts/utility/eval_checkpoints_utility.sh`
- `prod-run-dual-gpu.md`
- `prod-run-dual-vast.md`

Updates:

- replaced launcher-driven half-epoch checkpoint saving with an explicit
  epoch-targeted save path:
  - the wrapper now defaults to:
    - `NUM_EPOCHS=5`
    - `CHECKPOINT_EVERY_HALF_EPOCH=0`
    - `CHECKPOINT_EPOCHS=2`
    - `SAVE_TOTAL_LIMIT=2`
  - intermediate saves are now driven by `trainer.save_on_epochs=[2]`
  - the epoch-5 endpoint remains the normal top-level `trainer.save_model(...)`
    output instead of a duplicate `checkpoint-*`
- added `SaveOnEpochsCallback` and wired it in `load_trainer(...)` alongside
  the existing trace callback registration
- hardened the callback for resume safety:
  - when Trainer state already shows resumed progress, it reconstructs completed
    targets from existing `checkpoint-*/trainer_state.json`
  - fresh reruns into the same `run_dir` do not infer completion from stale
    checkpoints, so `FORCE_RERUN=1` still produces a new epoch-2 save
- updated every DUET / RWKU launcher used by
  `scripts/dualcf/run_campaign_one_lr.sh` so they now:
  - prefer `CHECKPOINT_EPOCHS` over half-epoch `save_steps` math
  - pass `++trainer.args.seed=${TRAIN_SEED}` and
    `++trainer.args.data_seed=${DATA_SEED}`
  - append `RUN_TAG_EXTRA=seed<SEED>` to `task_name`
  - accept optional `FULL_DETERMINISM=1`, which injects
    `++trainer.args.full_determinism=true`
- extended the campaign wrapper so it now:
  - accepts `[SEED]` as a fourth positional argument
  - exports `TRAIN_SEED`, `DATA_SEED`, `PYTHONHASHSEED`,
    `CUBLAS_WORKSPACE_CONFIG`, and `FULL_DETERMINISM`
  - includes the seed tag in LoKU tmp artifact naming as well
- checkpoint / utility sweeps now stop re-evaluating the top-level final
  adapter:
  - final forget / holdout metrics are reused from `run_dir/evals`
  - utility sweeps still evaluate:
    - `base_model_orig`
    - every surviving `checkpoint-*`
    - `final`
  - if top-level final weights were already cleaned, utility sweeps reuse an
    existing `checkpoint_evals_utility/final` result when present instead of
    pretending the last checkpoint is the final model
- added stale-cache cleanup for forced reruns:
  - `eval_checkpoints_duet.sh` and `eval_checkpoints_rwku.sh` now remove:
    - `checkpoint_evals/`
    - `checkpoint_evals_utility/`
    - `checkpoint_evals_merged/`
    when `FORCE_RERUN=1`, so cached `final` utility rows cannot leak into fresh
    runs
- LoKU cleanup guards now treat either half-epoch mode or explicit
  `CHECKPOINT_EPOCHS` as a checkpoint-eval workflow, so FILA artifacts are not
  removed too early

Validation:

- completed locally:
  - `python -m compileall src/trainer src/tools/checkpoint_summary_utils.py src/tools/summarize_checkpoint_metrics.py src/tools/merge_checkpoint_utility_summaries.py`
  - `bash -n` on the wrapper, all touched DUET / RWKU launchers, both
    checkpoint eval scripts, and `scripts/utility/eval_checkpoints_utility.sh`
  - direct callback logic smoke via a file-level import of
    `src/trainer/callbacks/save_on_epochs.py`
    - `1.99 -> False`
    - `2.0 -> True`
    - `4.999 -> False`
    - `5.0 -> False`
  - rerun-vs-resume callback smoke with an existing
    `checkpoint-*/trainer_state.json`
    - `fresh_rerun_at2 -> True`
    - `resume_at2 -> False`
- not completed in this turn:
  - end-to-end DUET or RWKU train / eval smoke
  - multi-seed campaign rerun on real artifacts

Remaining caveat:

- `check_saves.py` was not made seed-aware here. If multiple seeds are written
  into the same parent saves tree, the checker can still treat them as
  duplicates or extras. The lowest-risk workaround is one campaign root per
  seed.
