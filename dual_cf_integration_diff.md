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
- `configs/experiment/unlearn/popqa/dual_cf_lora.yaml`
- `configs/experiment/unlearn/rwku/dual_cf_lora.yaml`

### Launch scripts

- `scripts/duet/dual_cf_duet.sh`
- `scripts/popqa/dual_cf_popqa.sh`
- `scripts/rwku/dual_cf_rwku.sh`
- `scripts/duet/ga_duet.sh`
- `scripts/duet/npo_duet.sh`
- `scripts/duet/npo_sam_duet.sh`
- `scripts/duet/loku_duet.sh`
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
- GA / NPO / NPO-SAM / LoKU baseline dispatch

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
