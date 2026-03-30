# Production DualCF GPU Runs (v2)

This runbook is for the DualCF v2 iteration:

- clean counterfactuals first
- percentile-calibrate routing offline
- use hybrid retain attribution
- run DUET `rare -> popular -> merged`
- save / evaluate an explicit epoch-2 checkpoint plus the final endpoint

The validation profile below is the workspace-tested small-model setup:

- main model: `meta-llama/Llama-3.2-1B-Instruct`
- LoRA config: `configs/model/Llama-3.2-1B-Instruct-lora.yaml`
- vLLM generator: `Qwen/Qwen3-1.7B`
- epochs: `1`

After validation, switch the env vars back to your production values
(`Llama-3.1-8B-Instruct`, DUET SFT base if needed, larger Qwen3 generator,
`NUM_EPOCHS=5`).

## Common setup

Use explicit repo / data roots so the same commands work both here and on the
production box.

- workspace validation:
  - `REPO_ROOT=/workspace/unlearning`
  - `DATA_ROOT=/workspace/data/unlearning`
- production box:
  - `REPO_ROOT=/home/vkropoti/unlearning`
  - `DATA_ROOT=/data/home/vkropoti/unlearning`

```bash
export REPO_ROOT=${REPO_ROOT:-/workspace/unlearning}
export DATA_ROOT=${DATA_ROOT:-/workspace/data/unlearning}
export VENV_PATH=${VENV_PATH:-${REPO_ROOT}/.venv}
export VLLM_VENV_PATH=${VLLM_VENV_PATH:-${REPO_ROOT}/.venv-vllm}

cd "${REPO_ROOT}"
source "${VENV_PATH}/bin/activate"

export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in the shell first}
export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}

export HF_HOME=${DATA_ROOT}/.hf_home
export HF_DATASETS_CACHE=${DATA_ROOT}/.hf_datasets_cache
export TRITON_CACHE_DIR=${DATA_ROOT}/.triton
export ARTIFACT_ROOT=${DATA_ROOT}/artifacts/dualcf
export OUTPUT_ROOT=${DATA_ROOT}/saves/unlearn
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" \
  "${ARTIFACT_ROOT}" "${OUTPUT_ROOT}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

export BASE_MODEL=Llama-3.2-1B-Instruct
export MODEL_CONFIG=Llama-3.2-1B-Instruct-lora
export MODEL_CFG=configs/model/Llama-3.2-1B-Instruct.yaml
export LORA_MODEL_CFG=configs/model/Llama-3.2-1B-Instruct-lora.yaml
export HF_BASE_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct
export BASE_MODEL_PATH=${HF_BASE_MODEL_PATH}
export SFT_MODEL_PATH=${HF_BASE_MODEL_PATH}
export SFT_SUBFOLDER=
export USE_SFT_BASE=0

export VLLM_MODEL=Qwen/Qwen3-1.7B

export PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-2}
export NUM_EPOCHS=${NUM_EPOCHS:-1}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
export DELETE_MODEL_SAFETENSORS_AFTER_EVAL=${DELETE_MODEL_SAFETENSORS_AFTER_EVAL:-1}
export CHECKPOINT_EVERY_HALF_EPOCH=${CHECKPOINT_EVERY_HALF_EPOCH:-0}
export CHECKPOINT_EPOCHS=${CHECKPOINT_EPOCHS:-2}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
export UTILITY=${UTILITY:-3k}
export UTILITY_ROOT=${UTILITY_ROOT:-${REPO_ROOT}/artifacts/evals/utility_3k_v1}
export BASELINE_CACHE_ROOT=${BASELINE_CACHE_ROOT:-${DATA_ROOT}/saves/eval/utility_baselines}
export RUN_UTILITY_EVAL=${RUN_UTILITY_EVAL:-1}
export EVAL_RUN_BASE_MODEL=${EVAL_RUN_BASE_MODEL:-0}
export UTILITY_EVAL_BATCH_SIZE=${UTILITY_EVAL_BATCH_SIZE:-16}
export UTILITY_APPLY_CHAT_TEMPLATE=${UTILITY_APPLY_CHAT_TEMPLATE:-true}
export BASE_MODEL_EVAL_CONFIG=${BASE_MODEL_EVAL_CONFIG:-${BASE_MODEL}}
export LORA_MODEL_EVAL_CONFIG=${LORA_MODEL_EVAL_CONFIG:-${MODEL_CONFIG}}
export UTILITY_FORGET_TAU=${UTILITY_FORGET_TAU:-}

export LRS="${LRS:-1e-5}"
export TAU_DS="${TAU_DS:-0.5}"
export TAU_AS="${TAU_AS:-0.5}"
export TEMP_DS="${TEMP_DS:-0.2}"
export TEMP_AS="${TEMP_AS:-0.2}"
export RISK_FORGET_SCALES="${RISK_FORGET_SCALES:-0.5}"
export LAMBDA_RET_HIS="${LAMBDA_RET_HIS:-3.0}"
export ALPHA_EFF_STATS="${ALPHA_EFF_STATS:-topk_mean}"
export ALPHA_EFF_TOPK_FRACS="${ALPHA_EFF_TOPK_FRACS:-0.25}"

export GENERATOR_CONCURRENCY=${GENERATOR_CONCURRENCY:-4}
export GENERATOR_BATCH_SIZE=${GENERATOR_BATCH_SIZE:-8}
export DIFFICULTY_BATCH_SIZE=${DIFFICULTY_BATCH_SIZE:-8}
export ATTR_RETAIN_BATCH_SIZE=${ATTR_RETAIN_BATCH_SIZE:-4}
export ATTR_RETAIN_MAX_STEPS=${ATTR_RETAIN_MAX_STEPS:-0}
export ATTR_FORGET_MAX_STEPS=${ATTR_FORGET_MAX_STEPS:-0}
```

`OUTPUT_ROOT` is respected by:

- `scripts/duet/dual_cf_duet.sh`
- `scripts/rwku/dual_cf_rwku.sh`
- `scripts/duet/ada_pop_duet.sh`
- `scripts/duet/ga_duet.sh`
- `scripts/duet/npo_duet.sh`
- `scripts/duet/npo_sam_duet.sh`
- `scripts/duet/loku_duet.sh`
- `scripts/rwku/ada_pop_rwku.sh`
- `scripts/rwku/ga_rwku.sh`
- `scripts/rwku/npo_rwku.sh`
- `scripts/rwku/npo_sam_rwku.sh`
- `scripts/rwku/loku_rwku.sh`

So the run directories land under:

- `${OUTPUT_ROOT}/<task_name>`

The direct AdaPop launchers now use the same train -> endpoint eval ->
checkpoint eval -> utility panel -> cleanup flow as the other DUET / RWKU
baselines. If you need to override the dynamic popularity curve shape, export
`BETA_A` and `BETA_B` before launching `scripts/duet/ada_pop_duet.sh` or
`scripts/rwku/ada_pop_rwku.sh`.

For `LoKU`, importance files also move out of the repo and default to:

- `${DATA_ROOT}/saves/importances/duet/loku`
- `${DATA_ROOT}/saves/importances/rwku/loku`

Half-epoch note for the `NUM_EPOCHS=1` validation profile:

Explicit checkpoint note for the `NUM_EPOCHS=1` validation profile:

- keep `MAX_STEPS=0`
- keep `CHECKPOINT_EPOCHS=2`
- expect only the final run directory at `NUM_EPOCHS=1`
- when you switch back to `NUM_EPOCHS=5`, expect one intermediate
  `checkpoint-*` around epoch 2 plus the final run directory

## Utility-3K panel

Build the default general-knowledge panel once per machine and reuse it for every
method, seed, and checkpoint run.

```bash
mkdir -p "${UTILITY_ROOT}" "${BASELINE_CACHE_ROOT}"

python src/tools/build_utility_1k_panel.py \
  --output-dir "${UTILITY_ROOT}" \
  --seed 1337 \
  --mmlu-pro 1200 \
  --truthfulqa-bin 600 \
  --arc 600 \
  --winogrande 600 \
  --arc-split test
```

If you already have a forget-target alias file, rebuild the panel with:

```bash
python src/tools/build_utility_1k_panel.py \
  --output-dir "${UTILITY_ROOT}" \
  --seed 1337 \
  --mmlu-pro 1200 \
  --truthfulqa-bin 600 \
  --arc 600 \
  --winogrande 600 \
  --arc-split test \
  --exclude-targets-file /data/home/vkropoti/unlearning/evals/forget_target_aliases.txt
```

For matched legacy reruns, switch back explicitly:

```bash
export UTILITY=1k
export UTILITY_ROOT=${REPO_ROOT}/artifacts/evals/utility_1k_v1

python src/tools/build_utility_1k_panel.py \
  --output-dir "${UTILITY_ROOT}" \
  --seed 1337 \
  --mmlu-pro 400 \
  --truthfulqa-bin 200 \
  --arc 200 \
  --winogrande 200
```

## vLLM generator

Run the Qwen3 generator in a separate env, build all Qwen-dependent
counterfactual files first, then stop the server before any Llama scoring or
training.

```bash
cd "${REPO_ROOT}"
source "${VLLM_VENV_PATH}/bin/activate"

export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in the shell first}
export HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}
export HF_HOME=${DATA_ROOT}/.hf_home

export CUDA_VISIBLE_DEVICES=0
export MODEL=${VLLM_MODEL}
export TP=1
export MAX_LEN=2048
export PORT=8000
export GPU_UTIL=0.25

scripts/vllm/start_qwen3_cf_server.sh
```

In the training shell:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=Qwen/Qwen3-1.7B
```

After all `step1b_counterfactuals_clean.jsonl` files are ready, stop the vLLM
server to free GPU memory before the Llama scoring / training steps.

## Artifact prep

Each prep script now supports a two-phase flow:

- `STOP_AFTER_CLEAN_CF=1`:
  build / clean the Qwen counterfactual file only
- `SKIP_CF_GENERATION=1`:
  reuse the cleaned counterfactual file and run the Llama scoring stages
- `REBUILD_CLEAN_CF=1`:
  rebuild `step1b_counterfactuals_clean.jsonl` from the saved raw Qwen output
  before scoring

`DROP_INVALID_AFTER_CLEAN=1` is the default and should be left on for the
validation profile. This drops rows that still fail strict alternate-answer
validation after repair.

### Phase A: Qwen clean counterfactuals only

Run these while the vLLM server is up.

### DUET rare

```bash
export CUDA_VISIBLE_DEVICES=0
export FORGET_LABEL=rare
export OUT_DIR=${ARTIFACT_ROOT}/duet/rare_llama32_1b_v2
export STOP_AFTER_CLEAN_CF=1
unset SKIP_CF_GENERATION

scripts/duet/prepare_dual_cf_duet_v2.sh
```

### DUET popular

```bash
export CUDA_VISIBLE_DEVICES=0
export FORGET_LABEL=popular
export OUT_DIR=${ARTIFACT_ROOT}/duet/popular_llama32_1b_v2
export STOP_AFTER_CLEAN_CF=1
unset SKIP_CF_GENERATION

scripts/duet/prepare_dual_cf_duet_v2.sh
```

### DUET merged

Run this only after rare and popular are clean.

```bash
export CUDA_VISIBLE_DEVICES=0
export FORGET_LABEL=merged
export OUT_DIR=${ARTIFACT_ROOT}/duet/merged_llama32_1b_v2
export STOP_AFTER_CLEAN_CF=1
unset SKIP_CF_GENERATION

scripts/duet/prepare_dual_cf_duet_v2.sh
```

### RWKU

```bash
export CUDA_VISIBLE_DEVICES=0
export FORGET_SPLIT=forget_level2
export RETAIN_SPLIT=neighbor_level2
export OUT_DIR=${ARTIFACT_ROOT}/rwku/llama32_1b_level2_v2
export STOP_AFTER_CLEAN_CF=1
unset SKIP_CF_GENERATION

scripts/rwku/prepare_dual_cf_rwku_v2.sh
```

After all `step1b_counterfactuals_clean.jsonl` files are written, stop the vLLM
server and confirm the GPU is free before continuing.

### Phase B: Llama scoring / calibration only

Run the same prep commands again with the generator stage disabled:

```bash
unset STOP_AFTER_CLEAN_CF
export SKIP_CF_GENERATION=1
export DROP_INVALID_AFTER_CLEAN=1
```

Then rerun the same four commands above for:

- DUET rare
- DUET popular
- DUET merged
- RWKU

The scoring stages use:

- `DIFFICULTY_BATCH_SIZE` for `score_difficulty.py`
- `ATTR_RETAIN_BATCH_SIZE` for `score_attribution.py`
- `ATTR_RETAIN_MAX_STEPS` / `ATTR_FORGET_MAX_STEPS` only for bounded local
  validation; leave both at `0` for production artifacts

## DUET training

### Full DualCF

Rare:

```bash
export CUDA_VISIBLE_DEVICES=0
export CF_DATASET_DATA_FILES=${ARTIFACT_ROOT}/duet/rare_llama32_1b_v2/dualcf_rare_v2.jsonl
export METHOD_VARIANT=full
export FORGET_LABEL=rare
export MAX_STEPS=0

scripts/duet/run_dualcf_ablation_v2.sh
```

Popular:

```bash
export CUDA_VISIBLE_DEVICES=0
export CF_DATASET_DATA_FILES=${ARTIFACT_ROOT}/duet/popular_llama32_1b_v2/dualcf_popular_v2.jsonl
export METHOD_VARIANT=full
export FORGET_LABEL=popular
export MAX_STEPS=0

scripts/duet/run_dualcf_ablation_v2.sh
```

Merged:

```bash
export CUDA_VISIBLE_DEVICES=0
export CF_DATASET_DATA_FILES=${ARTIFACT_ROOT}/duet/merged_llama32_1b_v2/dualcf_merged_v2.jsonl
export METHOD_VARIANT=full
export FORGET_LABEL=merged
export MAX_STEPS=0

scripts/duet/run_dualcf_ablation_v2.sh
```

### Required ablations

Difficulty-only:

```bash
export METHOD_VARIANT=d_only
scripts/duet/run_dualcf_ablation_v2.sh
```

Attribution-only:

```bash
export METHOD_VARIANT=a_only
scripts/duet/run_dualcf_ablation_v2.sh
```

Uniform counterfactual DPO:

```bash
export METHOD_VARIANT=dpo
scripts/duet/run_dualcf_ablation_v2.sh
```

Baselines:

```bash
export METHOD_VARIANT=ga
scripts/duet/run_dualcf_ablation_v2.sh

export METHOD_VARIANT=npo
scripts/duet/run_dualcf_ablation_v2.sh

export METHOD_VARIANT=npo_sam
scripts/duet/run_dualcf_ablation_v2.sh

export METHOD_VARIANT=loku
scripts/duet/run_dualcf_ablation_v2.sh
```

SpanCF family (new):

```bash
# SpanCF with asymmetric 4-token weights
export METHOD_VARIANT=span_cf
export SPAN_MODE=lcs
export SPAN_ALT_SHARED_TOKEN_WEIGHT=0.0
export SPAN_ALT_UNIQUE_TOKEN_WEIGHT=1.0
export SPAN_ORIG_SHARED_TOKEN_WEIGHT=0.10
export SPAN_ORIG_UNIQUE_TOKEN_WEIGHT=1.0
scripts/duet/run_dualcf_ablation_v2.sh

# SpanCFSimNPO
export METHOD_VARIANT=span_cf_simnpo
export SPAN_SIMNPO_DELTA=0.0
scripts/duet/run_dualcf_ablation_v2.sh

# SpanCFSimNPO + local retain
# Local-retain variants must use the merged artifact path, not dualcf_*.jsonl.
export CF_DATASET_DATA_FILES=${ARTIFACT_ROOT}/duet/rare_llama32_1b_v2/span_local_retain_rare_v1.jsonl
export METHOD_VARIANT=span_cf_simnpo_local_retain
export SPAN_LOCAL_RETAIN_WEIGHT=0.2
export SPAN_BOUNDARY_MARGIN_WEIGHT=0.0
scripts/duet/run_dualcf_ablation_v2.sh

# SpanCFSimNPO + SAM
export CF_DATASET_DATA_FILES=${ARTIFACT_ROOT}/duet/rare_llama32_1b_v2/dualcf_rare_v2.jsonl
export METHOD_VARIANT=span_cf_simnpo_sam
export SPAN_SAM_RHO=0.01
export SPAN_SAM_ADAPTIVE=false
scripts/duet/run_dualcf_ablation_v2.sh

# SpanCFSimNPO + projected conflict handling
export METHOD_VARIANT=span_cf_simnpo_projected
export SPAN_PROJECTION_COS_THRESHOLD=0.0
scripts/duet/run_dualcf_ablation_v2.sh
```

Every method variant above uses the same trajectory-saving behavior:

- one intermediate `checkpoint-*` save when `CHECKPOINT_EPOCHS=2` and
  `NUM_EPOCHS >= 2`
- endpoint eval into `run_dir/evals`
- training trace in `run_dir/dualcf_trace.jsonl`
- top-level adapter safetensor cleanup after endpoint eval
- the same post-hoc checkpoint evaluator for forget/locality plus the selected utility panel
  when `RUN_UTILITY_EVAL=1`

## RWKU training

```bash
export CUDA_VISIBLE_DEVICES=0
export CF_DATASET_DATA_FILES=${ARTIFACT_ROOT}/rwku/llama32_1b_level2_v2/dualcf_forget_level2_v2.jsonl
export METHOD_VARIANT=full
export MAX_STEPS=0

scripts/rwku/run_dualcf_ablation_v2.sh
```

Run the same ablation variants on RWKU after DUET rare / popular are stable.
For RWKU local-retain variants, set:
`CF_DATASET_DATA_FILES=${ARTIFACT_ROOT}/rwku/llama32_1b_level2_v2/span_local_retain_forget_level2_v1.jsonl`.

## Checkpoint evaluation

DUET:

```bash
RUN_UTILITY_EVAL=1 scripts/duet/eval_checkpoints_duet.sh \
  /path/to/run_dir \
  city_forget_rare_5 \
  city_fast_retain_500 \
  ${HF_BASE_MODEL_PATH} \
  ${HF_BASE_MODEL_PATH} \
  ${LORA_MODEL_EVAL_CONFIG} \
  ${BASE_MODEL_EVAL_CONFIG}
```

For LoKU, the checkpoint evaluator auto-detects `run_dir/base_model` and uses
it instead of the original base path. If you want a second utility baseline on
that LoKU base model, set `EVAL_RUN_BASE_MODEL=1`. To clean the FILA base after
trajectory eval:

```bash
RUN_UTILITY_EVAL=1 EVAL_RUN_BASE_MODEL=1 DELETE_RUN_BASE_MODEL_AFTER_EVAL=1 \
scripts/duet/eval_checkpoints_duet.sh \
  /path/to/loku_run_dir \
  city_forget_rare_5 \
  city_fast_retain_500 \
  ${HF_BASE_MODEL_PATH} \
  ${HF_BASE_MODEL_PATH} \
  ${LORA_MODEL_EVAL_CONFIG} \
  ${BASE_MODEL_EVAL_CONFIG}
```

RWKU:

```bash
RUN_UTILITY_EVAL=1 scripts/rwku/eval_checkpoints_rwku.sh \
  /path/to/run_dir \
  forget_level2 \
  neighbor_level2 \
  ${HF_BASE_MODEL_PATH} \
  ${HF_BASE_MODEL_PATH} \
  ${LORA_MODEL_EVAL_CONFIG} \
  ${BASE_MODEL_EVAL_CONFIG}
```

For LoKU on RWKU, the same cleanup pattern works:

```bash
RUN_UTILITY_EVAL=1 EVAL_RUN_BASE_MODEL=1 DELETE_RUN_BASE_MODEL_AFTER_EVAL=1 \
scripts/rwku/eval_checkpoints_rwku.sh \
  /path/to/loku_run_dir \
  forget_level2 \
  neighbor_level2 \
  ${HF_BASE_MODEL_PATH} \
  ${HF_BASE_MODEL_PATH} \
  ${LORA_MODEL_EVAL_CONFIG} \
  ${BASE_MODEL_EVAL_CONFIG}
```

Each script writes:

- per-checkpoint eval folders under `checkpoint_evals/`
- `checkpoint_evals/summary.tsv`
- per-checkpoint utility eval folders under `checkpoint_evals_utility/`
- `checkpoint_evals_utility/summary.tsv`
- `checkpoint_evals_merged/summary.tsv`
- `checkpoint_evals_merged/trajectory_metrics.json`

If the top-level run directory was already cleaned by
`DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1`, the checkpoint-eval scripts skip
reloading that endpoint adapter and reuse the existing `run_dir/evals`
summary instead. Utility evaluation will reuse an existing
`checkpoint_evals_utility/final` result when present; otherwise the final
utility row is skipped because the epoch-2 checkpoint is not treated as a
final-model proxy.

By default, the checkpoint-eval scripts also delete
`checkpoint-*/adapter_model.safetensors` after successful trajectory eval.
Set `DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL=0` if you need to keep
them for a rerun.

`UTILITY_FORGET_TAU` is optional. If you set it before the checkpoint sweep,
the merged trajectory JSON will also report `U@F_tau` at the nearest matched
forget score.

## Campaign order

1. Start vLLM and run all prep scripts with `STOP_AFTER_CLEAN_CF=1`.
2. Stop vLLM and confirm GPU memory is free.
3. Rerun all prep scripts with `SKIP_CF_GENERATION=1` for Llama scoring / calibration.
4. Train DUET rare full + ablations.
5. Train DUET popular, then merged.
6. Run RWKU training / ablations.
