# Production DualCF GPU Runs (v2)

This is the offline production runbook for the current DualCF v2 campaign.

Scope:

- `Llama-3.1-8B-Instruct` only
- DualCF v2 ablations plus matched baselines
- DUET `rare -> popular -> merged`
- Utility-3K built once and evaluated automatically during each run
- RWKU kept as phase 2 until DUET is stable
- sequential end-to-end shell blocks for one box; no per-GPU sharding in this file yet

## Common setup

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

ln -sfn /data/home/vkropoti/unlearning/SwetieePawsss \
  /home/vkropoti/diploma/open-unlearning/SwetieePawsss

export HF_HOME=/data/home/vkropoti/unlearning/.hf_home
export HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache
export TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton
export ARTIFACT_ROOT=/data/home/vkropoti/unlearning/artifacts/dualcf
export OUTPUT_ROOT=/data/home/vkropoti/unlearning/saves/unlearn
export UTILITY=${UTILITY:-3k}
export UTILITY_ROOT=${UTILITY_ROOT:-/data/home/vkropoti/unlearning/evals/utility_3k_v1}
export BASELINE_CACHE_ROOT=/data/home/vkropoti/unlearning/saves/eval/utility_baselines
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" \
  "$ARTIFACT_ROOT" "$OUTPUT_ROOT" "$UTILITY_ROOT" "$BASELINE_CACHE_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Llama 8B production defaults
export BASE_MODEL=Llama-3.1-8B-Instruct
export MODEL_CONFIG=Llama-3.1-8B-Instruct-lora
export MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml
export LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml
export HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct
export BASE_MODEL_PATH=${HF_BASE_MODEL_PATH}
export BASE_MODEL_EVAL_CONFIG=Llama-3.1-8B-Instruct
export LORA_MODEL_EVAL_CONFIG=Llama-3.1-8B-Instruct-lora

# DUET SFT base for DUET artifact prep and DUET runs
export DUET_LOCAL_SFT_BASE=/data/home/vkropoti/unlearning/SwetieePawsss/DUET_ft_models
export DUET_SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft

# production LoRA parity
export LORA_RS=32
export LORA_ALPHAS=64
export LORA_DROPOUTS=0.0

# calibrated DualCF defaults
export TAU_DS=0.6
export TAU_AS=0.6
export TEMP_DS=0.15
export TEMP_AS=0.15
export LAMBDA_RET_HIS=3.0
export ALPHA_EFF_STATS=topk_mean
export ALPHA_EFF_TOPK_FRACS=0.25
export RISK_FORGET_SCALES=0.5

# ablation LR shortlist
export LRS="${LRS:-5e-6 1e-5 5e-5 1e-4}"

# trajectory behavior
export NUM_EPOCHS=5
export CHECKPOINT_EVERY_HALF_EPOCH=0
export CHECKPOINT_EPOCHS=2
export SAVE_TOTAL_LIMIT=2
export DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1
export DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL=1

# default production cadence:
# train -> endpoint eval -> checkpoint eval -> utility panel -> cleanup -> next run
export RUN_CHECKPOINT_EVAL=1
export RUN_UTILITY_EVAL=1
export EVAL_RUN_BASE_MODEL=0
export UTILITY_EVAL_BATCH_SIZE=64
export UTILITY_APPLY_CHAT_TEMPLATE=true
```

All DUET and RWKU launchers now write task directories directly under
`${OUTPUT_ROOT}` and skip rerunning finished jobs unless `FORCE_RERUN=1`.
The campaign wrapper defaults to one intermediate `checkpoint-*` at epoch 2
plus the normal top-level epoch-5 endpoint save.

`scripts/dualcf/run_campaign_one_lr.sh` now includes `ada_pop` in its default
`METHOD_VARIANTS`, so the campaign path covers DualCF plus GA / AdaPop / NPO /
SimNPO / NPO-SAM / LoKU / SimpleCE without a separate wrapper edit. The
standalone AdaPop launchers also accept `BETA_A` and `BETA_B` overrides for the
dynamic popularity curve while keeping the same checkpoint / cleanup cadence as
the other baselines. The wrapper also accepts an optional fourth positional
argument for `SEED`; when set, runs get a matching `_seed<SEED>` suffix. It now
defaults to `UTILITY=3k`; set `UTILITY=1k` to keep the old Utility-1K panel.

## Hardware profile

Pick one profile for the current shell. Keep the H100-sized profile active by
default; switch to the commented L40S profile only if needed. This file does
not split work across devices yet.

```bash
# active train / scoring profile
export PER_DEVICE_TRAIN_BS=32
export GRAD_ACCUM=1
export EVAL_BATCH_SIZE=192
export IMPORTANCE_BATCH_SIZE=32
export DIFFICULTY_BATCH_SIZE=32
export ATTR_RETAIN_BATCH_SIZE=4
export ATTR_RETAIN_MAX_STEPS=0
export ATTR_FORGET_MAX_STEPS=0

# lighter fallback profile
# export PER_DEVICE_TRAIN_BS=16
# export GRAD_ACCUM=2
# export EVAL_BATCH_SIZE=128
# export IMPORTANCE_BATCH_SIZE=16
# export DIFFICULTY_BATCH_SIZE=16
# export ATTR_RETAIN_BATCH_SIZE=2
```

## Utility-3K panel

Build the default general-knowledge panel once per machine and reuse it for
every training run and checkpoint sweep.

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

If you need the older panel for a matched rerun, switch both the selector and
the root:

```bash
export UTILITY=1k
export UTILITY_ROOT=/data/home/vkropoti/unlearning/evals/utility_1k_v1

python src/tools/build_utility_1k_panel.py \
  --output-dir "${UTILITY_ROOT}" \
  --seed 1337 \
  --mmlu-pro 400 \
  --truthfulqa-bin 200 \
  --arc 200 \
  --winogrande 200
```

## vLLM generator

Run the Qwen3.5 counterfactual generator in a separate shell and separate env.
The offline default below uses the local snapshot path directly.

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-vllm-venv/bin/activate

export HF_HOME=/data/home/vkropoti/unlearning/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_MODEL=${VLLM_MODEL:-/data/home/vkropoti/models/Qwen3.5-27B}
export VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-1}
export CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES}
export MODEL=${VLLM_MODEL}
export TP=${TP:-1}
export MAX_LEN=${MAX_LEN:-4096}
export PORT=${PORT:-8000}
export GPU_UTIL=${GPU_UTIL:-0.90}
export DTYPE=${DTYPE:-auto}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
export TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
export ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-1}
export ASYNC_SCHEDULING=${ASYNC_SCHEDULING:-0}
export CALCULATE_KV_SCALES=${CALCULATE_KV_SCALES:-1}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
export MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}
export STRUCTURED_OUTPUTS_BACKEND=${STRUCTURED_OUTPUTS_BACKEND:-guidance}

bash scripts/vllm/start_qwen3_cf_server.sh

```

In the training shell:

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=${VLLM_MODEL:-/data/home/vkropoti/models/Qwen3.5-27B}
export VLLM_USE_STRUCTURED_OUTPUTS=${VLLM_USE_STRUCTURED_OUTPUTS:-0}
```

Build all clean counterfactual files first, then stop the vLLM server before
any Llama scoring or training.

## Artifact prep

Only four commands matter here:

1. DUET Phase A: generate and clean
2. DUET Phase B: score and calibrate
3. RWKU Phase A: generate and clean
4. RWKU Phase B: score and calibrate

For DUET, `merged` is included in the same command and reuses the `rare` and
`popular` split artifacts by default.

### 1. DUET Phase A

Run this while the vLLM server is up.

```bash
export CUDA_VISIBLE_DEVICES=${PREP_CUDA_VISIBLE_DEVICES:-1}
export MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml
export LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml
export SFT_MODEL_PATH=${DUET_LOCAL_SFT_BASE}
export SFT_SUBFOLDER=${DUET_SFT_SUBFOLDER}
export VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}
export VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}
export VLLM_MODEL=${VLLM_MODEL:-/data/home/vkropoti/models/Qwen3.5-27B}
unset FORGET_SPLIT
unset RETAIN_SPLIT
unset RWKU_DATASET_PATH_LOCAL
unset DATASET_PATH
export DUET_DATASET_PATH_LOCAL=SwetieePawsss/DUET
export DROP_INVALID_AFTER_CLEAN=1
export STOP_AFTER_CLEAN_CF=1
export GENERATOR_CONCURRENCY=${GENERATOR_CONCURRENCY:-128}
export GENERATOR_BATCH_SIZE=${GENERATOR_BATCH_SIZE:-512}
export GENERATOR_TEMPERATURE=${GENERATOR_TEMPERATURE:-0.2}
export GENERATOR_TOP_P=${GENERATOR_TOP_P:-0.8}
export GENERATOR_MAX_NEW_TOKENS=${GENERATOR_MAX_NEW_TOKENS:-32}
unset SKIP_CF_GENERATION

for FORGET_LABEL in rare popular merged; do
  export FORGET_LABEL
  export OUT_DIR="${ARTIFACT_ROOT}/duet/${FORGET_LABEL}_llama31_8b_v2"
  bash scripts/duet/prepare_dual_cf_duet_v2.sh
done
```

Single-shell form used on the H100 box:

```bash
source /data/home/vkropoti/unlearning-venv/bin/activate && cd /home/vkropoti/diploma/open-unlearning && export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/home/vkropoti/unlearning/.hf_home HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml DUET_LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models DUET_SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft SFT_MODEL_PATH=SwetieePawsss/DUET_ft_models SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft VLLM_BASE_URL=http://127.0.0.1:8000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=/data/home/vkropoti/models/Qwen3.5-27B DUET_DATASET_PATH_LOCAL=SwetieePawsss/DUET DROP_INVALID_AFTER_CLEAN=1 STOP_AFTER_CLEAN_CF=1 GENERATOR_CONCURRENCY=128 GENERATOR_BATCH_SIZE=512 GENERATOR_TEMPERATURE=0.2 GENERATOR_TOP_P=0.8 GENERATOR_MAX_NEW_TOKENS=32 && unset FORGET_SPLIT && unset RETAIN_SPLIT && unset RWKU_DATASET_PATH_LOCAL && unset DATASET_PATH && unset SKIP_CF_GENERATION && for FORGET_LABEL in rare popular merged; do export FORGET_LABEL OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/${FORGET_LABEL}_llama31_8b_v2; bash scripts/duet/prepare_dual_cf_duet_v2.sh; done
```

### 2. DUET Phase B

Stop the vLLM server before this step.

```bash
export CUDA_VISIBLE_DEVICES=${PREP_CUDA_VISIBLE_DEVICES:-1}
export MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml
export LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml
export SFT_MODEL_PATH=${DUET_LOCAL_SFT_BASE}
export SFT_SUBFOLDER=${DUET_SFT_SUBFOLDER}
unset FORGET_SPLIT
unset RETAIN_SPLIT
unset RWKU_DATASET_PATH_LOCAL
unset DATASET_PATH
export DUET_DATASET_PATH_LOCAL=SwetieePawsss/DUET
export DIFFICULTY_BATCH_SIZE=${DIFFICULTY_BATCH_SIZE:-64}
export ATTR_RETAIN_BATCH_SIZE=${ATTR_RETAIN_BATCH_SIZE:-8}
export ATTR_RETAIN_MAX_STEPS=${ATTR_RETAIN_MAX_STEPS:-0}
export ATTR_FORGET_MAX_STEPS=${ATTR_FORGET_MAX_STEPS:-0}
unset STOP_AFTER_CLEAN_CF
export SKIP_CF_GENERATION=1
export DROP_INVALID_AFTER_CLEAN=1

for FORGET_LABEL in rare popular merged; do
  export FORGET_LABEL
  export OUT_DIR="${ARTIFACT_ROOT}/duet/${FORGET_LABEL}_llama31_8b_v2"
  bash scripts/duet/prepare_dual_cf_duet_v2.sh
done
```

Single-shell form used on the H100 box:

```bash
source /data/home/vkropoti/unlearning-venv/bin/activate && cd /home/vkropoti/diploma/open-unlearning && export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/home/vkropoti/unlearning/.hf_home HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml DUET_LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models DUET_SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft SFT_MODEL_PATH=SwetieePawsss/DUET_ft_models SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft DUET_DATASET_PATH_LOCAL=SwetieePawsss/DUET DIFFICULTY_BATCH_SIZE=64 ATTR_RETAIN_BATCH_SIZE=8 ATTR_RETAIN_MAX_STEPS=0 ATTR_FORGET_MAX_STEPS=0 SKIP_CF_GENERATION=1 DROP_INVALID_AFTER_CLEAN=1 && unset FORGET_SPLIT && unset RETAIN_SPLIT && unset RWKU_DATASET_PATH_LOCAL && unset DATASET_PATH && unset STOP_AFTER_CLEAN_CF && for FORGET_LABEL in rare popular merged; do export FORGET_LABEL OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/${FORGET_LABEL}_llama31_8b_v2; bash scripts/duet/prepare_dual_cf_duet_v2.sh; done
```

### DUET vLLM Repair Notes

For DUET we used the built-in candidate-bank repair path instead of the RWKU
manual patch workflow.

What we used to keep DUET vLLM generations clean:

1. `scripts/duet/prepare_dual_cf_duet_v2.sh` first builds
   `step0_candidate_bank.jsonl` with relation-consistent candidates.
2. `make_counterfactuals.py` receives that bank through `--candidate-bank`, so
   generation is constrained to same-relation alternatives.
3. `clean_counterfactuals.py` reruns with:
   - `--candidate-bank`
   - `--repair-invalid`
   - `--reject-gold-substring`
   - `--require-short-answer`
   - `--max-overlap-ratio 0.85`
   - `--max-alt-length-chars 128`
4. `DROP_INVALID_AFTER_CLEAN=1` removes any rows still invalid after that bank
   repair step.
5. If we need to rebuild clean artifacts without rerunning vLLM generation, we
   use:
   - `SKIP_CF_GENERATION=1`
   - `REBUILD_CLEAN_CF=1`

### 3. RWKU Phase A

Start the vLLM server again before this step.

```bash
export CUDA_VISIBLE_DEVICES=${PREP_CUDA_VISIBLE_DEVICES:-1}
export MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml
export LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml
export BASE_MODEL_PATH=${HF_BASE_MODEL_PATH}
export VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}
export VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}
export VLLM_MODEL=${VLLM_MODEL:-/data/home/vkropoti/models/Qwen3.5-27B}
export FORGET_SPLIT=forget_level2
export RETAIN_SPLIT=neighbor_level2
export OUT_DIR="${ARTIFACT_ROOT}/rwku/llama31_8b_level2_v2"
export DROP_INVALID_AFTER_CLEAN=1
export STOP_AFTER_CLEAN_CF=1
export GENERATOR_CONCURRENCY=${GENERATOR_CONCURRENCY:-128}
export GENERATOR_BATCH_SIZE=${GENERATOR_BATCH_SIZE:-512}
export GENERATOR_TEMPERATURE=${GENERATOR_TEMPERATURE:-0.2}
export GENERATOR_TOP_P=${GENERATOR_TOP_P:-0.8}
export GENERATOR_MAX_NEW_TOKENS=${GENERATOR_MAX_NEW_TOKENS:-32}
export RETRY_INVALID_CF_PASSES=${RETRY_INVALID_CF_PASSES:-2}
export RETRY_INVALID_CF_CONCURRENCY=${RETRY_INVALID_CF_CONCURRENCY:-8}
export RETRY_INVALID_CF_BATCH_SIZE=${RETRY_INVALID_CF_BATCH_SIZE:-32}
unset SKIP_CF_GENERATION

bash scripts/rwku/prepare_dual_cf_rwku_v2.sh
```

Single-shell form used on the H100 box:

```bash
source /data/home/vkropoti/unlearning-venv/bin/activate && cd /home/vkropoti/diploma/open-unlearning && export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/home/vkropoti/unlearning/.hf_home HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct VLLM_BASE_URL=http://127.0.0.1:8000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=/data/home/vkropoti/models/Qwen3.5-27B FORGET_SPLIT=forget_level2 RETAIN_SPLIT=neighbor_level2 OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku/llama31_8b_level2_v2 DROP_INVALID_AFTER_CLEAN=1 STOP_AFTER_CLEAN_CF=1 GENERATOR_CONCURRENCY=128 GENERATOR_BATCH_SIZE=512 GENERATOR_TEMPERATURE=0.2 GENERATOR_TOP_P=0.8 GENERATOR_MAX_NEW_TOKENS=32 RETRY_INVALID_CF_PASSES=2 RETRY_INVALID_CF_CONCURRENCY=8 RETRY_INVALID_CF_BATCH_SIZE=32 && unset SKIP_CF_GENERATION && bash scripts/rwku/prepare_dual_cf_rwku_v2.sh
```

### 4. RWKU Phase B

Stop the vLLM server before this step.

```bash
export CUDA_VISIBLE_DEVICES=${PREP_CUDA_VISIBLE_DEVICES:-1}
export MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml
export LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml
export BASE_MODEL_PATH=${HF_BASE_MODEL_PATH}
export FORGET_SPLIT=forget_level2
export RETAIN_SPLIT=neighbor_level2
export OUT_DIR="${ARTIFACT_ROOT}/rwku/llama31_8b_level2_v2"
export DIFFICULTY_BATCH_SIZE=${DIFFICULTY_BATCH_SIZE:-64}
export ATTR_RETAIN_BATCH_SIZE=${ATTR_RETAIN_BATCH_SIZE:-8}
export ATTR_RETAIN_MAX_STEPS=${ATTR_RETAIN_MAX_STEPS:-0}
export ATTR_FORGET_MAX_STEPS=${ATTR_FORGET_MAX_STEPS:-0}
unset STOP_AFTER_CLEAN_CF
export SKIP_CF_GENERATION=1
export DROP_INVALID_AFTER_CLEAN=1

bash scripts/rwku/prepare_dual_cf_rwku_v2.sh
```

Single-shell form used on the H100 box:

```bash
source /data/home/vkropoti/unlearning-venv/bin/activate && cd /home/vkropoti/diploma/open-unlearning && export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/home/vkropoti/unlearning/.hf_home HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct FORGET_SPLIT=forget_level2 RETAIN_SPLIT=neighbor_level2 OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku/llama31_8b_level2_v2 SKIP_CF_GENERATION=1 DROP_INVALID_AFTER_CLEAN=1 DIFFICULTY_BATCH_SIZE=64 ATTR_RETAIN_BATCH_SIZE=8 ATTR_RETAIN_MAX_STEPS=0 ATTR_FORGET_MAX_STEPS=0 && unset STOP_AFTER_CLEAN_CF && bash scripts/rwku/prepare_dual_cf_rwku_v2.sh
```

### RWKU vLLM Repair Notes

For the current RWKU run, raw vLLM generation produced `464` invalid rows:

- `416` `empty`
- `5` `exact_match`
- `43` `gold_substring`

What we used to recover the artifact:

1. The built-in numeric fallback already repaired `414` rows during raw -> clean.
2. The remaining `50` rows were fixed manually with:
   - `tmp_rwku_fix.txt`
   - `tmp_rwku_apply_manual_fixes.py`
3. The clean artifact was checked with:
   - `python tmp_rwku_verify_clean.py --out-dir "${OUT_DIR}" --preview 50`
4. After the manual fixes, Phase B was rerun with `SKIP_CF_GENERATION=1` so
   `step2_*`, `step3_*`, and the final `dualcf_*.jsonl` were rebuilt from the
   patched clean file.
5. Final success means:
   - `clean_rows=2879`
   - `final_rows=2879`
   - `clean_invalid_rows=0`
   - `final_invalid_rows=0`
   - `verification=passed`

## Training on 4x H100

Use one LR per H100:

- GPU `0`: `5e-6`
- GPU `1`: `1e-5`
- GPU `2`: `5e-5`
- GPU `3`: `1e-4`

Run the full campaign with terminal logs and a tag prefix per process:

```bash
bash -lc 'set -o pipefail; bash scripts/dualcf/run_campaign_one_lr.sh 0 5e-6 all 2>&1 | sed -u "s/^/[gpu0 lr=5e-6 all] /"' &
bash -lc 'set -o pipefail; bash scripts/dualcf/run_campaign_one_lr.sh 1 1e-5 all 2>&1 | sed -u "s/^/[gpu1 lr=1e-5 all] /"' &
bash -lc 'set -o pipefail; bash scripts/dualcf/run_campaign_one_lr.sh 2 5e-5 all 2>&1 | sed -u "s/^/[gpu2 lr=5e-5 all] /"' &
bash -lc 'set -o pipefail; bash scripts/dualcf/run_campaign_one_lr.sh 3 1e-4 all 2>&1 | sed -u "s/^/[gpu3 lr=1e-4 all] /"' &
wait
```

Omitting the fourth positional arg keeps the default `SEED=42`. For explicit
multi-seed runs, call
`bash scripts/dualcf/run_campaign_one_lr.sh GPU LR PHASE SEED`.
The wrapper also supports serial multi-seed execution through
`SEEDS="42 43" bash scripts/dualcf/run_campaign_one_lr.sh GPU LR PHASE`.

This runs on each H100, in order:

- `duet_rare`
- `duet_popular`
- `duet_merged`
- `rwku`

If you want only DUET and no RWKU yet, change `all` to `duet_all` in the same
four commands.

## Additional SimpleCE Runs

Also ran these standalone `simple_ce` campaigns:

Default launcher tuple for these runs is now:

- `CF_WEIGHTS=0.5`
- `RETAIN_WEIGHTS=1`
- `GAMMAS=0`

```bash
METHOD_VARIANTS=simple_ce bash scripts/dualcf/run_campaign_one_lr.sh 0 1e-4 all

METHOD_VARIANTS=simple_ce bash scripts/dualcf/run_campaign_one_lr.sh 0 5e-5 all

METHOD_VARIANTS=simple_ce \
CF_WEIGHTS="0.5 1 2" \
RETAIN_WEIGHTS=1 \
GAMMAS="0 1 2" \
bash scripts/dualcf/run_campaign_one_lr.sh 5 5e-5 all

METHOD_VARIANTS=simple_ce \
CF_WEIGHTS="0.5 1 2" \
RETAIN_WEIGHTS=1 \
GAMMAS="0 1 2" \
bash scripts/dualcf/run_campaign_one_lr.sh 4 1e-4 all
```

## Additional AdaPop Runs

Also ran these standalone `ada_pop` campaigns:

```bash
METHOD_VARIANTS=ada_pop bash scripts/dualcf/run_campaign_one_lr.sh 0 1e-4 all

METHOD_VARIANTS=ada_pop bash scripts/dualcf/run_campaign_one_lr.sh 0 5e-5 all

METHOD_VARIANTS=ada_pop \
GAMMAS="1.0" \
ALPHA_CONST="none 0.5 1.0" \
BETA_CONST="none 0.5 1.0" \
bash scripts/dualcf/run_campaign_one_lr.sh 5 5e-5 all

METHOD_VARIANTS=ada_pop \
GAMMAS="1.0" \
ALPHA_CONST="none 0.5 1.0" \
BETA_CONST="none 0.5 1.0" \
bash scripts/dualcf/run_campaign_one_lr.sh 4 1e-4 all
```

## Post-run cosine-sim sweep

After the campaign finishes, write cosine-sim artifacts alongside every saved
`DUET_EVAL.json` under `${OUTPUT_ROOT}`:

```bash
python scripts/calc_cos_sim.py \
  --path_to_saves "${OUTPUT_ROOT}" \
  --gpu "${COS_SIM_CUDA_VISIBLE_DEVICES:-0}"
```

## Package clean saves

Package the summary-only saves tree into a local clean directory plus zip:

```bash
bash package_saves.sh \
  --path_to_saves "${OUTPUT_ROOT%/unlearn}" \
  --out_path /home/vkropoti/diploma/open-unlearning/saves-clean \
  --save_eval 0
```

## Multi-seed Example

Run the same campaign serially for multiple seeds:

```bash
SEEDS="42 43" bash scripts/dualcf/run_campaign_one_lr.sh 0 1e-4 all
```

If you want only the full DualCF variant for each seed:

```bash
SEEDS="42 43" METHOD_VARIANTS=full bash scripts/dualcf/run_campaign_one_lr.sh 0 1e-4 all
```

## Planned 6 Runs

```bash
# 1. Baselines first, lr=1e-4, GPU 2
SEEDS="42 179 1137" METHOD_VARIANTS="ga npo simnpo npo_sam loku" bash scripts/dualcf/run_campaign_one_lr.sh 2 1e-4 all

# 2. Baselines first, lr=5e-5, GPU 1
SEEDS="42 179 1137" METHOD_VARIANTS="ga npo simnpo npo_sam loku" bash scripts/dualcf/run_campaign_one_lr.sh 1 5e-5 all

# 3. Old artifacts, lr=1e-4, GPU 2
SEEDS="42 179 1137" METHOD_VARIANTS="full d_only a_only dpo simple_ce" bash scripts/dualcf/run_campaign_one_lr.sh 2 1e-4 all

# 4. Old artifacts, lr=5e-5, GPU 1
SEEDS="42 179 1137" METHOD_VARIANTS="full d_only a_only dpo simple_ce" bash scripts/dualcf/run_campaign_one_lr.sh 1 5e-5 all

# 5. New artifacts, lr=1e-4, GPU 2
SEEDS="42 179 1137" METHOD_VARIANTS="full d_only a_only dpo simple_ce" bash scripts/dualcf/run_campaign_one_lr.sh 2 1e-4 all

# 6. New artifacts, lr=5e-5, GPU 1
SEEDS="42 179 1137" METHOD_VARIANTS="full d_only a_only dpo simple_ce" bash scripts/dualcf/run_campaign_one_lr.sh 1 5e-5 all
```

## New Method Runs

First-pass new-method ablations should freeze the shared routed DualCF core and
vary only the method-specific knobs. Keep `tau_d`, `tau_a`, `temp_d`,
`temp_a`, `lambda_neg_max`, `lambda_ret_lo`, `lambda_ret_hi`, `cf_weight`,
`risk_forget_scale`, and the difficulty / attribution routing flags fixed.

Each command below runs the requested `1e-4` / `all` ablations serially for one
method. MultiCF is split across two GPUs and uses a reduced counterfactual
count to avoid the H100 OOM seen with the original `k=6` / `k=8` plan.
New-method output dirs now abbreviate method tags in the launcher:
MultiCF uses `agwm`/`agm`/`agt1` and `wrr`/`wuni`,
BoundaryCF uses `lr`/`bm`,
SpanCF uses `mlc`/`mso` plus `sw`/`uw`.
These shorter tags keep run dirs under filesystem name limits.
Each loop continues to the next spec if one spec fails.

```bash
# MultiCF GPU 0: M1, M2, M4
GPU_ID=1
for spec in \
  "M1|2|weighted_mean|rerank|0.7" \
  "M2|2|weighted_mean|rerank|0.5" \
  "M4|3|weighted_mean|rerank|0.7"
do
  IFS='|' read -r tag max_alts agg_mode weight_mode set_temp <<<"${spec}"
  echo "[multicf] ${tag}: k=${max_alts} agg=${agg_mode} weight=${weight_mode} temp=${set_temp}"
  if SEEDS="42 179 1137" \
    METHOD_VARIANTS="multicf" \
    MULTICF_MAX_ALTERNATES_USED="${max_alts}" \
    MULTICF_ALT_AGG_MODE="${agg_mode}" \
    MULTICF_ALT_WEIGHT_MODE="${weight_mode}" \
    MULTICF_ALT_SET_TEMPERATURE="${set_temp}" \
    bash scripts/dualcf/run_campaign_one_lr.sh "${GPU_ID}" 1e-4 all
  then
    echo "[multicf] ${tag} done"
  else
    echo "[multicf] ${tag} failed, continuing to next spec"
  fi
done

# MultiCF GPU 1: M3, M5, M6
GPU_ID=2
for spec in \
  "M5|3|weighted_mean|rerank|0.5" \
  "M3|2|weighted_mean|rerank|1.0" \
  "M6|2|mean|uniform|1.0"
do
  IFS='|' read -r tag max_alts agg_mode weight_mode set_temp <<<"${spec}"
  echo "[multicf] ${tag}: k=${max_alts} agg=${agg_mode} weight=${weight_mode} temp=${set_temp}"
  if SEEDS="42 179 1137" \
    METHOD_VARIANTS="multicf" \
    MULTICF_MAX_ALTERNATES_USED="${max_alts}" \
    MULTICF_ALT_AGG_MODE="${agg_mode}" \
    MULTICF_ALT_WEIGHT_MODE="${weight_mode}" \
    MULTICF_ALT_SET_TEMPERATURE="${set_temp}" \
    bash scripts/dualcf/run_campaign_one_lr.sh "${GPU_ID}" 1e-4 all
  then
    echo "[multicf] ${tag} done"
  else
    echo "[multicf] ${tag} failed, continuing to next spec"
  fi
done

# BoundaryCF: B1-B6
GPU_ID=4
for spec in \
  "B1|0.75|0.5" \
  "B2|1.0|0.5" \
  "B3|0.5|0.5" \
  "B4|0.5|1.0" \
  "B5|0.75|1.0" \
  "B6|1.0|1.0"
do
  IFS='|' read -r tag local_retain margin <<<"${spec}"
  echo "[boundary_cf] ${tag}: local_retain=${local_retain} margin=${margin}"
  if SEEDS="42 179 1137" \
    METHOD_VARIANTS="boundary_cf" \
    BOUNDARY_LOCAL_RETAIN_WEIGHT="${local_retain}" \
    BOUNDARY_MARGIN_WEIGHT="${margin}" \
    bash scripts/dualcf/run_campaign_one_lr.sh "${GPU_ID}" 1e-4 all
  then
    echo "[boundary_cf] ${tag} done"
  else
    echo "[boundary_cf] ${tag} failed, continuing to next spec"
  fi
done

# SpanCF: S1-S6
GPU_ID=5
for spec in \
  "S1|lcs|0.10|1.25" \
  "S2|lcs|0.10|1.0" \
  "S3|lcs|0.25|1.25" \
  "S4|lcs|0.0|1.0" \
  "S5|lcs|0.25|1.0" \
  "S6|set_overlap|0.10|1.25"
do
  IFS='|' read -r tag span_mode shared_weight unique_weight <<<"${spec}"
  echo "[span_cf] ${tag}: mode=${span_mode} shared=${shared_weight} unique=${unique_weight}"
  if SEEDS="42 179 1137" \
    METHOD_VARIANTS="span_cf" \
    SPAN_MODE="${span_mode}" \
    SPAN_SHARED_TOKEN_WEIGHT="${shared_weight}" \
    SPAN_UNIQUE_TOKEN_WEIGHT="${unique_weight}" \
    bash scripts/dualcf/run_campaign_one_lr.sh "${GPU_ID}" 1e-4 all
  then
    echo "[span_cf] ${tag} done"
  else
    echo "[span_cf] ${tag} failed, continuing to next spec"
  fi
done
```

Interpret `boundary_cf` DUET-popular results carefully: the current artifact is
mostly fallback hard negatives there, not true lexical near-miss boundaries.
