# Production DualCF GPU Runs (v3 / Idea2)

This is the offline production runbook for the current DualCF v3 / Idea2
campaign.

Scope:

- `Llama-3.1-8B-Instruct` only
- DualCF v3 / Idea2 ablations plus matched baselines
- DUET `rare -> popular -> merged`
- Utility-1K built once and evaluated automatically during each run
- RWKU kept as phase 2 until DUET is stable
- sequential end-to-end shell blocks for one box; no per-GPU sharding in this file yet

Validation status for this runbook refresh:

- completed locally without GPUs:
  - `python -m unittest discover -s tests -p 'test_*.py'` -> `OK (skipped=3)`
- note:
  - the three skipped tests require the optional `lm_eval` package
- not completed for this refresh:
  - GPU-backed vLLM generation checks
  - GPU-backed attribution scoring
  - prep smokes against a live GPU server
  - train / eval runs

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
export UTILITY_ROOT=/data/home/vkropoti/unlearning/evals/utility_1k_v1
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
export CHECKPOINT_EVERY_HALF_EPOCH=1
export SAVE_TOTAL_LIMIT=12
export DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1
export DELETE_CHECKPOINT_ADAPTER_SAFETENSORS_AFTER_EVAL=1

# default production cadence:
# train -> endpoint eval -> checkpoint eval -> Utility-1K -> cleanup -> next run
export RUN_CHECKPOINT_EVAL=1
export RUN_UTILITY_EVAL=1
export EVAL_RUN_BASE_MODEL=0
export UTILITY_EVAL_BATCH_SIZE=64
export UTILITY_APPLY_CHAT_TEMPLATE=true
```

All DUET and RWKU launchers now write task directories directly under
`${OUTPUT_ROOT}` and skip rerunning finished jobs unless `FORCE_RERUN=1`.

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

## Utility-1K panel

Build the fixed general-knowledge panel once per machine and reuse it for every
training run and checkpoint sweep.

```bash
mkdir -p "${UTILITY_ROOT}" "${BASELINE_CACHE_ROOT}"

python src/tools/build_utility_1k_panel.py \
  --output-dir "${UTILITY_ROOT}" \
  --seed 1337 \
  --mmlu-pro 400 \
  --truthfulqa-bin 200 \
  --arc 200 \
  --winogrande 200
```

For DualCF v3 utility-anchor attribution, export the flat QA anchor file in the
same step:

```bash
python src/tools/build_utility_1k_panel.py \
  --output-dir "${UTILITY_ROOT}" \
  --seed 1337 \
  --mmlu-pro 400 \
  --truthfulqa-bin 200 \
  --arc 200 \
  --winogrande 200 \
  --qa-anchor-output-path "${UTILITY_ROOT}/utility_qa_anchor_v3.jsonl" \
  --qa-anchor-truthfulqa-bin 128 \
  --qa-anchor-mmlu-pro 64 \
  --qa-anchor-arc 32 \
  --qa-anchor-winogrande 32
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
export VLLM_USE_STRUCTURED_OUTPUTS=${VLLM_USE_STRUCTURED_OUTPUTS:-1}
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

### DualCF v3 entrypoints

Use the additive v3 prep scripts when you want:

- multi-alternate counterfactual selection
- multi-bank attribution with semantic and utility anchors
- offline belief-bank generation before calibration

Runtime note:

- sidecar mode supports real multi-alternate selection directly
- `vllm_openai` with `NUM_ALTERNATES>1` now fails fast unless
  `VLLM_USE_STRUCTURED_OUTPUTS=1`
- plain-text vLLM remains supported only for the legacy
  `NUM_ALTERNATES=1` path

Common v3 env additions:

```bash
export UTILITY_ANCHOR_JSONL="${UTILITY_ROOT}/utility_qa_anchor_v3.jsonl"
export NUM_ALTERNATES=${NUM_ALTERNATES:-4}
export MAX_EXAMPLES=${MAX_EXAMPLES:-0}
export ALLOW_LOW_CONFIDENCE_FALLBACK=${ALLOW_LOW_CONFIDENCE_FALLBACK:-0}
export BELIEF_MAX_NEW_TOKENS=${BELIEF_MAX_NEW_TOKENS:-16}
export BELIEF_NUM_RETURN_SEQUENCES=${BELIEF_NUM_RETURN_SEQUENCES:-3}
export BELIEF_NUM_BEAMS=${BELIEF_NUM_BEAMS:-4}
```

Leave `PROMPT_FAMILY` unset unless you need to override the script defaults:

- DUET defaults to `duet_relation_safe`
- RWKU defaults to `rwku_shared_fact_safe`

Optional external sidecar wiring for verified multi-alternate CFs:

```bash
export CF_SIDECAR_JSONL=/abs/path/to/counterfactual_sidecar.jsonl
export CF_SIDECAR_ALTERNATE_KEY=alternates
export CF_SIDECAR_SCORE_KEY=scores
export CF_SIDECAR_RELATION_SCORE_KEY=relation_scores
export CF_SIDECAR_SHARED_FACT_SCORE_KEY=shared_fact_scores
export CF_SIDECAR_SOURCE_KEY=candidate_sources
```

Then replace the prep entrypoint:

- DUET: `bash scripts/duet/prepare_dual_cf_duet_v3.sh`
- RWKU: `bash scripts/rwku/prepare_dual_cf_rwku_v3.sh`

The v3 scripts preserve the existing two-phase flow:

- `STOP_AFTER_CLEAN_CF=1` for clean-counterfactual generation only
- `SKIP_CF_GENERATION=1` for score/calibrate-only reuse
- `REBUILD_CLEAN_CF=1` to regenerate the clean JSONL from saved raw CF output

### Local macOS Phase A helpers

For local non-GPU counterfactual generation before any GPU scoring or training,
use the additive helpers under `scripts/api_cf/`. They generate an external
sidecar JSONL first, feed it into the existing v3 prep scripts through
`CF_SIDECAR_JSONL`, stop at `STOP_AFTER_CLEAN_CF=1`, and then verify the saved
`step1*` / `step1b*` files locally.

OpenAI API path:

```bash
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_MODEL=gpt-4.1-mini
export MAX_EXAMPLES=16
export NUM_ALTERNATES=4

bash scripts/api_cf/run_duet_phase_a_openai.sh
bash scripts/api_cf/run_rwku_phase_a_openai.sh
```

Codex CLI path using ChatGPT login instead of API billing:

```bash
unset CODEX_API_KEY
codex login status
# if a real `codex exec` reports a stale refresh token:
# codex logout && codex login

export CODEX_MODEL=gpt-5.4-mini
export CODEX_REASONING_EFFORT=medium
export CODEX_CONCURRENT=1
export MAX_EXAMPLES=16
export NUM_ALTERNATES=4

bash scripts/api_cf/run_duet_phase_a_codex.sh
bash scripts/api_cf/run_rwku_phase_a_codex.sh
```

Those Codex wrappers call `scripts/api_cf/generate_codex_cf_sidecar.py`.
They also accept:

- `CODEX_REASONING_EFFORT=low|medium|high|xhigh`
- `CODEX_CONCURRENT=N` to run up to `N` Codex batch requests in parallel
- `RUN_TAG=...` to isolate outputs for different model runs
- `STOP_AFTER_SIDECAR=1` to stop after writing `api_sidecar.jsonl`
- `SKIP_SIDECAR_GENERATION=1` to reuse an already merged sidecar, including a
  mixed Codex + imported ChatGPT Pro sidecar that records `model=multiple`
- `DUET_TARGETS="rare popular"` to skip direct merged generation during
  multi-model sidecar collection

Both helper families write split-matched artifacts under
`artifacts/dualcf_api_v3/`:

- DUET: `rare_api_v3`, `popular_api_v3`, `merged_api_v3` for OpenAI API, and
  `rare_codex_v3`, `popular_codex_v3`, `merged_codex_v3` for Codex CLI
- RWKU: `${FORGET_SPLIT}_api_v3` or `${FORGET_SPLIT}_codex_v3`

Each output directory should contain:

- `api_sidecar.jsonl`
- `api_sidecar.jsonl.meta.json`
- `api_sidecar.jsonl.summary.json`
- `step1_counterfactuals_raw_v3.jsonl`
- `step1b_counterfactuals_clean_v3.jsonl`
- `step1b_clean_report.json`

If a Codex run is interrupted after `api_sidecar.jsonl` has been appended but
before the launcher finishes, rerunning the same command now bootstraps a
missing `api_sidecar.jsonl.meta.json` from the existing sidecar rows and then
continues the normal coverage check. New runs also write the metadata sidecar
before batch generation starts, so this recovery path is mainly for older or
previously interrupted artifacts.

DUET also writes `step0_candidate_bank.jsonl` and
`step0_candidate_bank_stats_v3.json`.

When the 16-row smoke passes, rerun with:

```bash
export MAX_EXAMPLES=0
```

and keep the same split order: DUET `rare -> popular -> merged`, then RWKU.

If you do not want a separate merged Codex generation pass, DUET now also has a
merged-from-parts helper that rebuilds `merged_codex_v3` from the existing
`rare_codex_v3` and `popular_codex_v3` artifacts:

```bash
bash scripts/api_cf/run_duet_phase_a_codex_merged_from_parts.sh
```

That helper:

- writes `merged_input.jsonl` under the merged artifact directory
- rewrites the combined sidecar indices to synthetic merged-local indices
- runs `scripts/duet/prepare_dual_cf_duet_v3.sh` against `DATASET_PATH=json`
  and `DATA_FILES=.../merged_input.jsonl`
- preserves the standard merged Phase A outputs:
  `step0_candidate_bank.jsonl`, `step1_counterfactuals_raw_v3.jsonl`,
  `step1b_counterfactuals_clean_v3.jsonl`, `step1b_clean_report.json`

This is the correct way to make merged equal rare+popular for partial local
smokes. A raw `cat rare_sidecar + popular_sidecar` is not enough on its own,
because the prep flow joins on local `index` and the official merged split
prefix is not the same as a partial rare+popular union.

For multi-model runs where you want, for example, `2` candidates from each of
`4` different models and then one final `8`-candidate sidecar, first run the
Codex wrappers with `NUM_ALTERNATES=2`, `STOP_AFTER_SIDECAR=1`, and distinct
`RUN_TAG` values, optionally raise `CODEX_CONCURRENT`, then combine the
resulting sidecars with:

```bash
python scripts/api_cf/merge_codex_sidecars.py \
  --input-dir artifacts/dualcf_api_v3/duet/rare_codex_v3__run1 \
  --input-dir artifacts/dualcf_api_v3/duet/rare_codex_v3__run2 \
  --input-dir artifacts/dualcf_api_v3/duet/rare_codex_v3__run3 \
  --input-dir artifacts/dualcf_api_v3/duet/rare_codex_v3__run4 \
  --output-path artifacts/dualcf_api_v3/duet/rare_codex_v3__mix/api_sidecar.jsonl \
  --max-alternates 8
```

Use the same merge helper for `popular` and RWKU. Do not raw-`cat` sidecars.
The Codex generator also shows a `tqdm` batch bar during generation.

The standard train launchers now default to the v3 experiment configs, so
regular DualCF runs do not need an `EXPERIMENT=` override:

- `scripts/duet/dual_cf_duet.sh` -> `unlearn/duet/dual_cf_v3_lora.yaml`
- `scripts/rwku/dual_cf_rwku.sh` -> `unlearn/rwku/dual_cf_v3_lora.yaml`

Phase B also writes artifact-quality reports:

- `${OUT_DIR}/step4_calibration_stats_v3.json` from
  `calibrate_dual_cf_scores.py --sidecar-path ...`, including
  `artifact_quality`
- `${OUT_DIR}/step4_artifact_report_v3.json` from
  `validate_dual_cf_artifact.py --report-path ...`

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
  export OUT_DIR="${ARTIFACT_ROOT}/duet/${FORGET_LABEL}_llama31_8b_v3"
  bash scripts/duet/prepare_dual_cf_duet_v3.sh
done
```

Single-shell form used on the H100 box:

```bash
source /data/home/vkropoti/unlearning-venv/bin/activate && cd /home/vkropoti/diploma/open-unlearning && export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/home/vkropoti/unlearning/.hf_home HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml DUET_LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models DUET_SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft SFT_MODEL_PATH=SwetieePawsss/DUET_ft_models SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft VLLM_BASE_URL=http://127.0.0.1:8000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=/data/home/vkropoti/models/Qwen3.5-27B DUET_DATASET_PATH_LOCAL=SwetieePawsss/DUET DROP_INVALID_AFTER_CLEAN=1 STOP_AFTER_CLEAN_CF=1 GENERATOR_CONCURRENCY=128 GENERATOR_BATCH_SIZE=512 GENERATOR_TEMPERATURE=0.2 GENERATOR_TOP_P=0.8 GENERATOR_MAX_NEW_TOKENS=32 && unset FORGET_SPLIT && unset RETAIN_SPLIT && unset RWKU_DATASET_PATH_LOCAL && unset DATASET_PATH && unset SKIP_CF_GENERATION && for FORGET_LABEL in rare popular merged; do export FORGET_LABEL OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/${FORGET_LABEL}_llama31_8b_v3; bash scripts/duet/prepare_dual_cf_duet_v3.sh; done
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
  export OUT_DIR="${ARTIFACT_ROOT}/duet/${FORGET_LABEL}_llama31_8b_v3"
  bash scripts/duet/prepare_dual_cf_duet_v3.sh
done
```

Single-shell form used on the H100 box:

```bash
source /data/home/vkropoti/unlearning-venv/bin/activate && cd /home/vkropoti/diploma/open-unlearning && export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/home/vkropoti/unlearning/.hf_home HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml DUET_LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models DUET_SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft SFT_MODEL_PATH=SwetieePawsss/DUET_ft_models SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft DUET_DATASET_PATH_LOCAL=SwetieePawsss/DUET DIFFICULTY_BATCH_SIZE=64 ATTR_RETAIN_BATCH_SIZE=8 ATTR_RETAIN_MAX_STEPS=0 ATTR_FORGET_MAX_STEPS=0 SKIP_CF_GENERATION=1 DROP_INVALID_AFTER_CLEAN=1 && unset FORGET_SPLIT && unset RETAIN_SPLIT && unset RWKU_DATASET_PATH_LOCAL && unset DATASET_PATH && unset STOP_AFTER_CLEAN_CF && for FORGET_LABEL in rare popular merged; do export FORGET_LABEL OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/${FORGET_LABEL}_llama31_8b_v3; bash scripts/duet/prepare_dual_cf_duet_v3.sh; done
```

### DUET vLLM Repair Notes

For DUET we used the built-in candidate-bank repair path instead of the RWKU
manual patch workflow.

What we used to keep DUET vLLM generations clean:

1. `scripts/duet/prepare_dual_cf_duet_v3.sh` first builds
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
export OUT_DIR="${ARTIFACT_ROOT}/rwku/llama31_8b_level2_v3"
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

bash scripts/rwku/prepare_dual_cf_rwku_v3.sh
```

Single-shell form used on the H100 box:

```bash
source /data/home/vkropoti/unlearning-venv/bin/activate && cd /home/vkropoti/diploma/open-unlearning && export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/home/vkropoti/unlearning/.hf_home HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct VLLM_BASE_URL=http://127.0.0.1:8000/v1 VLLM_API_KEY=EMPTY VLLM_MODEL=/data/home/vkropoti/models/Qwen3.5-27B FORGET_SPLIT=forget_level2 RETAIN_SPLIT=neighbor_level2 OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku/llama31_8b_level2_v3 DROP_INVALID_AFTER_CLEAN=1 STOP_AFTER_CLEAN_CF=1 GENERATOR_CONCURRENCY=128 GENERATOR_BATCH_SIZE=512 GENERATOR_TEMPERATURE=0.2 GENERATOR_TOP_P=0.8 GENERATOR_MAX_NEW_TOKENS=32 RETRY_INVALID_CF_PASSES=2 RETRY_INVALID_CF_CONCURRENCY=8 RETRY_INVALID_CF_BATCH_SIZE=32 && unset SKIP_CF_GENERATION && bash scripts/rwku/prepare_dual_cf_rwku_v3.sh
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
export OUT_DIR="${ARTIFACT_ROOT}/rwku/llama31_8b_level2_v3"
export DIFFICULTY_BATCH_SIZE=${DIFFICULTY_BATCH_SIZE:-64}
export ATTR_RETAIN_BATCH_SIZE=${ATTR_RETAIN_BATCH_SIZE:-8}
export ATTR_RETAIN_MAX_STEPS=${ATTR_RETAIN_MAX_STEPS:-0}
export ATTR_FORGET_MAX_STEPS=${ATTR_FORGET_MAX_STEPS:-0}
unset STOP_AFTER_CLEAN_CF
export SKIP_CF_GENERATION=1
export DROP_INVALID_AFTER_CLEAN=1

bash scripts/rwku/prepare_dual_cf_rwku_v3.sh
```

Single-shell form used on the H100 box:

```bash
source /data/home/vkropoti/unlearning-venv/bin/activate && cd /home/vkropoti/diploma/open-unlearning && export CUDA_VISIBLE_DEVICES=1 HF_HOME=/data/home/vkropoti/unlearning/.hf_home HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct FORGET_SPLIT=forget_level2 RETAIN_SPLIT=neighbor_level2 OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku/llama31_8b_level2_v3 SKIP_CF_GENERATION=1 DROP_INVALID_AFTER_CLEAN=1 DIFFICULTY_BATCH_SIZE=64 ATTR_RETAIN_BATCH_SIZE=8 ATTR_RETAIN_MAX_STEPS=0 ATTR_FORGET_MAX_STEPS=0 && unset STOP_AFTER_CLEAN_CF && bash scripts/rwku/prepare_dual_cf_rwku_v3.sh
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

This runs on each H100, in order:

- `duet_rare`
- `duet_popular`
- `duet_merged`
- `rwku`

If you want only DUET and no RWKU yet, change `all` to `duet_all` in the same
four commands.
