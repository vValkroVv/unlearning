#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo_root"

export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$repo_root/artifacts/dualcf_api_v3}"
export RWKU_DATASET_PATH_LOCAL="${RWKU_DATASET_PATH_LOCAL:-SwetieePawsss/exp_r}"
export CODEX_MODEL="${CODEX_MODEL:-gpt-5.4-mini}"
export CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-}"
export FORGET_SPLIT="${FORGET_SPLIT:-forget_level2}"
export RETAIN_SPLIT="${RETAIN_SPLIT:-neighbor_level2}"
export NUM_ALTERNATES="${NUM_ALTERNATES:-4}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-16}"
export ALLOW_LOW_CONFIDENCE_FALLBACK="${ALLOW_LOW_CONFIDENCE_FALLBACK:-0}"
export CODEX_BATCH_SIZE="${CODEX_BATCH_SIZE:-10}"
export CODEX_CONCURRENT="${CODEX_CONCURRENT:-1}"
export CODEX_TIMEOUT_SECONDS="${CODEX_TIMEOUT_SECONDS:-240}"
export CODEX_MAX_ATTEMPTS="${CODEX_MAX_ATTEMPTS:-3}"
export STOP_AFTER_SIDECAR="${STOP_AFTER_SIDECAR:-0}"
export SKIP_SIDECAR_GENERATION="${SKIP_SIDECAR_GENERATION:-0}"
export RUN_TAG="${RUN_TAG:-}"

if [[ "${CODEX_USE_CHATGPT_LOGIN:-1}" == "1" ]]; then
  unset CODEX_API_KEY || true
fi

parse_key_value_output() {
  local payload="$1"
  local target_name="$2"
  local value=""
  value=$(printf '%s\n' "$payload" | awk -F= -v key="$target_name" '$1 == key {print $2}')
  if [[ -z "$value" ]]; then
    echo "[run_rwku_phase_a_codex] missing ${target_name} in helper output" >&2
    exit 1
  fi
  printf '%s' "$value"
}

safe_run_tag="${RUN_TAG// /_}"
safe_run_tag="${safe_run_tag//\//_}"
safe_run_tag="${safe_run_tag//:/_}"
suffix=""
if [[ -n "$safe_run_tag" ]]; then
  suffix="__${safe_run_tag}"
fi

reasoning_args=()
if [[ -n "$CODEX_REASONING_EFFORT" ]]; then
  reasoning_args=(--reasoning-effort "$CODEX_REASONING_EFFORT")
fi

out_dir="$ARTIFACT_ROOT/rwku/${FORGET_SPLIT}_codex_v3${suffix}"
mkdir -p "$out_dir"
sidecar_path="$out_dir/api_sidecar.jsonl"

echo "=== RWKU ${FORGET_SPLIT} / codex_cli ==="
echo "out_dir=${out_dir}"

resume_info=$(
  python scripts/api_cf/codex_phase_a_resume.py resolve \
    --dataset-path "$RWKU_DATASET_PATH_LOCAL" \
    --dataset-name "$FORGET_SPLIT" \
    --split test \
    --question-key query \
    --answer-key answer \
    --sidecar-path "$sidecar_path" \
    --model "$CODEX_MODEL" \
    --prompt-family rwku_shared_fact_safe \
    --num-alternates "$NUM_ALTERNATES" \
    --requested-max-examples "$MAX_EXAMPLES"
)
printf '%s\n' "$resume_info"

effective_max_examples=$(parse_key_value_output "$resume_info" effective_max_examples)
completed_rows=$(parse_key_value_output "$resume_info" completed_rows)
remaining_rows=$(parse_key_value_output "$resume_info" remaining_rows)

echo "[run_rwku_phase_a_codex] requested_max_examples=${MAX_EXAMPLES} completed_rows=${completed_rows} effective_max_examples=${effective_max_examples} remaining_rows=${remaining_rows}"
export MAX_EXAMPLES="$effective_max_examples"

if [[ "$SKIP_SIDECAR_GENERATION" == "1" ]]; then
  if [[ ! -s "$sidecar_path" ]]; then
    echo "[run_rwku_phase_a_codex] missing sidecar while SKIP_SIDECAR_GENERATION=1: $sidecar_path" >&2
    exit 1
  fi
  echo "[run_rwku_phase_a_codex] skipping sidecar generation"
else
  python scripts/api_cf/generate_codex_cf_sidecar.py \
    --dataset-path "$RWKU_DATASET_PATH_LOCAL" \
    --dataset-name "$FORGET_SPLIT" \
    --split test \
    --question-key query \
    --answer-key answer \
    --output-path "$sidecar_path" \
    --model "$CODEX_MODEL" \
    --prompt-family rwku_shared_fact_safe \
    --num-alternates "$NUM_ALTERNATES" \
    --batch-size "$CODEX_BATCH_SIZE" \
    --concurrent "$CODEX_CONCURRENT" \
    --max-examples "$effective_max_examples" \
    "${reasoning_args[@]}" \
    --timeout-seconds "$CODEX_TIMEOUT_SECONDS" \
    --max-attempts "$CODEX_MAX_ATTEMPTS" \
    --resume
fi

verify_info=$(
  python scripts/api_cf/codex_phase_a_resume.py verify \
    --dataset-path "$RWKU_DATASET_PATH_LOCAL" \
    --dataset-name "$FORGET_SPLIT" \
    --split test \
    --question-key query \
    --answer-key answer \
    --sidecar-path "$sidecar_path" \
    --model "$CODEX_MODEL" \
    --prompt-family rwku_shared_fact_safe \
    --num-alternates "$NUM_ALTERNATES" \
    --effective-max-examples "$effective_max_examples"
)
printf '%s\n' "$verify_info"

if [[ "$STOP_AFTER_SIDECAR" == "1" ]]; then
  echo "[run_rwku_phase_a_codex] stop_after_sidecar=1; skipping Phase A prep"
  exit 0
fi

export FORGET_SPLIT
export RETAIN_SPLIT
export OUT_DIR="$out_dir"
export CF_SIDECAR_JSONL="$sidecar_path"
export CF_SIDECAR_ALTERNATE_KEY=alternates
export CF_SIDECAR_SCORE_KEY=scores
export CF_SIDECAR_RELATION_SCORE_KEY=relation_scores
export CF_SIDECAR_SHARED_FACT_SCORE_KEY=shared_fact_scores
export CF_SIDECAR_SOURCE_KEY=candidate_sources
export STOP_AFTER_CLEAN_CF=1
export DROP_INVALID_AFTER_CLEAN=1
export NUM_ALTERNATES="$NUM_ALTERNATES"
export MAX_EXAMPLES="$MAX_EXAMPLES"
export ALLOW_LOW_CONFIDENCE_FALLBACK="$ALLOW_LOW_CONFIDENCE_FALLBACK"
unset SKIP_CF_GENERATION

bash scripts/rwku/prepare_dual_cf_rwku_v3.sh

python src/tools/clean_counterfactuals.py \
  --input-path "$out_dir/step1_counterfactuals_raw_v3.jsonl" \
  --output-path "$out_dir/step1b_counterfactuals_clean_v3.jsonl" \
  --repair-invalid \
  --drop-invalid \
  --reject-gold-substring \
  --require-short-answer \
  --max-overlap-ratio 0.85 \
  --max-alt-length-chars 128 \
  --report-path "$out_dir/step1b_clean_report.json"

python scripts/api_cf/check_phase_a_outputs.py \
  --dataset rwku \
  --out-dir "$out_dir" \
  --question-key query
