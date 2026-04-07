#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo_root"

export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$repo_root/artifacts/dualcf_api_v3}"
export RWKU_DATASET_PATH_LOCAL="${RWKU_DATASET_PATH_LOCAL:-SwetieePawsss/exp_r}"
export OPENAI_MODEL="${OPENAI_MODEL:?set OPENAI_MODEL first}"
export FORGET_SPLIT="${FORGET_SPLIT:-forget_level2}"
export RETAIN_SPLIT="${RETAIN_SPLIT:-neighbor_level2}"
export NUM_ALTERNATES="${NUM_ALTERNATES:-4}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-16}"
export ALLOW_LOW_CONFIDENCE_FALLBACK="${ALLOW_LOW_CONFIDENCE_FALLBACK:-0}"
export OPENAI_TIMEOUT_SECONDS="${OPENAI_TIMEOUT_SECONDS:-180}"
export OPENAI_MAX_ATTEMPTS="${OPENAI_MAX_ATTEMPTS:-3}"

out_dir="$ARTIFACT_ROOT/rwku/${FORGET_SPLIT}_api_v3"
mkdir -p "$out_dir"

echo "=== RWKU ${FORGET_SPLIT} / openai_api ==="
echo "out_dir=${out_dir}"

python scripts/api_cf/generate_external_cf_sidecar.py \
  --backend openai_api \
  --dataset-path "$RWKU_DATASET_PATH_LOCAL" \
  --dataset-name "$FORGET_SPLIT" \
  --split test \
  --question-key query \
  --answer-key answer \
  --output-path "$out_dir/api_sidecar.jsonl" \
  --model "$OPENAI_MODEL" \
  --prompt-family rwku_shared_fact_safe \
  --num-alternates "$NUM_ALTERNATES" \
  --max-examples "$MAX_EXAMPLES" \
  --timeout-seconds "$OPENAI_TIMEOUT_SECONDS" \
  --max-attempts "$OPENAI_MAX_ATTEMPTS" \
  --resume

export FORGET_SPLIT
export RETAIN_SPLIT
export OUT_DIR="$out_dir"
export CF_SIDECAR_JSONL="$out_dir/api_sidecar.jsonl"
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
