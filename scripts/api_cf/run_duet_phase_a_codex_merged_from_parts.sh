#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo_root"

export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$repo_root/artifacts/dualcf_api_v3}"
export DUET_DATASET_PATH_LOCAL="${DUET_DATASET_PATH_LOCAL:-SwetieePawsss/DUET}"
export OUT_DIR="${OUT_DIR:-$ARTIFACT_ROOT/duet/merged_codex_v3}"
export RARE_DIR="${RARE_DIR:-$ARTIFACT_ROOT/duet/rare_codex_v3}"
export POPULAR_DIR="${POPULAR_DIR:-$ARTIFACT_ROOT/duet/popular_codex_v3}"
export ALLOW_LOW_CONFIDENCE_FALLBACK="${ALLOW_LOW_CONFIDENCE_FALLBACK:-0}"

mkdir -p "$OUT_DIR"

python scripts/api_cf/build_duet_merged_sidecar_from_parts.py \
  --dataset-path "$DUET_DATASET_PATH_LOCAL" \
  --rare-dir "$RARE_DIR" \
  --popular-dir "$POPULAR_DIR" \
  --output-dir "$OUT_DIR"

merged_rows=$(wc -l < "$OUT_DIR/api_sidecar.jsonl" | tr -d '[:space:]')
merged_data="$OUT_DIR/merged_input.jsonl"
if [[ -z "$merged_rows" || "$merged_rows" == "0" ]]; then
  echo "[run_duet_phase_a_codex_merged_from_parts] merged sidecar is empty: $OUT_DIR/api_sidecar.jsonl" >&2
  exit 1
fi
if [[ ! -f "$merged_data" ]]; then
  echo "[run_duet_phase_a_codex_merged_from_parts] missing synthetic merged dataset: $merged_data" >&2
  exit 1
fi

echo "=== DUET merged / codex_cli / from_parts ==="
echo "rare_dir=$RARE_DIR"
echo "popular_dir=$POPULAR_DIR"
echo "out_dir=$OUT_DIR"
echo "merged_rows=$merged_rows"
echo "merged_data=$merged_data"

export FORGET_LABEL=merged
export FORGET_SPLIT=train
export DATASET_PATH=json
export DATA_FILES="$merged_data"
export CF_SIDECAR_JSONL="$OUT_DIR/api_sidecar.jsonl"
export CF_SIDECAR_ALTERNATE_KEY=alternates
export CF_SIDECAR_SCORE_KEY=scores
export CF_SIDECAR_RELATION_SCORE_KEY=relation_scores
export CF_SIDECAR_SHARED_FACT_SCORE_KEY=shared_fact_scores
export CF_SIDECAR_SOURCE_KEY=candidate_sources
export STOP_AFTER_CLEAN_CF=1
export DROP_INVALID_AFTER_CLEAN=1
export NUM_ALTERNATES="${NUM_ALTERNATES:-4}"
export MAX_EXAMPLES=0
export ALLOW_LOW_CONFIDENCE_FALLBACK
unset SKIP_CF_GENERATION

bash scripts/duet/prepare_dual_cf_duet_v3.sh

python src/tools/clean_counterfactuals.py \
  --input-path "$OUT_DIR/step1_counterfactuals_raw_v3.jsonl" \
  --output-path "$OUT_DIR/step1b_counterfactuals_clean_v3.jsonl" \
  --candidate-bank "$OUT_DIR/step0_candidate_bank.jsonl" \
  --repair-invalid \
  --drop-invalid \
  --reject-gold-substring \
  --require-short-answer \
  --max-overlap-ratio 0.85 \
  --max-alt-length-chars 128 \
  --report-path "$OUT_DIR/step1b_clean_report.json"

python scripts/api_cf/check_phase_a_outputs.py \
  --dataset duet \
  --out-dir "$OUT_DIR" \
  --question-key question
