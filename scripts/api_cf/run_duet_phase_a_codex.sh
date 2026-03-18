#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo_root"

export ARTIFACT_ROOT="${ARTIFACT_ROOT:-$repo_root/artifacts/dualcf_api_v3}"
export DUET_DATASET_PATH_LOCAL="${DUET_DATASET_PATH_LOCAL:-SwetieePawsss/DUET}"
export CODEX_MODEL="${CODEX_MODEL:-gpt-5.4-mini}"
export CODEX_REASONING_EFFORT="${CODEX_REASONING_EFFORT:-}"
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
export DUET_TARGETS="${DUET_TARGETS:-rare popular merged}"

if [[ "${CODEX_USE_CHATGPT_LOGIN:-1}" == "1" ]]; then
  unset CODEX_API_KEY || true
fi

mkdir -p "$ARTIFACT_ROOT/duet"

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

for label in $DUET_TARGETS; do
  case "$label" in
    rare)
      forget_split="city_forget_rare_5"
      ;;
    popular)
      forget_split="city_forget_popular_5"
      ;;
    merged)
      forget_split="city_forget_rare_5+city_forget_popular_5"
      ;;
    *)
      echo "[run_duet_phase_a_codex] unknown DUET target: $label" >&2
      exit 1
      ;;
  esac
  out_dir="$ARTIFACT_ROOT/duet/${label}_codex_v3${suffix}"

  echo "=== DUET ${label} / codex_cli ==="
  echo "split=${forget_split}"
  echo "out_dir=${out_dir}"

  mkdir -p "$out_dir"

  python src/tools/build_duet_candidate_bank.py \
    --dataset-path "$DUET_DATASET_PATH_LOCAL" \
    --split "$forget_split" \
    --output-path "$out_dir/step0_candidate_bank.jsonl" \
    --question-key question \
    --answer-key answer \
    --candidates-per-row 12 \
    --max-examples "$MAX_EXAMPLES" \
    --sidecar-path "$out_dir/step0_candidate_bank_stats_v3.json"

  if [[ "$SKIP_SIDECAR_GENERATION" == "1" ]]; then
    if [[ ! -s "$out_dir/api_sidecar.jsonl" ]]; then
      echo "[run_duet_phase_a_codex] missing sidecar while SKIP_SIDECAR_GENERATION=1: $out_dir/api_sidecar.jsonl" >&2
      exit 1
    fi
    echo "[run_duet_phase_a_codex] skipping sidecar generation for $label"
  else
    python scripts/api_cf/generate_codex_cf_sidecar.py \
      --dataset-path "$DUET_DATASET_PATH_LOCAL" \
      --split "$forget_split" \
      --question-key question \
      --answer-key answer \
      --candidate-bank "$out_dir/step0_candidate_bank.jsonl" \
      --output-path "$out_dir/api_sidecar.jsonl" \
      --model "$CODEX_MODEL" \
      --prompt-family duet_relation_safe \
      --num-alternates "$NUM_ALTERNATES" \
      --batch-size "$CODEX_BATCH_SIZE" \
      --concurrent "$CODEX_CONCURRENT" \
      --max-examples "$MAX_EXAMPLES" \
      "${reasoning_args[@]}" \
      --timeout-seconds "$CODEX_TIMEOUT_SECONDS" \
      --max-attempts "$CODEX_MAX_ATTEMPTS" \
      --resume
  fi

  if [[ "$STOP_AFTER_SIDECAR" == "1" ]]; then
    echo "[run_duet_phase_a_codex] stop_after_sidecar=1; skipping Phase A prep for $label"
    continue
  fi

  export FORGET_LABEL="$label"
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

  bash scripts/duet/prepare_dual_cf_duet_v3.sh

  python src/tools/clean_counterfactuals.py \
    --input-path "$out_dir/step1_counterfactuals_raw_v3.jsonl" \
    --output-path "$out_dir/step1b_counterfactuals_clean_v3.jsonl" \
    --candidate-bank "$out_dir/step0_candidate_bank.jsonl" \
    --repair-invalid \
    --drop-invalid \
    --reject-gold-substring \
    --require-short-answer \
    --max-overlap-ratio 0.85 \
    --max-alt-length-chars 128 \
    --report-path "$out_dir/step1b_clean_report.json"

  python scripts/api_cf/check_phase_a_outputs.py \
    --dataset duet \
    --out-dir "$out_dir" \
    --question-key question
done
