#!/usr/bin/env bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

METHOD_VARIANT=${METHOD_VARIANT:-full}
FORGET_LABEL=${FORGET_LABEL:-rare}

case "${FORGET_LABEL}" in
  rare)
    export MERGE_POPULARITY_FORGET=0
    export FORGET_SPLIT_OVERRIDE=${FORGET_SPLIT_OVERRIDE:-city_forget_rare_5}
    export RETAIN_SPLIT_OVERRIDE=${RETAIN_SPLIT_OVERRIDE:-city_fast_retain_500}
    export FORGET_LABEL_OVERRIDE=${FORGET_LABEL_OVERRIDE:-${FORGET_SPLIT_OVERRIDE}}
    ;;
  popular)
    export MERGE_POPULARITY_FORGET=0
    export FORGET_SPLIT_OVERRIDE=${FORGET_SPLIT_OVERRIDE:-city_forget_popular_5}
    export RETAIN_SPLIT_OVERRIDE=${RETAIN_SPLIT_OVERRIDE:-city_fast_retain_500}
    export FORGET_LABEL_OVERRIDE=${FORGET_LABEL_OVERRIDE:-${FORGET_SPLIT_OVERRIDE}}
    ;;
  merged)
    export MERGE_POPULARITY_FORGET=1
    ;;
  *)
    echo "[run_dualcf_ablation_v2] Unsupported FORGET_LABEL=${FORGET_LABEL}" >&2
    exit 1
    ;;
esac

case "${METHOD_VARIANT}" in
  full)
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  d_only)
    export DISABLE_ATTRIBUTION_ROUTES=${DISABLE_ATTRIBUTION_ROUTES:-true}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  a_only)
    export DISABLE_DIFFICULTY_ROUTES=${DISABLE_DIFFICULTY_ROUTES:-true}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  dpo)
    export TRAINER=${TRAINER:-DPO}
    export METHOD_NAME=${METHOD_NAME:-dpo_cf}
    export RUN_LABEL=${RUN_LABEL:-DPO}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  simple_ce)
    exec bash "${script_dir}/simple_ce_duet.sh"
    ;;
  multicf)
    export TRAINER=${TRAINER:-MultiCF}
    export METHOD_NAME=${METHOD_NAME:-multicf}
    export RUN_LABEL=${RUN_LABEL:-MultiCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/duet/multicf_lora.yaml}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  boundary_cf)
    export TRAINER=${TRAINER:-BoundaryCF}
    export METHOD_NAME=${METHOD_NAME:-boundary_cf}
    export RUN_LABEL=${RUN_LABEL:-BoundaryCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/duet/boundary_cf_lora.yaml}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  span_cf)
    export TRAINER=${TRAINER:-SpanCF}
    export METHOD_NAME=${METHOD_NAME:-span_cf}
    export RUN_LABEL=${RUN_LABEL:-SpanCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/duet/span_cf_lora.yaml}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  span_cf_simnpo)
    export TRAINER=${TRAINER:-SpanCFSimNPO}
    export METHOD_NAME=${METHOD_NAME:-span_cf_simnpo}
    export RUN_LABEL=${RUN_LABEL:-SpanCFSimNPO}
    export EXPERIMENT=${EXPERIMENT:-unlearn/duet/span_cf_simnpo_lora.yaml}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  span_cf_local_retain)
    export TRAINER=${TRAINER:-SpanCFLocalRetain}
    export METHOD_NAME=${METHOD_NAME:-span_cf_local_retain}
    export RUN_LABEL=${RUN_LABEL:-SpanCFLocalRetain}
    export EXPERIMENT=${EXPERIMENT:-unlearn/duet/span_cf_local_retain_lora.yaml}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  span_cf_simnpo_local_retain)
    export TRAINER=${TRAINER:-SpanCFSimNPOLocalRetain}
    export METHOD_NAME=${METHOD_NAME:-span_cf_simnpo_local_retain}
    export RUN_LABEL=${RUN_LABEL:-SpanCFSimNPOLocalRetain}
    export EXPERIMENT=${EXPERIMENT:-unlearn/duet/span_cf_simnpo_local_retain_lora.yaml}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  span_cf_simnpo_sam)
    export TRAINER=${TRAINER:-SpanCFSimNPOSAM}
    export METHOD_NAME=${METHOD_NAME:-span_cf_simnpo_sam}
    export RUN_LABEL=${RUN_LABEL:-SpanCFSimNPOSAM}
    export EXPERIMENT=${EXPERIMENT:-unlearn/duet/span_cf_simnpo_sam_lora.yaml}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  span_cf_simnpo_projected)
    export TRAINER=${TRAINER:-SpanCFSimNPOProjected}
    export METHOD_NAME=${METHOD_NAME:-span_cf_simnpo_projected}
    export RUN_LABEL=${RUN_LABEL:-SpanCFSimNPOProjected}
    export EXPERIMENT=${EXPERIMENT:-unlearn/duet/span_cf_simnpo_projected_lora.yaml}
    exec bash "${script_dir}/dual_cf_duet.sh"
    ;;
  ga)
    exec bash "${script_dir}/ga_duet.sh"
    ;;
  ada_pop)
    exec bash "${script_dir}/ada_pop_duet.sh"
    ;;
  npo)
    exec bash "${script_dir}/npo_duet.sh"
    ;;
  simnpo)
    exec bash "${script_dir}/simnpo_duet.sh"
    ;;
  npo_sam)
    exec bash "${script_dir}/npo_sam_duet.sh"
    ;;
  loku)
    exec bash "${script_dir}/loku_duet.sh"
    ;;
  *)
    echo "[run_dualcf_ablation_v2] Unsupported METHOD_VARIANT=${METHOD_VARIANT}" >&2
    exit 1
    ;;
esac
