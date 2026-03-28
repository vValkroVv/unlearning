#!/usr/bin/env bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

METHOD_VARIANT=${METHOD_VARIANT:-full}

case "${METHOD_VARIANT}" in
  full)
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  d_only)
    export DISABLE_ATTRIBUTION_ROUTES=${DISABLE_ATTRIBUTION_ROUTES:-true}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  a_only)
    export DISABLE_DIFFICULTY_ROUTES=${DISABLE_DIFFICULTY_ROUTES:-true}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  dpo)
    export TRAINER=${TRAINER:-DPO}
    export METHOD_NAME=${METHOD_NAME:-dpo_cf}
    export RUN_LABEL=${RUN_LABEL:-DPO}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  simple_ce)
    exec bash "${script_dir}/simple_ce_rwku.sh"
    ;;
  multicf)
    export TRAINER=${TRAINER:-MultiCF}
    export METHOD_NAME=${METHOD_NAME:-multicf}
    export RUN_LABEL=${RUN_LABEL:-MultiCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/multicf_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  boundary_cf)
    export TRAINER=${TRAINER:-BoundaryCF}
    export METHOD_NAME=${METHOD_NAME:-boundary_cf}
    export RUN_LABEL=${RUN_LABEL:-BoundaryCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/boundary_cf_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  span_cf)
    export TRAINER=${TRAINER:-SpanCF}
    export METHOD_NAME=${METHOD_NAME:-span_cf}
    export RUN_LABEL=${RUN_LABEL:-SpanCF}
    export EXPERIMENT=${EXPERIMENT:-unlearn/rwku/span_cf_lora.yaml}
    exec bash "${script_dir}/dual_cf_rwku.sh"
    ;;
  ga)
    exec bash "${script_dir}/ga_rwku.sh"
    ;;
  ada_pop)
    exec bash "${script_dir}/ada_pop_rwku.sh"
    ;;
  npo)
    exec bash "${script_dir}/npo_rwku.sh"
    ;;
  simnpo)
    exec bash "${script_dir}/simnpo_rwku.sh"
    ;;
  npo_sam)
    exec bash "${script_dir}/npo_sam_rwku.sh"
    ;;
  loku)
    exec bash "${script_dir}/loku_rwku.sh"
    ;;
  *)
    echo "[run_dualcf_ablation_v2] Unsupported METHOD_VARIANT=${METHOD_VARIANT}" >&2
    exit 1
    ;;
esac
