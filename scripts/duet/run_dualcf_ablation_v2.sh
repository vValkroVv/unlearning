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
  ga)
    exec bash "${script_dir}/ga_duet.sh"
    ;;
  npo)
    exec bash "${script_dir}/npo_duet.sh"
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
