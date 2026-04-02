#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash package_saves.sh --path_to_saves PATH --out_path PATH [--save_eval 0|1]

Examples:
  bash package_saves.sh \
    --path_to_saves /data/home/vkropoti/unlearning/saves \
    --out_path /data/home/vkropoti/unlearning/zips_for_gpu/saves-clean \
    --save_eval 0

  bash package_saves.sh \
    --path_to_saves /data/home/vkropoti/unlearning/saves/unlearn \
    --out_path /data/home/vkropoti/unlearning/zips_for_gpu/unlearn-clean \
    --save_eval 1

Notes:
  --out_path is the clean directory path. The script also writes --out_path.zip.
  --save_eval 1 keeps only endpoint benchmark eval JSON files under run_dir/evals/.
  --save_eval 0 keeps only summaries plus the main Hydra config per run.
EOF
}

PATH_TO_SAVES=""
OUT_PATH=""
SAVE_EVAL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path_to_saves)
      PATH_TO_SAVES="${2:-}"
      shift 2
      ;;
    --out_path)
      OUT_PATH="${2:-}"
      shift 2
      ;;
    --save_eval)
      SAVE_EVAL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PATH_TO_SAVES}" || -z "${OUT_PATH}" ]]; then
  echo "Error: --path_to_saves and --out_path are required." >&2
  usage >&2
  exit 1
fi

if [[ "${SAVE_EVAL}" != "0" && "${SAVE_EVAL}" != "1" ]]; then
  echo "Error: --save_eval must be 0 or 1." >&2
  exit 1
fi

if [[ ! -d "${PATH_TO_SAVES}" ]]; then
  echo "Error: path_to_saves does not exist: ${PATH_TO_SAVES}" >&2
  exit 1
fi

PATH_TO_SAVES="${PATH_TO_SAVES%/}"
OUT_PATH="${OUT_PATH%/}"

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' command is required but not found." >&2
  exit 1
fi

src_dir="$(realpath "${PATH_TO_SAVES}")"
out_parent="$(dirname "${OUT_PATH}")"
mkdir -p "${out_parent}"
out_parent="$(cd "${out_parent}" && pwd -P)"
clean_dir="${out_parent}/$(basename "${OUT_PATH}")"
zip_path="${clean_dir}.zip"

echo "[package_saves] src_dir=${src_dir}"
echo "[package_saves] clean_dir=${clean_dir}"
echo "[package_saves] zip_path=${zip_path}"
echo "[package_saves] save_eval=${SAVE_EVAL}"

rm -rf "${clean_dir}" "${zip_path}"
mkdir -p "${clean_dir}"

copied_files=0
skipped_files=0

should_keep_endpoint_eval_file() {
  local rel_path="$1"
  local base_name=""
  local dir_name=""
  local parent_name=""

  base_name="$(basename "${rel_path}")"
  dir_name="$(dirname "${rel_path}")"
  parent_name="$(basename "${dir_name}")"

  if [[ "${parent_name}" != "evals" ]]; then
    return 1
  fi

  if [[ "${base_name}" != *_EVAL.json ]]; then
    return 1
  fi

  # Keep the main benchmark eval plus optional analysis sidecars, but not the
  # large derived detail payloads.
  if [[ "${base_name}" == "COS_SIM_EVAL.json" || "${base_name}" == "WRONG_GENERATIONS_EVAL.json" ]]; then
    return 1
  fi

  return 0
}

should_keep_main_hydra_config() {
  local rel_path="$1"

  if [[ "${rel_path}" != */.hydra/config.yaml ]]; then
    return 1
  fi

  case "${rel_path}" in
    */evals/.hydra/config.yaml|*/checkpoint_evals/*/.hydra/config.yaml|*/checkpoint_evals_utility/*/.hydra/config.yaml|*/checkpoint_evals_merged/*/.hydra/config.yaml)
      return 1
      ;;
  esac

  return 0
}

should_keep_file() {
  local rel_path="$1"
  local base_name=""

  base_name="$(basename "${rel_path}")"

  if [[ "${base_name}" == *_SUMMARY.json ]]; then
    return 0
  fi

  if [[ "${SAVE_EVAL}" == "1" ]] && should_keep_endpoint_eval_file "${rel_path}"; then
    return 0
  fi

  # Keep merged trajectory summaries from checkpoint + utility evals.
  if [[ "${base_name}" == "trajectory_metrics.json" ]]; then
    return 0
  fi

  # Keep only the run-level resolved Hydra config for reproducibility.
  if should_keep_main_hydra_config "${rel_path}"; then
    return 0
  fi

  return 1
}

while IFS= read -r -d '' src_file; do
  rel_path="${src_file#${src_dir}/}"

  if ! should_keep_file "${rel_path}"; then
    skipped_files=$((skipped_files + 1))
    continue
  fi

  dst_file="${clean_dir}/${rel_path}"
  mkdir -p "$(dirname "${dst_file}")"
  cp -p "${src_file}" "${dst_file}"
  copied_files=$((copied_files + 1))
done < <(find "${src_dir}" -type f -print0)

echo "[package_saves] copied files: ${copied_files}"
echo "[package_saves] skipped files: ${skipped_files}"

(
  cd "$(dirname "${clean_dir}")"
  zip -rq "${zip_path}" "$(basename "${clean_dir}")"
)

echo "[package_saves] wrote ${zip_path}"
echo "[package_saves] done"
