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
  --save_eval 1 keeps *_EVAL.json files. --save_eval 0 keeps only summaries.
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

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' command is required but not found." >&2
  exit 1
fi

src_dir="$(realpath "${PATH_TO_SAVES}")"
out_parent="$(dirname "${OUT_PATH}")"
mkdir -p "${out_parent}"
clean_dir="$(realpath -m "${OUT_PATH}")"
zip_path="${clean_dir}.zip"

echo "[package_saves] src_dir=${src_dir}"
echo "[package_saves] clean_dir=${clean_dir}"
echo "[package_saves] zip_path=${zip_path}"
echo "[package_saves] save_eval=${SAVE_EVAL}"

rm -rf "${clean_dir}" "${zip_path}"
mkdir -p "${clean_dir}"

copied_files=0
skipped_files=0

should_keep_file() {
  local rel_path="$1"
  local base_name=""
  local dir_name=""

  base_name="$(basename "${rel_path}")"
  dir_name="$(dirname "${rel_path}")"

  if [[ "${base_name}" == *_SUMMARY.json ]]; then
    return 0
  fi

  if [[ "${SAVE_EVAL}" == "1" && "${base_name}" == *_EVAL.json ]]; then
    return 0
  fi

  # Keep summary tables and merged trajectory summaries from checkpoint + utility evals.
  if [[ "${base_name}" == "summary.tsv" || "${base_name}" == "trajectory_metrics.json" ]]; then
    return 0
  fi

  # Keep Hydra configs for reproducibility.
  if [[ "${dir_name}" == */.hydra ]] && [[ "${base_name}" == *.yaml || "${base_name}" == *.yml ]]; then
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
