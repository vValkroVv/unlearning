#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash compare_saves_clean_sizes.sh [--path_to_saves_clean PATH]

Examples:
  bash compare_saves_clean_sizes.sh

  bash compare_saves_clean_sizes.sh \
    --path_to_saves_clean /data/home/vkropoti/unlearning/zips_for_gpu/saves-clean

Notes:
  - Compares packaged `saves-clean` footprint across `.hydra/*`, `*.json`, and `*.tsv`.
  - Reports any remaining files outside those buckets under `other`.
EOF
}

PATH_TO_SAVES_CLEAN="./saves-clean"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --path_to_saves_clean)
      PATH_TO_SAVES_CLEAN="${2:-}"
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

if [[ -z "${PATH_TO_SAVES_CLEAN}" ]]; then
  echo "Error: --path_to_saves_clean must not be empty." >&2
  exit 1
fi

if [[ ! -d "${PATH_TO_SAVES_CLEAN}" ]]; then
  echo "Error: saves-clean path does not exist: ${PATH_TO_SAVES_CLEAN}" >&2
  exit 1
fi

root_dir="$(cd "${PATH_TO_SAVES_CLEAN}" && pwd -P)"

json_bytes=0
json_files=0
hydra_bytes=0
hydra_files=0
tsv_bytes=0
tsv_files=0
other_bytes=0
other_files=0

file_size_bytes() {
  local file_path="$1"
  wc -c < "${file_path}" | tr -d '[:space:]'
}

format_bytes() {
  local total_bytes="$1"
  awk -v total_bytes="${total_bytes}" '
    BEGIN {
      split("B KiB MiB GiB TiB PiB", units, " ")
      size = total_bytes + 0
      unit = 1
      while (size >= 1024 && unit < 6) {
        size /= 1024
        unit += 1
      }
      if (unit == 1) {
        printf "%d %s", size, units[unit]
      } else {
        printf "%.2f %s", size, units[unit]
      }
    }
  '
}

while IFS= read -r -d '' file_path; do
  rel_path="${file_path#${root_dir}/}"
  size_bytes="$(file_size_bytes "${file_path}")"

  # `.hydra` is a directory bucket; JSON and TSV stay extension-based.
  if [[ "${rel_path}" == */.hydra/* ]]; then
    hydra_bytes=$((hydra_bytes + size_bytes))
    hydra_files=$((hydra_files + 1))
    continue
  fi

  if [[ "${rel_path}" == *.json ]]; then
    json_bytes=$((json_bytes + size_bytes))
    json_files=$((json_files + 1))
    continue
  fi

  if [[ "${rel_path}" == *.tsv ]]; then
    tsv_bytes=$((tsv_bytes + size_bytes))
    tsv_files=$((tsv_files + 1))
    continue
  fi

  other_bytes=$((other_bytes + size_bytes))
  other_files=$((other_files + 1))
done < <(find "${root_dir}" -type f -print0)

print_bucket() {
  local label="$1"
  local files="$2"
  local bytes="$3"

  printf '%-8s files=%-8s bytes=%-12s human=%s\n' \
    "${label}" \
    "${files}" \
    "${bytes}" \
    "$(format_bytes "${bytes}")"
}

echo "[compare_saves_clean_sizes] root_dir=${root_dir}"
print_bucket ".hydra" "${hydra_files}" "${hydra_bytes}"
print_bucket "json" "${json_files}" "${json_bytes}"
print_bucket "tsv" "${tsv_files}" "${tsv_bytes}"
print_bucket "other" "${other_files}" "${other_bytes}"

largest_bucket=".hydra"
largest_bytes="${hydra_bytes}"
runner_up_bytes=-1
has_tie=0

for bucket_spec in "json:${json_bytes}" "tsv:${tsv_bytes}"; do
  bucket_name="${bucket_spec%%:*}"
  bucket_bytes="${bucket_spec#*:}"

  if (( bucket_bytes > largest_bytes )); then
    runner_up_bytes="${largest_bytes}"
    largest_bucket="${bucket_name}"
    largest_bytes="${bucket_bytes}"
    has_tie=0
    continue
  fi

  if (( bucket_bytes == largest_bytes )); then
    has_tie=1
    continue
  fi

  if (( bucket_bytes > runner_up_bytes )); then
    runner_up_bytes="${bucket_bytes}"
  fi
done

if (( hydra_bytes > runner_up_bytes && hydra_bytes < largest_bytes )); then
  runner_up_bytes="${hydra_bytes}"
fi

if (( has_tie == 1 )); then
  echo "[compare_saves_clean_sizes] larger_bucket=tie diff_bytes=0 diff_human=0 B"
else
  if (( runner_up_bytes < 0 )); then
    runner_up_bytes=0
  fi

  diff_bytes=$((largest_bytes - runner_up_bytes))
  echo "[compare_saves_clean_sizes] larger_bucket=${largest_bucket} diff_bytes=${diff_bytes} diff_human=$(format_bytes "${diff_bytes}")"
fi
