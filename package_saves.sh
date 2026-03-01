#!/usr/bin/env bash

set -euo pipefail

if [[ ! -d "saves" ]]; then
    echo "Error: run this script from repo root (missing ./saves directory)." >&2
    exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
    echo "Error: 'zip' command is required but not found." >&2
    exit 1
fi

src_dir="saves"
clean_dir="saves-clean"
zip_path="${clean_dir}.zip"

echo "[package_saves] Rebuilding ${clean_dir} and ${zip_path}"
rm -rf "${clean_dir}" "${zip_path}" "saves-summary"
mkdir -p "${clean_dir}"

copied_files=0
skipped_files=0

while IFS= read -r -d '' src_file; do
    rel_path="${src_file#${src_dir}/}"
    base_name="$(basename "${src_file}")"
    dir_name="$(dirname "${src_file}")"

    keep=0
    if [[ "${base_name}" == *_SUMMARY.json ]]; then
        keep=1
    elif [[ "${dir_name}" == */.hydra ]] && [[ "${base_name}" == *.yaml || "${base_name}" == *.yml ]]; then
        keep=1
    fi

    if [[ "${keep}" -eq 0 ]]; then
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

zip -rq "${zip_path}" "${clean_dir}"
echo "[package_saves] wrote ${zip_path}"
echo "[package_saves] done"
