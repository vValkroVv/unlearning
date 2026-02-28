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
summary_dir="saves-summary"
zip_path="${clean_dir}.zip"

# Tokenizer artifacts to exclude from saves-clean.
tokenizer_patterns=(
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "added_tokens.json"
    "vocab.json"
    "merges.txt"
    "spiece.model"
    "sentencepiece.bpe.model"
    "tokenizer.model"
)

echo "[package_saves] Rebuilding ${clean_dir}, ${summary_dir}, ${zip_path}"
rm -rf "${clean_dir}" "${summary_dir}" "${zip_path}"

mkdir -p "${clean_dir}"

copied_clean=0
excluded_tokenizer=0
copied_summary=0

while IFS= read -r -d '' src_file; do
    rel_path="${src_file#${src_dir}/}"
    base_name="$(basename "${src_file}")"

    skip=0
    for pattern in "${tokenizer_patterns[@]}"; do
        if [[ "${base_name}" == "${pattern}" ]]; then
            skip=1
            break
        fi
    done

    if [[ "${skip}" -eq 1 ]]; then
        excluded_tokenizer=$((excluded_tokenizer + 1))
        continue
    fi

    dst_file="${clean_dir}/${rel_path}"
    mkdir -p "$(dirname "${dst_file}")"
    cp -p "${src_file}" "${dst_file}"
    copied_clean=$((copied_clean + 1))
done < <(find "${src_dir}" -type f -print0)

echo "[package_saves] saves-clean files copied: ${copied_clean}"
echo "[package_saves] tokenizer files excluded: ${excluded_tokenizer}"

zip -rq "${zip_path}" "${clean_dir}"
echo "[package_saves] wrote ${zip_path}"

mkdir -p "${summary_dir}"
while IFS= read -r -d '' src_file; do
    rel_path="${src_file#${src_dir}/}"
    dst_file="${summary_dir}/${rel_path}"
    mkdir -p "$(dirname "${dst_file}")"
    cp -p "${src_file}" "${dst_file}"
    copied_summary=$((copied_summary + 1))
done < <(find "${src_dir}" -type f -name '*_SUMMARY.json' -print0)

echo "[package_saves] saves-summary files copied: ${copied_summary}"
echo "[package_saves] done"
