#!/bin/bash

set -euo pipefail

# Base forget/retain pairs (edit once here for all duet scripts).
DUET_SPLITS=(
    "city_forget_rare_5 city_fast_retain_500"
    "city_forget_popular_5 city_fast_retain_500"
)

# Builds global array: forget_retain_splits
# If MERGE_POPULARITY_FORGET=1, merge rare+popular into a combined forget split.
set_forget_retain_splits() {
    local merge_flag="${MERGE_POPULARITY_FORGET:-0}"
    if [[ "${merge_flag}" != "1" ]]; then
        forget_retain_splits=("${DUET_SPLITS[@]}")
    else
        local pair forget retain base key retain_value found idx
        local -a merge_keys=()
        local -a merge_forgets=()
        local -a merge_labels=()
        for pair in "${DUET_SPLITS[@]}"; do
            forget=$(echo "$pair" | cut -d' ' -f1)
            retain=$(echo "$pair" | cut -d' ' -f2)
            base=${forget/_rare_/_}
            base=${base/_popular_/_}
            key="${base}|${retain}"
            found=0
            for idx in "${!merge_keys[@]}"; do
                if [[ "${merge_keys[$idx]}" == "${key}" ]]; then
                    merge_forgets[$idx]="${merge_forgets[$idx]}+${forget}"
                    found=1
                    break
                fi
            done
            if [[ "${found}" != "1" ]]; then
                merge_keys+=("${key}")
                merge_forgets+=("${forget}")
                merge_labels+=("${base}")
            fi
        done

        forget_retain_splits=()
        for idx in "${!merge_keys[@]}"; do
            retain_value=${merge_keys[$idx]#*|}
            forget_retain_splits+=(
                "${merge_forgets[$idx]} ${retain_value} ${merge_labels[$idx]}"
            )
        done
    fi

    if [[ -n "${FORGET_SPLIT_OVERRIDE:-}" && -n "${RETAIN_SPLIT_OVERRIDE:-}" ]]; then
        local override_label="${FORGET_LABEL_OVERRIDE:-${FORGET_SPLIT_OVERRIDE}}"
        forget_retain_splits=(
            "${FORGET_SPLIT_OVERRIDE} ${RETAIN_SPLIT_OVERRIDE} ${override_label}"
        )
    fi
}
