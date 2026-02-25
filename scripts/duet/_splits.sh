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
        return
    fi

    declare -A merge_forget
    declare -A merge_label
    local pair forget retain base key
    for pair in "${DUET_SPLITS[@]}"; do
        forget=$(echo "$pair" | cut -d' ' -f1)
        retain=$(echo "$pair" | cut -d' ' -f2)
        base=${forget/_rare_/_}
        base=${base/_popular_/_}
        key="${base}|${retain}"
        if [[ -z "${merge_forget[$key]+x}" ]]; then
            merge_forget[$key]="${forget}"
        else
            merge_forget[$key]="${merge_forget[$key]}+${forget}"
        fi
        merge_label[$key]="${base}"
    done

    forget_retain_splits=()
    for key in "${!merge_forget[@]}"; do
        retain=${key#*|}
        forget_retain_splits+=("${merge_forget[$key]} ${retain} ${merge_label[$key]}")
    done
}
