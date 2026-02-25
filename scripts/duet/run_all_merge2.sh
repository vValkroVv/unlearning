#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LRS=${LRS:-"1e-5 5e-5 1e-4 5e-4 1e-3"}
export MERGE_POPULARITY_FORGET=1

echo "[duet] Running WGA (merged forget)"
bash "${script_dir}/wga_duet.sh"

echo "[duet] Running AdaWGD (merged forget)"
bash "${script_dir}/ada_wgd_duet.sh"
