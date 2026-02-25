#!/bin/bash

set -euo pipefail

script_dir=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LRS=${LRS:-"1e-5 5e-5 1e-4 5e-4 1e-3 5e-3"}

echo "[rwku] Running AdaWGD"
bash "${script_dir}/ada_wgd_rwku.sh"

echo "[rwku] Running AdaPop"
bash "${script_dir}/ada_pop_rwku.sh"
