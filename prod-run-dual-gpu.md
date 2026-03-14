# Production DualCF GPU Runs (v2)

This runbook is for the DualCF v2 iteration:

- clean counterfactuals first
- percentile-calibrate routing offline
- use hybrid retain attribution
- run DUET `rare -> popular -> merged`
- train for `5` epochs
- save/evaluate half-epoch checkpoints

Do not use Gemma as the main evidence path until counterfactual leakage is fixed.

## Common setup

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

export HF_HOME=/data/home/vkropoti/unlearning/.hf_home
export HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache
export TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PER_DEVICE_TRAIN_BS=16
export GRAD_ACCUM=2
export NUM_EPOCHS=5
export EVAL_BATCH_SIZE=64
export DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1
export CHECKPOINT_EVERY_HALF_EPOCH=1
export SAVE_TOTAL_LIMIT=12
export OUTPUT_ROOT=/data/home/vkropoti/unlearning/saves/unlearn

export LRS="1e-6 5e-6 1e-5 5e-5 1e-4"
export TAU_DS="0.4 0.5 0.6"
export TAU_AS="0.4 0.5 0.6"
export TEMP_DS="0.1 0.2 0.4"
export TEMP_AS="0.1 0.2 0.4"
export RISK_FORGET_SCALES="0.25 0.5 0.75"
export LAMBDA_RET_HIS="2.0 3.0 4.0"
export ALPHA_EFF_STATS="topk_mean"
export ALPHA_EFF_TOPK_FRACS="0.25"
```

`OUTPUT_ROOT` is respected by:

- `scripts/duet/dual_cf_duet.sh`
- `scripts/rwku/dual_cf_rwku.sh`
- `scripts/duet/ga_duet.sh`
- `scripts/duet/npo_duet.sh`
- `scripts/duet/npo_sam_duet.sh`
- `scripts/duet/loku_duet.sh`
- `scripts/rwku/ga_rwku.sh`
- `scripts/rwku/npo_rwku.sh`
- `scripts/rwku/npo_sam_rwku.sh`
- `scripts/rwku/loku_rwku.sh`

So the run directories land under:

- `/data/home/vkropoti/unlearning/saves/unlearn/<task_name>`

For `LoKU`, importance files also move out of the repo and default to:

- `/data/home/vkropoti/unlearning/saves/importances/duet/loku`
- `/data/home/vkropoti/unlearning/saves/importances/rwku/loku`

With the matched launcher updates, all DUET/RWKU ablations now also:

- save `checkpoint-*` directories every ~0.5 epoch when
  `CHECKPOINT_EVERY_HALF_EPOCH=1`
- write `dualcf_trace.jsonl` into each run directory
- remove only top-level `*.safetensors` from `run_dir/` after endpoint eval when
  `DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1`

Checkpoint safetensors are intentionally kept until you finish trajectory eval.

LoKU-specific note:

- `DELETE_FILA_BASE_AFTER_EVAL` now defaults to `0`
- if you force `DELETE_FILA_BASE_AFTER_EVAL=1` together with
  `CHECKPOINT_EVERY_HALF_EPOCH=1`, the launcher overrides it back to `0`
- this is required because LoKU checkpoint eval needs `run_dir/base_model`

## vLLM generator

Run the Qwen3 generator in a separate env or container, not in the training env.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
export TP=4
export MAX_LEN=8192
export PORT=8000

scripts/vllm/start_qwen3_cf_server.sh
```

In the training shell:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

## Artifact prep

### DUET rare

```bash
export CUDA_VISIBLE_DEVICES=4
export MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml
export LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml
export SFT_MODEL_PATH=SwetieePawsss/DUET_ft_models
export SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft
export FORGET_LABEL=rare
export OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/rare_llama_v2

scripts/duet/prepare_dual_cf_duet_v2.sh
```

### DUET popular

```bash
export CUDA_VISIBLE_DEVICES=4
export FORGET_LABEL=popular
export OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/popular_llama_v2

scripts/duet/prepare_dual_cf_duet_v2.sh
```

### DUET merged

Run this only after rare and popular are clean.

```bash
export CUDA_VISIBLE_DEVICES=4
export FORGET_LABEL=merged
export OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/merged_llama_v2

scripts/duet/prepare_dual_cf_duet_v2.sh
```

### RWKU

```bash
export CUDA_VISIBLE_DEVICES=4
export MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml
export LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml
export BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct
export FORGET_SPLIT=forget_level2
export RETAIN_SPLIT=neighbor_level2
export OUT_DIR=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku/llama_level2_v2

scripts/rwku/prepare_dual_cf_rwku_v2.sh
```

## DUET training

### Full DualCF

Rare:

```bash
export CUDA_VISIBLE_DEVICES=4
export CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/rare_llama_v2/dualcf_rare_v2.jsonl
export METHOD_VARIANT=full
export FORGET_LABEL=rare

scripts/duet/run_dualcf_ablation_v2.sh
```

Popular:

```bash
export CUDA_VISIBLE_DEVICES=5
export CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/popular_llama_v2/dualcf_popular_v2.jsonl
export METHOD_VARIANT=full
export FORGET_LABEL=popular

scripts/duet/run_dualcf_ablation_v2.sh
```

Merged:

```bash
export CUDA_VISIBLE_DEVICES=6
export CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/merged_llama_v2/dualcf_merged_v2.jsonl
export METHOD_VARIANT=full
export FORGET_LABEL=merged

scripts/duet/run_dualcf_ablation_v2.sh
```

### Required ablations

Difficulty-only:

```bash
export METHOD_VARIANT=d_only
scripts/duet/run_dualcf_ablation_v2.sh
```

Attribution-only:

```bash
export METHOD_VARIANT=a_only
scripts/duet/run_dualcf_ablation_v2.sh
```

Uniform counterfactual DPO:

```bash
export METHOD_VARIANT=dpo
scripts/duet/run_dualcf_ablation_v2.sh
```

Baselines:

```bash
export METHOD_VARIANT=ga
scripts/duet/run_dualcf_ablation_v2.sh

export METHOD_VARIANT=npo
scripts/duet/run_dualcf_ablation_v2.sh

export METHOD_VARIANT=npo_sam
scripts/duet/run_dualcf_ablation_v2.sh

export METHOD_VARIANT=loku
scripts/duet/run_dualcf_ablation_v2.sh
```

Every method variant above now uses the same trajectory-saving behavior:

- half-epoch `checkpoint-*` saves
- endpoint eval into `run_dir/evals`
- training trace in `run_dir/dualcf_trace.jsonl`
- top-level adapter safetensor cleanup after endpoint eval

## RWKU training

```bash
export CUDA_VISIBLE_DEVICES=7
export CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku/llama_level2_v2/dualcf_forget_level2_v2.jsonl
export METHOD_VARIANT=full

scripts/rwku/run_dualcf_ablation_v2.sh
```

Run the same ablation variants on RWKU after DUET rare/popular are stable.

## Checkpoint evaluation

DUET:

```bash
scripts/duet/eval_checkpoints_duet.sh \
  /path/to/run_dir \
  city_forget_rare_5 \
  city_fast_retain_500 \
  /data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
  /data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct
```

For LoKU, the checkpoint evaluator auto-detects `run_dir/base_model` and uses it
instead of the original base path. To clean that FILA base after trajectory eval:

```bash
DELETE_RUN_BASE_MODEL_AFTER_EVAL=1 scripts/duet/eval_checkpoints_duet.sh \
  /path/to/loku_run_dir \
  city_forget_rare_5 \
  city_fast_retain_500 \
  /data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
  /data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct
```

RWKU:

```bash
scripts/rwku/eval_checkpoints_rwku.sh \
  /path/to/run_dir \
  forget_level2 \
  neighbor_level2 \
  /data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
  /data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct
```

For LoKU on RWKU, the same cleanup pattern works:

```bash
DELETE_RUN_BASE_MODEL_AFTER_EVAL=1 scripts/rwku/eval_checkpoints_rwku.sh \
  /path/to/loku_run_dir \
  forget_level2 \
  neighbor_level2 \
  /data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
  /data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct
```

Each script writes:

- per-checkpoint eval folders under `checkpoint_evals/`
- `checkpoint_evals/summary.tsv`

## Campaign order

1. DUET rare, full + ablations.
2. DUET popular, full + ablations.
3. DUET merged, full only after split runs are clean.
4. RWKU with the hybrid proxy retain map.
5. Expand to Qwen only after the Llama path is stable.
