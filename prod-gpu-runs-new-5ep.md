# Production GPU Runs (Offline Server, DUET + RWKU)

This runbook keeps only the current production targets:

- datasets: `DUET`, `RWKU`
- methods: `GA`, `NPO`, `NPO-SAM`, `LoKU`
- model groups in this file: `Llama`, `Qwen`, `Gemma`

Common rules used everywhere below:

- `PER_DEVICE_TRAIN_BS=16`
- `GRAD_ACCUM=2`
- `NUM_EPOCHS=5`
- `LRS="1e-6 5e-6 1e-5 5e-5 1e-4"`
- `EVAL_BATCH_SIZE=64`
- `DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1`
- current production LoRA adapters use only `q_proj`, `k_proj`, `v_proj`, `o_proj`
- DUET always uses `MERGE_POPULARITY_FORGET=1`
- RWKU does not use `USE_SFT_BASE`; it loads the base model directly via `HF_BASE_MODEL_PATH`

## Current Local Structure

Expected layout inside `/data/home/vkropoti/unlearning`:

```text
SwetieePawsss/
  DUET/
    data/*.parquet
  exp_r/
    data/*.parquet
  DUET_ft_models/
    llama-3.1-8b-instruct-tripunlamb-ft/
    qwen2.5-7b-instruct-tripunlamb-ft/
    gemma-7b-it-tripunlamb-ft/
models/
  BASE/
    Llama-3.1-8B-Instruct/
    Qwen2.5-7B-Instruct/
    gemma-7b-it/
```

Notes:

- DUET scripts read the dataset from `SwetieePawsss/DUET`.
- RWKU scripts read the dataset from `SwetieePawsss/exp_r`.
- DUET finetuned checkpoints should live under `SwetieePawsss/DUET_ft_models`, one subfolder per model.
- RWKU base models should live under `/data/home/vkropoti/unlearning/models/BASE`.
- Use direct local paths in `HF_BASE_MODEL_PATH` for RWKU offline runs.

## One-time path wiring

If your code is in `/home/vkropoti/diploma/open-unlearning` but datasets/models are in
`/data/home/vkropoti/unlearning/SwetieePawsss`, create this symlink once:

```bash
ln -sfn /data/home/vkropoti/unlearning/SwetieePawsss /home/vkropoti/diploma/open-unlearning/SwetieePawsss
```

## Common setup

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

export HF_HOME=/data/home/vkropoti/unlearning/.hf_home
export HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache
export TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## Llama

### 1) GA - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/ga_duet.sh
```

### 2) NPO - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/npo_duet.sh
```

### 3) NPO-SAM - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/npo_sam_duet.sh
```

### 4) LoKU - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
IMPORTANCE_BATCH_SIZE=16 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/duet_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/loku_duet.sh
```

### 5) GA - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/ga_rwku.sh
```

### 6) NPO - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/npo_rwku.sh
```

### 7) NPO-SAM - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/npo_sam_rwku.sh
```

### 8) LoKU - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
IMPORTANCE_BATCH_SIZE=16 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/rwku_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/loku_rwku.sh
```

## Qwen

### 1) GA - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
BASE_MODEL=Qwen2.5-7B-Instruct \
MODEL_CONFIG=Qwen2.5-7B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=qwen2.5-7b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/ga_duet.sh
```

### 2) NPO - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
BASE_MODEL=Qwen2.5-7B-Instruct \
MODEL_CONFIG=Qwen2.5-7B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=qwen2.5-7b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/npo_duet.sh
```

### 3) NPO-SAM - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
BASE_MODEL=Qwen2.5-7B-Instruct \
MODEL_CONFIG=Qwen2.5-7B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=qwen2.5-7b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/npo_sam_duet.sh
```

### 4) LoKU - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
BASE_MODEL=Qwen2.5-7B-Instruct \
MODEL_CONFIG=Qwen2.5-7B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=qwen2.5-7b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
IMPORTANCE_BATCH_SIZE=16 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/qwen_duet_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/loku_duet.sh
```

### 5) GA - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
BASE_MODEL=Qwen2.5-7B-Instruct \
MODEL_CONFIG=Qwen2.5-7B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/ga_rwku.sh
```

### 6) NPO - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
BASE_MODEL=Qwen2.5-7B-Instruct \
MODEL_CONFIG=Qwen2.5-7B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/npo_rwku.sh
```

### 7) NPO-SAM - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
BASE_MODEL=Qwen2.5-7B-Instruct \
MODEL_CONFIG=Qwen2.5-7B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/npo_sam_rwku.sh
```

### 8) LoKU - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
BASE_MODEL=Qwen2.5-7B-Instruct \
MODEL_CONFIG=Qwen2.5-7B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
IMPORTANCE_BATCH_SIZE=16 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/qwen_rwku_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/loku_rwku.sh
```

## Gemma

### 1) GA - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
BASE_MODEL=gemma-7b-it \
MODEL_CONFIG=gemma-7b-it-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=gemma-7b-it-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/ga_duet.sh
```

### 2) NPO - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
BASE_MODEL=gemma-7b-it \
MODEL_CONFIG=gemma-7b-it-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=gemma-7b-it-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/npo_duet.sh
```

### 3) NPO-SAM - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
BASE_MODEL=gemma-7b-it \
MODEL_CONFIG=gemma-7b-it-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=gemma-7b-it-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/npo_sam_duet.sh
```

### 4) LoKU - DUET (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
BASE_MODEL=gemma-7b-it \
MODEL_CONFIG=gemma-7b-it-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=gemma-7b-it-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
IMPORTANCE_BATCH_SIZE=16 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/gemma_duet_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/loku_duet.sh
```

### 5) GA - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
BASE_MODEL=gemma-7b-it \
MODEL_CONFIG=gemma-7b-it-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/ga_rwku.sh
```

### 6) NPO - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
BASE_MODEL=gemma-7b-it \
MODEL_CONFIG=gemma-7b-it-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/npo_rwku.sh
```

### 7) NPO-SAM - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
BASE_MODEL=gemma-7b-it \
MODEL_CONFIG=gemma-7b-it-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/npo_sam_rwku.sh
```

### 8) LoKU - RWKU (done)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
BASE_MODEL=gemma-7b-it \
MODEL_CONFIG=gemma-7b-it-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=5 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
IMPORTANCE_BATCH_SIZE=16 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/gemma_rwku_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/loku_rwku.sh
```
