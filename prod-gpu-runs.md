# Production GPU Runs (Offline Server)

This is the same runbook as `prod-runs.md`, adapted for your offline server root:

- `/data/home/vkropoti/unlearning`
- local repos under `SwetieePawsss/...`

## Current Local Structure

Expected layout inside `/data/home/vkropoti/unlearning`:

```text
SwetieePawsss/
  DUET/
    data/*.parquet
  exp_UNLamb/
    data/*.parquet
  DUET_ft_models/
    llama-3.1-8b-instruct-tripunlamb-ft/
  UNLamb_ft_models/
    llama-3.1-8b-instruct-popqa-ft/
```

Notes:
- Keep `LOCAL_SFT_BASE` as repo-id-like local path root, and set `SFT_SUBFOLDER` to the model folder.
- Folder name must be exactly `SwetieePawsss` (not misspelled).

## One-time path wiring

If your code is in `/home/vkropoti/diploma/open-unlearning` but models/datasets are in
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

# force offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Important:
- `HF_DATASETS_OFFLINE` must not contain spaces.
- Wrong: `export HF_DATASETS OFFLINE=1`
- Correct: `export HF_DATASETS_OFFLINE=1`

## Eval batch size in same run

For all scripts below (`1` to `8`), evaluation runs inside the same command and supports:

- `EVAL_BATCH_SIZE` (default: `8`)

Example style (same as train params):

```bash
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
EVAL_BATCH_SIZE=32 \
bash scripts/duet/npo_sam_duet.sh
```

## 1) GA - DUET

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/ga_duet.sh
```

## 2) GA - UNLamb

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/popqa/ga_popqa.sh
```

## 3) NPO - DUET

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/npo_duet.sh
```

## 4) NPO - UNLamb

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/popqa/npo_popqa.sh
```

## 5) FALCON - DUET

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
MI_SELECT_LAYERS=1 \
MI_MODEL_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MI_TOKENIZER_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
bash scripts/duet/falcon_duet.sh
```

## 6) FALCON - UNLamb

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
MI_SELECT_LAYERS=1 \
MI_MODEL_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MI_TOKENIZER_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
bash scripts/popqa/falcon_popqa.sh
```

## 7) NPO-SAM - DUET

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/npo_sam_duet.sh
```

## 8) NPO-SAM - UNLamb

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/popqa/npo_sam_popqa.sh
```

## 9) LoKU - DUET

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=32 \
GRAD_ACCUM=1 \
IMPORTANCE_BATCH_SIZE=32 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/duet_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/loku_duet.sh ; \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=0 \
PER_DEVICE_TRAIN_BS=32 \
GRAD_ACCUM=1 \
IMPORTANCE_BATCH_SIZE=32 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/duet_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/loku_duet.sh
```

## 10) LoKU - UNLamb

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=32 \
GRAD_ACCUM=1 \
IMPORTANCE_BATCH_SIZE=32 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/popqa_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/popqa/loku_popqa.sh ; \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=0 \
PER_DEVICE_TRAIN_BS=32 \
GRAD_ACCUM=1 \
IMPORTANCE_BATCH_SIZE=32 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/popqa_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/popqa/loku_popqa.sh
```

Notes:
- LoKU includes an extra importance-measurement stage before training, so keep `IMPORTANCE_BATCH_SIZE` conservative.
- For smoke checks use `IMPORTANCE_MAX_STEPS` (for example `50`) before full runs.

## LoKU Importance Path and Auto-Delete

Use these params with either `scripts/duet/loku_duet.sh` or `scripts/popqa/loku_popqa.sh`:

- `IMPORTANCE_PATH`: exact path (or template) for saved importance file.
- `IMPORTANCE_ROOT`: custom directory root for auto naming.
- `DELETE_IMPORTANCE_AFTER_RUN=1`: delete measured importance file when the script exits.

### Example A: Exact file path

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/DUET_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
IMPORTANCE_BATCH_SIZE=1 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_tmp/duet_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_tmp/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
LRS="1e-4" \
bash scripts/duet/loku_duet.sh
```

### Example B: Directory root + template placeholders

Supported placeholders in `IMPORTANCE_PATH`:
- `{base_model}`
- `{forget_label}`
- `{retain_split}`
- `{targets_tag}`

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
USE_SFT_BASE=1 \
LOCAL_SFT_BASE=SwetieePawsss/UNLamb_ft_models \
SFT_SUBFOLDER=llama-3.1-8b-instruct-popqa-ft \
MERGE_POPULARITY_FORGET=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=32 \
IMPORTANCE_BATCH_SIZE=1 \
IMPORTANCE_MAX_STEPS=0 \
IMPORTANCE_ROOT=/data/home/vkropoti/unlearning/importance_custom \
IMPORTANCE_PATH=/data/home/vkropoti/unlearning/importance_custom/{base_model}_{forget_label}_{retain_split}_{targets_tag}.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
FILA_BASE_PATH=/data/home/vkropoti/unlearning/fila_base_custom/{task_name} \
DELETE_FILA_BASE_AFTER_EVAL=1 \
EVAL_BATCH_SIZE=8 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
LRS="1e-4" \
bash scripts/popqa/loku_popqa.sh
```

## LoKU FILA Base Path and Auto-Delete

Use these params to keep LoKU residual base checkpoints off `/home`:

- `FILA_BASE_PATH`: exact path (or template) where LoKU saves FILA residual base model.
- `FILA_BASE_ROOT`: root directory alternative; script auto-uses `${FILA_BASE_ROOT}/{task_name}`.
- `DELETE_FILA_BASE_AFTER_EVAL=1`: remove FILA residual base directory right after eval.

Supported placeholders in `FILA_BASE_PATH`:
- `{base_model}`
- `{forget_label}`
- `{retain_split}`
- `{task_name}`
