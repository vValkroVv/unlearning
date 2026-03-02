# Production Runs (Default Script Hyperparams)

## Current HF Cache Structure

Current workspace cache layout (`/workspace/unlearning`) for datasets and models:

```text
.hf_datasets_cache/
  SwetieePawsss___duet/
    default/0.0.0/
  SwetieePawsss___exp_un_lamb/
    default/0.0.0/

.hf_home/
  datasets/
    SwetieePawsss___duet/default/0.0.0/
    SwetieePawsss___exp_un_lamb/default/0.0.0/
  hub/
    datasets--SwetieePawsss--DUET/
    datasets--SwetieePawsss--exp_UNLamb/
    models--SwetieePawsss--DUET_ft_models/
    models--SwetieePawsss--UNLamb_ft_models/
    models--unsloth--Llama-3.1-8B-Instruct/
  models--SwetieePawsss--DUET_ft_models/
    snapshots/<rev>/llama-3.1-8b-instruct-tripunlamb-ft/
  models--SwetieePawsss--UNLamb_ft_models/
    snapshots/<rev>/llama-3.1-8b-instruct-popqa-ft/
  models--unsloth--Llama-3.1-8B-Instruct/
    snapshots/<rev>/
```

Notes:
- `<rev>` is HF snapshot hash and changes by revision.
- Keep `LOCAL_SFT_BASE` as repo id/path root and set `SFT_SUBFOLDER` to the model folder inside snapshot/repo.

## Common setup

```bash
cd /workspace/unlearning
source .venv/bin/activate
export HF_HOME=/workspace/unlearning/.hf_home
export HF_DATASETS_CACHE=/workspace/unlearning/.hf_datasets_cache
export TRITON_CACHE_DIR=/workspace/unlearning/.triton
```

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
bash scripts/popqa/npo_sam_popqa.sh
```

## 9) LoKU - DUET

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
EVAL_BATCH_SIZE=8 \
LRS="1e-4" \
bash scripts/duet/loku_duet.sh
```

## 10) LoKU - UNLamb

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
EVAL_BATCH_SIZE=8 \
LRS="1e-4" \
bash scripts/popqa/loku_popqa.sh
```

Notes:
- LoKU runs a separate importance pass before training; keep `IMPORTANCE_BATCH_SIZE` small (usually `1`) to avoid OOM.
- If you only need a quick validation run, set `IMPORTANCE_MAX_STEPS` to a small value (for example `50`).

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
IMPORTANCE_PATH=/workspace/unlearning/saves/importances/tmp/duet_loku_imp.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
EVAL_BATCH_SIZE=8 \
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
IMPORTANCE_ROOT=/workspace/unlearning/saves/importances/custom \
IMPORTANCE_PATH=/workspace/unlearning/saves/importances/custom/{base_model}_{forget_label}_{retain_split}_{targets_tag}.pt \
DELETE_IMPORTANCE_AFTER_RUN=1 \
EVAL_BATCH_SIZE=8 \
LRS="1e-4" \
bash scripts/popqa/loku_popqa.sh
```
