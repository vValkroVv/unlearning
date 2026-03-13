# Production DualCF GPU Runs (Offline Server, DUET merged + RWKU)

This runbook is for the `DualCF` method only.

It includes:

- merged `DUET` artifact generation for `Llama`, `Qwen`, `Gemma`
- `RWKU` artifact generation for `Llama`, `Qwen`, `Gemma`
- final training launches for all six runs

Common production rules used below:

- `PER_DEVICE_TRAIN_BS=16`
- `GRAD_ACCUM=2`
- `NUM_EPOCHS=2`
- `LRS="1e-6 5e-6 1e-5 5e-5 1e-4"`
- `EVAL_BATCH_SIZE=64`
- `DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1`
- current production LoRA adapters use only `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `DUET` in this file is merged only via `MERGE_POPULARITY_FORGET=1`
- full-artifact attribution scoring is explicit via `--retain-max-steps 0` and `--forget-max-steps 0`
- for H100 80GB artifact prep in this file:
  `score_difficulty.py --batch-size 32`
  and `score_attribution.py --retain-batch-size 4`
- attribution scoring is forced to production LoRA shape to match training:
  `--lora-r 32 --lora-alpha 64 --lora-dropout 0.0`

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
artifacts/
  dualcf/
    duet/
    rwku/
```

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
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## DUET merged artifacts

Use merged forget only for `DUET` in this file:

- `city_forget_rare_5+city_forget_popular_5` for counterfactual generation
- `city_fast_retain_500` for attribution retain bank

Always run validation before training.

### Llama - DUET merged artifact

```bash
export CUDA_VISIBLE_DEVICES=5

ART_ROOT=/data/home/vkropoti/unlearning/artifacts/dualcf/duet
mkdir -p "$ART_ROOT"

MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml
LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml
BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct
SFT_MODEL_PATH=SwetieePawsss/DUET_ft_models
SFT_SUBFOLDER=llama-3.1-8b-instruct-tripunlamb-ft
ART_PREFIX="$ART_ROOT/duet_llama_merged"

## done
python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/DUET \
  --split 'city_forget_rare_5+city_forget_popular_5' \
  --output-path "${ART_PREFIX}_step1.jsonl" \
  --question-key question \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --max-new-tokens 32

## done
python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files "${ART_PREFIX}_step1.jsonl" \
  --output-path "${ART_PREFIX}_step2.jsonl" \
  --question-key question \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$SFT_MODEL_PATH" \
  --tokenizer-path "$SFT_MODEL_PATH" \
  --model-subfolder "$SFT_SUBFOLDER" \
  --tokenizer-subfolder "$SFT_SUBFOLDER" \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0 \
  --w-mrd 0.0 \
  --w-stage 0.0 \
  --batch-size 32

## done
python src/tools/score_attribution.py \
  --model-cfg "$LORA_MODEL_CFG" \
  --model-path "$SFT_MODEL_PATH" \
  --tokenizer-path "$SFT_MODEL_PATH" \
  --model-subfolder "$SFT_SUBFOLDER" \
  --tokenizer-subfolder "$SFT_SUBFOLDER" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${ART_PREFIX}_step2.jsonl" \
  --retain-dataset-path SwetieePawsss/DUET \
  --retain-split city_fast_retain_500 \
  --output-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key question \
  --retain-batch-size 4 \
  --retain-max-steps 0 \
  --forget-max-steps 0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --lora-only

python src/tools/validate_dual_cf_artifact.py \
  --artifact-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key question
```

### Qwen - DUET merged artifact

```bash
export CUDA_VISIBLE_DEVICES=4

ART_ROOT=/data/home/vkropoti/unlearning/artifacts/dualcf/duet
mkdir -p "$ART_ROOT"

MODEL_CFG=configs/model/Qwen2.5-7B-Instruct.yaml
LORA_MODEL_CFG=configs/model/Qwen2.5-7B-Instruct-lora.yaml
BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct
SFT_MODEL_PATH=SwetieePawsss/DUET_ft_models
SFT_SUBFOLDER=qwen2.5-7b-instruct-tripunlamb-ft
ART_PREFIX="$ART_ROOT/duet_qwen_merged"

python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/DUET \
  --split 'city_forget_rare_5+city_forget_popular_5' \
  --output-path "${ART_PREFIX}_step1.jsonl" \
  --question-key question \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --max-new-tokens 32

python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files "${ART_PREFIX}_step1.jsonl" \
  --output-path "${ART_PREFIX}_step2.jsonl" \
  --question-key question \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$SFT_MODEL_PATH" \
  --tokenizer-path "$SFT_MODEL_PATH" \
  --model-subfolder "$SFT_SUBFOLDER" \
  --tokenizer-subfolder "$SFT_SUBFOLDER" \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0 \
  --w-mrd 0.0 \
  --w-stage 0.0 \
  --batch-size 32

python src/tools/score_attribution.py \
  --model-cfg "$LORA_MODEL_CFG" \
  --model-path "$SFT_MODEL_PATH" \
  --tokenizer-path "$SFT_MODEL_PATH" \
  --model-subfolder "$SFT_SUBFOLDER" \
  --tokenizer-subfolder "$SFT_SUBFOLDER" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${ART_PREFIX}_step2.jsonl" \
  --retain-dataset-path SwetieePawsss/DUET \
  --retain-split city_fast_retain_500 \
  --output-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key question \
  --retain-batch-size 4 \
  --retain-max-steps 0 \
  --forget-max-steps 0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --lora-only

python src/tools/validate_dual_cf_artifact.py \
  --artifact-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key question
```

### Gemma - DUET merged artifact

```bash
export CUDA_VISIBLE_DEVICES=4

ART_ROOT=/data/home/vkropoti/unlearning/artifacts/dualcf/duet
mkdir -p "$ART_ROOT"

MODEL_CFG=configs/model/gemma-7b-it.yaml
LORA_MODEL_CFG=configs/model/gemma-7b-it-lora.yaml
BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it
SFT_MODEL_PATH=SwetieePawsss/DUET_ft_models
SFT_SUBFOLDER=gemma-7b-it-tripunlamb-ft
ART_PREFIX="$ART_ROOT/duet_gemma_merged"

python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/DUET \
  --split 'city_forget_rare_5+city_forget_popular_5' \
  --output-path "${ART_PREFIX}_step1.jsonl" \
  --question-key question \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --max-new-tokens 32

python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files "${ART_PREFIX}_step1.jsonl" \
  --output-path "${ART_PREFIX}_step2.jsonl" \
  --question-key question \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$SFT_MODEL_PATH" \
  --tokenizer-path "$SFT_MODEL_PATH" \
  --model-subfolder "$SFT_SUBFOLDER" \
  --tokenizer-subfolder "$SFT_SUBFOLDER" \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0 \
  --w-mrd 0.0 \
  --w-stage 0.0 \
  --batch-size 32

python src/tools/score_attribution.py \
  --model-cfg "$LORA_MODEL_CFG" \
  --model-path "$SFT_MODEL_PATH" \
  --tokenizer-path "$SFT_MODEL_PATH" \
  --model-subfolder "$SFT_SUBFOLDER" \
  --tokenizer-subfolder "$SFT_SUBFOLDER" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${ART_PREFIX}_step2.jsonl" \
  --retain-dataset-path SwetieePawsss/DUET \
  --retain-split city_fast_retain_500 \
  --output-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key question \
  --retain-batch-size 4 \
  --retain-max-steps 0 \
  --forget-max-steps 0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --lora-only

python src/tools/validate_dual_cf_artifact.py \
  --artifact-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key question
```

## RWKU artifacts

Use the benchmark forget/retain banks directly:

- forget: `forget_level2`
- retain: `neighbor_level2`

### Llama - RWKU artifact

```bash
export CUDA_VISIBLE_DEVICES=4

ART_ROOT=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku
mkdir -p "$ART_ROOT"

MODEL_CFG=configs/model/Llama-3.1-8B-Instruct.yaml
LORA_MODEL_CFG=configs/model/Llama-3.1-8B-Instruct-lora.yaml
BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct
ART_PREFIX="$ART_ROOT/rwku_llama"

##done
python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/exp_r \
  --dataset-name forget_level2 \
  --split test \
  --output-path "${ART_PREFIX}_step1.jsonl" \
  --question-key query \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --max-new-tokens 32

##done
python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files "${ART_PREFIX}_step1.jsonl" \
  --output-path "${ART_PREFIX}_step2.jsonl" \
  --question-key query \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0 \
  --w-mrd 0.0 \
  --w-stage 0.0 \
  --batch-size 32

##done
python src/tools/score_attribution.py \
  --model-cfg "$LORA_MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${ART_PREFIX}_step2.jsonl" \
  --retain-dataset-path SwetieePawsss/exp_r \
  --retain-dataset-name neighbor_level2 \
  --retain-split test \
  --output-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key query \
  --retain-question-key query \
  --retain-batch-size 4 \
  --retain-max-steps 0 \
  --forget-max-steps 0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --lora-only

##done
python src/tools/validate_dual_cf_artifact.py \
  --artifact-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key query
```

### Qwen - RWKU artifact

```bash
export CUDA_VISIBLE_DEVICES=2

ART_ROOT=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku
mkdir -p "$ART_ROOT"

MODEL_CFG=configs/model/Qwen2.5-7B-Instruct.yaml
LORA_MODEL_CFG=configs/model/Qwen2.5-7B-Instruct-lora.yaml
BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct
ART_PREFIX="$ART_ROOT/rwku_qwen"

python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/exp_r \
  --dataset-name forget_level2 \
  --split test \
  --output-path "${ART_PREFIX}_step1.jsonl" \
  --question-key query \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --max-new-tokens 32

python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files "${ART_PREFIX}_step1.jsonl" \
  --output-path "${ART_PREFIX}_step2.jsonl" \
  --question-key query \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0 \
  --w-mrd 0.0 \
  --w-stage 0.0 \
  --batch-size 32

python src/tools/score_attribution.py \
  --model-cfg "$LORA_MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${ART_PREFIX}_step2.jsonl" \
  --retain-dataset-path SwetieePawsss/exp_r \
  --retain-dataset-name neighbor_level2 \
  --retain-split test \
  --output-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key query \
  --retain-question-key query \
  --retain-batch-size 4 \
  --retain-max-steps 0 \
  --forget-max-steps 0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --lora-only

python src/tools/validate_dual_cf_artifact.py \
  --artifact-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key query
```

### Gemma - RWKU artifact

```bash
export CUDA_VISIBLE_DEVICES=5

ART_ROOT=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku
mkdir -p "$ART_ROOT"

MODEL_CFG=configs/model/gemma-7b-it.yaml
LORA_MODEL_CFG=configs/model/gemma-7b-it-lora.yaml
BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it
ART_PREFIX="$ART_ROOT/rwku_gemma"

python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/exp_r \
  --dataset-name forget_level2 \
  --split test \
  --output-path "${ART_PREFIX}_step1.jsonl" \
  --question-key query \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --max-new-tokens 32

python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files "${ART_PREFIX}_step1.jsonl" \
  --output-path "${ART_PREFIX}_step2.jsonl" \
  --question-key query \
  --answer-key answer \
  --model-cfg "$MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0 \
  --w-mrd 0.0 \
  --w-stage 0.0 \
  --batch-size 32

python src/tools/score_attribution.py \
  --model-cfg "$LORA_MODEL_CFG" \
  --model-path "$BASE_MODEL_PATH" \
  --tokenizer-path "$BASE_MODEL_PATH" \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files "${ART_PREFIX}_step2.jsonl" \
  --retain-dataset-path SwetieePawsss/exp_r \
  --retain-dataset-name neighbor_level2 \
  --retain-split test \
  --output-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key query \
  --retain-question-key query \
  --retain-batch-size 4 \
  --retain-max-steps 0 \
  --forget-max-steps 0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.0 \
  --lora-only

python src/tools/validate_dual_cf_artifact.py \
  --artifact-path "${ART_PREFIX}_dualcf.jsonl" \
  --question-key query
```

## Training launches

These launches assume the validated artifacts above already exist.

### Llama - DUET (done)

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
CF_DATASET_PATH=json \
CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/duet_llama_merged_dualcf.jsonl \
CF_DATASET_SPLIT=train \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=2 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
BETAS=0.5 \
TAU_DS=0.5 \
TAU_AS=0.5 \
TEMP_DS=0.25 \
TEMP_AS=0.25 \
LAMBDA_NEG_MAXS=1.0 \
LAMBDA_RET_LOS=1.0 \
LAMBDA_RET_HIS=2.0 \
CF_WEIGHTS=1.0 \
RISK_FORGET_SCALES=0.5 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/dual_cf_duet.sh
```

### Qwen - DUET

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
CF_DATASET_PATH=json \
CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/duet_qwen_merged_dualcf.jsonl \
CF_DATASET_SPLIT=train \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=2 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
BETAS=0.5 \
TAU_DS=0.5 \
TAU_AS=0.5 \
TEMP_DS=0.25 \
TEMP_AS=0.25 \
LAMBDA_NEG_MAXS=1.0 \
LAMBDA_RET_LOS=1.0 \
LAMBDA_RET_HIS=2.0 \
CF_WEIGHTS=1.0 \
RISK_FORGET_SCALES=0.5 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/dual_cf_duet.sh
```

### Gemma - DUET

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
CF_DATASET_PATH=json \
CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/duet/duet_gemma_merged_dualcf.jsonl \
CF_DATASET_SPLIT=train \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=2 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
BETAS=0.5 \
TAU_DS=0.5 \
TAU_AS=0.5 \
TEMP_DS=0.25 \
TEMP_AS=0.25 \
LAMBDA_NEG_MAXS=1.0 \
LAMBDA_RET_LOS=1.0 \
LAMBDA_RET_HIS=2.0 \
CF_WEIGHTS=1.0 \
RISK_FORGET_SCALES=0.5 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/duet/dual_cf_duet.sh
```

### Llama - RWKU

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=4 \
BASE_MODEL=Llama-3.1-8B-Instruct \
MODEL_CONFIG=Llama-3.1-8B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Llama-3.1-8B-Instruct \
CF_DATASET_PATH=json \
CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku/rwku_llama_dualcf.jsonl \
CF_DATASET_NAME=null \
CF_DATASET_SPLIT=train \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=2 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
BETAS=0.5 \
TAU_DS=0.5 \
TAU_AS=0.5 \
TEMP_DS=0.25 \
TEMP_AS=0.25 \
LAMBDA_NEG_MAXS=1.0 \
LAMBDA_RET_LOS=1.0 \
LAMBDA_RET_HIS=2.0 \
CF_WEIGHTS=1.0 \
RISK_FORGET_SCALES=0.5 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/dual_cf_rwku.sh
```

### Qwen - RWKU

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2 \
BASE_MODEL=Qwen2.5-7B-Instruct \
MODEL_CONFIG=Qwen2.5-7B-Instruct-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/Qwen2.5-7B-Instruct \
CF_DATASET_PATH=json \
CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku/rwku_qwen_dualcf.jsonl \
CF_DATASET_NAME=null \
CF_DATASET_SPLIT=train \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=2 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
BETAS=0.5 \
TAU_DS=0.5 \
TAU_AS=0.5 \
TEMP_DS=0.25 \
TEMP_AS=0.25 \
LAMBDA_NEG_MAXS=1.0 \
LAMBDA_RET_LOS=1.0 \
LAMBDA_RET_HIS=2.0 \
CF_WEIGHTS=1.0 \
RISK_FORGET_SCALES=0.5 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/dual_cf_rwku.sh
```

### Gemma - RWKU

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=5 \
BASE_MODEL=gemma-7b-it \
MODEL_CONFIG=gemma-7b-it-lora \
HF_BASE_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
TOKENIZER_MODEL_PATH=/data/home/vkropoti/unlearning/models/BASE/gemma-7b-it \
CF_DATASET_PATH=json \
CF_DATASET_DATA_FILES=/data/home/vkropoti/unlearning/artifacts/dualcf/rwku/rwku_gemma_dualcf.jsonl \
CF_DATASET_NAME=null \
CF_DATASET_SPLIT=train \
PER_DEVICE_TRAIN_BS=16 \
GRAD_ACCUM=2 \
NUM_EPOCHS=2 \
LRS="1e-6 5e-6 1e-5 5e-5 1e-4" \
BETAS=0.5 \
TAU_DS=0.5 \
TAU_AS=0.5 \
TEMP_DS=0.25 \
TEMP_AS=0.25 \
LAMBDA_NEG_MAXS=1.0 \
LAMBDA_RET_LOS=1.0 \
LAMBDA_RET_HIS=2.0 \
CF_WEIGHTS=1.0 \
RISK_FORGET_SCALES=0.5 \
EVAL_BATCH_SIZE=64 \
DELETE_MODEL_SAFETENSORS_AFTER_EVAL=1 \
bash scripts/rwku/dual_cf_rwku.sh
```
