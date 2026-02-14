# Part 1 OpenUnlearning - My Run Log

## 1) Activate environment

```bash
source /data/home/vkropoti/unlearning-venv/bin/activate
```

## 2) Exports used

```bash
export TOFU_FT_MODEL_PATH="/data/home/vkropoti/unlearning/models/tofu_Llama-3.2-1B-Instruct_full"
export HF_HOME="/data/home/vkropoti/unlearning/data"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_MODE=disabled
```

## 3) Sanity checks

### 3.1 GPU check

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"
```

Output:

```text
torch 2.4.1+cu121
cuda True
gpu NVIDIA H100 80GB HBM3
```

### 3.2 Local tokenizer check

```bash
python -c "import os; from transformers import AutoTokenizer; model_path=os.environ.get('BASE_LLAMA_PATH') or os.environ.get('TOFU_FT_MODEL_PATH'); assert model_path, 'Set BASE_LLAMA_PATH or TOFU_FT_MODEL_PATH'; assert os.path.isdir(model_path), f'Model dir not found: {model_path}'; AutoTokenizer.from_pretrained(model_path, local_files_only=True); print('tokenizer OK:', model_path)"
```

Output:

```text
tokenizer OK: /data/home/vkropoti/unlearning/models/tofu_Llama-3.2-1B-Instruct_full
/data/home/vkropoti/unlearning-venv/lib64/python3.11/site-packages/transformers/utils/hub.py:128: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
```

### 3.3 Datasets cache listing

```bash
ls -lah "$HF_HOME/datasets" | head
```

Output:

```text
total 28K
drwxr-xr-x.  4 vkropoti hpshm   16K Feb 10 23:40 .
drwx------.  4 vkropoti hpshm  103 Feb 10 23:39 ..
drwxr-xr-x. 61 vkropoti hpshm  4.0K Feb 10 22:45 cais__mmlu
drwxr-xr-x. 20 vkropoti hpshm  4.0K Feb 10 22:39 locuslab__tofu
-rw-r--r--.  1 vkropoti hpshm     0 Feb 10 22:42 _Users_valerii.kropotin_HQD_Diploma_open-unlearning_offline_zips_tmp_mmlu_cache_datasets_cais___mmlu_abstract_algebra_0.0.0_c30699e8356da336a370243923dbaf21066bb9fe.lock
-rw-r--r--.  1 vkropoti hpshm     0 Feb 10 22:42 _Users_valerii.kropotin_HQD_Diploma_open-unlearning_offline_zips_tmp_mmlu_cache_datasets_cais___mmlu_all_0.0.0_c30699e8356da336a370243923dbaf21066bb9fe.lock
-rw-r--r--.  1 vkropoti hpshm     0 Feb 10 22:42 _Users_valerii.kropotin_HQD_Diploma_open-unlearning_offline_zips_tmp_mmlu_cache_datasets_cais___mmlu_anatomy_0.0.0_c30699e8356da336a370243923dbaf21066bb9fe.lock
-rw-r--r--.  1 vkropoti hpshm     0 Feb 10 22:42 _Users_valerii.kropotin_HQD_Diploma_open-unlearning_offline_zips_tmp_mmlu_cache_datasets_cais___mmlu_astronomy_0.0.0_c30699e8356da336a370243923dbaf21066bb9fe.lock
-rw-r--r--.  1 vkropoti hpshm     0 Feb 10 22:42 _Users_valerii.kropotin_HQD_Diploma_open-unlearning_offline_zips_tmp_mmlu_cache_datasets_cais___mmlu_auxiliary_train_0.0.0_c30699e8356da336a370243923dbaf21066bb9fe.lock
```

### 3.4 Hub cache listing

```bash
ls -lah "$HF_HOME/hub" | head
```

Output:

```text
total 4.0K
drwxr-xr-x. 5 vkropoti hpshm  119 Feb 11 13:23 .
drwx------. 4 vkropoti hpshm  103 Feb 10 23:39 ..
drwxr-xr-x. 6 vkropoti hpshm   85 Feb 10 22:45 datasets--cais--mmlu
drwxr-xr-x. 6 vkropoti hpshm   85 Feb 10 22:39 datasets--locuslab--TOFU
drwxr-xr-x. 4 vkropoti hpshm   78 Feb 10 23:40 .locks
-rw-------. 1 vkropoti hpshm    1 Feb 11 13:23 version.txt
```

## 4) Troubleshooting log (what failed and how it was fixed)

### 4.1 `ModuleNotFoundError: No module named 'lm_eval'`

Checks:

```bash
python - <<'PY'
import importlib.util
print("lm_eval importable:", importlib.util.find_spec("lm_eval") is not None)
PY
python -m pip show lm-eval || echo "lm-eval is not installed"
```

Fix:

```bash
python -m pip install -e ".[lm-eval]"
```

### 4.2 FlashAttention issue (`flash_attn` missing while FA3-only was present)

Observed:
- `flash_attn_interface` import worked.
- `flash_attn` import failed.
- `transformers==4.45.1` in this setup expects FA2 package import path (`flash_attn`) for `attn_implementation=flash_attention_2`.

Fix:
- Build/install FA2 (`flash_attn==2.6.3`) in the same venv.

Verification:

```bash
python - <<'PY'
import importlib.util
for m in ["flash_attn", "flash_attn_interface", "flashattn_hopper_cuda"]:
    print(m, bool(importlib.util.find_spec(m)))
PY
```

### 4.3 Local model path mismatch

Problem:
- `Incorrect path_or_model_id` happened because the provided local path string did not resolve.

Checks:

```bash
MODEL_DIR="/data/home/vkropoti/unlearning/models/tofu_Llama-3.2-1B-Instruct_full"
python - <<'PY'
from pathlib import Path
p = Path("/data/home/vkropoti/unlearning/models/tofu_Llama-3.2-1B-Instruct_full")
print("exists:", p.exists(), ", is_dir:", p.is_dir())
print("config.json:", (p / "config.json").exists())
print("tokenizer_config.json:", (p / "tokenizer_config.json").exists())
PY
```

### 4.4 Hydra override key mistakes

Common mistakes that caused errors:
- `attention_implementation` (wrong)
- correct key is `attn_implementation`
- `local_files_only` must be added with `+...` because key not in structured config

Correct pattern:

```bash
model.model_args.attn_implementation=flash_attention_2
+model.model_args.local_files_only=true
+model.tokenizer_args.local_files_only=true
```

### 4.5 TOFU cache + `datasets==3.0.1` metadata incompatibility

Problem:
- Loading `locuslab/TOFU` from local cache failed with:
- `TypeError: must be called with a dataclass type or instance`

Reason:
- Cached TOFU metadata contained `"_type": "List"` where `datasets==3.0.1` expects `"_type": "Sequence"`.

Fix used:

```bash
python scripts/fix_tofu_cache_datasets_3_0_1.py \
  --datasets-cache "$HF_DATASETS_CACHE" \
  --dataset-dir-name locuslab__tofu
```

Notes:
- If your cache directory is `locuslab___tofu` (triple underscore), pass that name instead.

### 4.6 Reference metrics crash from empty `retain_logs_path`

Problem:
- Run failed when override list showed `retain_logs_path=` (empty value).
- Reference-based metrics (`forget_quality`, `privleak`) require retain eval logs JSON.

Checks:

```bash
find saves/eval -maxdepth 3 -type f -name "TOFU_EVAL.json"
```

Set:

```bash
export RETAIN_LOGS="$PWD/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
test -f "$RETAIN_LOGS"
```

## 5) Final working eval command (with reference metrics)

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

export MODEL_DIR="/data/home/vkropoti/unlearning/models/tofu_Llama-3.2-1B-Instruct_full"
export TOKENIZER_DIR="$MODEL_DIR"
export RETAIN_LOGS="$PWD/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"

CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget10 holdout_split=holdout10 \
  model.model_args.attn_implementation=flash_attention_2 \
  model.model_args.pretrained_model_name_or_path="$MODEL_DIR" \
  +model.model_args.local_files_only=true \
  model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_DIR" \
  +model.tokenizer_args.local_files_only=true \
  retain_logs_path="$RETAIN_LOGS" \
  eval.tofu.overwrite=true \
  task_name="EVAL_tofu_full_Llama-3.2-1B-Instruct_TOFU_forget10_with_ref"
```

## 6) Final output snapshot

```text
root
extraction_strength 0.4330492058933082
forget_Q_A_Prob 0.7430841289460659
forget_Q_A_ROUGE 0.479102541080718
forget_quality 3.4004838027527454e-21
model_utility 0.6093719454201277
privleak -99.35210866767595
```

## 7) Unlearning training runs (executed, but metrics low)

### 7.1 GradAscent run

Run command:

```bash
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=GradAscent \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget10 retain_split=retain90 holdout_split=holdout10 \
  model.model_args.attn_implementation=flash_attention_2 \
  model.model_args.pretrained_model_name_or_path="$TOFU_FT_MODEL_PATH" \
  +model.model_args.local_files_only=true \
  model.tokenizer_args.pretrained_model_name_or_path="$TOFU_FT_MODEL_PATH" \
  +model.tokenizer_args.local_files_only=true \
  retain_logs_path=null \
  task_name="UNLEARN_GradAscent_Llama-3.2-1B-Instruct_forget10"
```

Eval snapshot for `checkpoint-60`:

```text
root
extraction_strength 0.03250892997513522
forget_Q_A_Prob 3.559116883352978e-30
forget_Q_A_ROUGE 0
model_utility 0
privleak -14.171249997165742
```

### 7.2 NPO run

Run command:

```bash
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=NPO \
  model=Llama-3.2-1B-Instruct \
  forget_split=forget10 retain_split=retain90 holdout_split=holdout10 \
  model.model_args.attn_implementation=flash_attention_2 \
  model.model_args.pretrained_model_name_or_path="$TOFU_FT_MODEL_PATH" \
  +model.model_args.local_files_only=true \
  model.tokenizer_args.pretrained_model_name_or_path="$TOFU_FT_MODEL_PATH" \
  +model.tokenizer_args.local_files_only=true \
  trainer.args.learning_rate=5e-6 \
  trainer.args.num_train_epochs=3 \
  trainer.method_args.beta=0.1 \
  task_name="UNLEARN_NPO_Llama-3.2-1B-Instruct_forget10"
```

Eval snapshot:

```text
root
extraction_strength 0.16343589598900693
forget_Q_A_Prob 0.2076463395729661
forget_Q_A_ROUGE 0.02471596921874099
model_utility 0.13166610058079445
privleak -82.85749998342848
```

## 8) Section 5 + Section 7 commands for NPO checkpoint and reference eval

```bash
cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

export MODEL_NAME="Llama-3.2-1B-Instruct"
export NPO_TASK="UNLEARN_NPO_${MODEL_NAME}_forget10"
export RETAIN_LOGS="$PWD/saves/eval/tofu_${MODEL_NAME}_retain90/TOFU_EVAL.json"

# 5.1 Inspect unlearning output folder
ls -lah saves/unlearn/
ls -lah "saves/unlearn/${NPO_TASK}"

# 5.2 Find candidate checkpoints that contain model weights
find "saves/unlearn/${NPO_TASK}" -maxdepth 4 -type f \
  \( -name "model.safetensors" -o -name "pytorch_model*.bin" \) -print

# 5.3 For this NPO run, final model is saved in the task root directory
export NPO_MODEL_PATH="saves/unlearn/${NPO_TASK}"
test -f "${NPO_MODEL_PATH}/config.json" || echo "config.json not found in ${NPO_MODEL_PATH}"
test -f "$RETAIN_LOGS" || echo "retain logs missing: $RETAIN_LOGS"

# 7.1 Evaluate NPO checkpoint with reference metrics
CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=${MODEL_NAME} \
  forget_split=forget10 holdout_split=holdout10 \
  model.model_args.attn_implementation=flash_attention_2 \
  model.model_args.pretrained_model_name_or_path="${NPO_MODEL_PATH}" \
  +model.model_args.local_files_only=true \
  model.tokenizer_args.pretrained_model_name_or_path="${NPO_MODEL_PATH}" \
  +model.tokenizer_args.local_files_only=true \
  retain_logs_path="${RETAIN_LOGS}" \
  eval.tofu.overwrite=true \
  task_name="EVAL_unlearned_NPO_${MODEL_NAME}_TOFU_forget10_with_ref"
```

## 9) NPO pre-run checks and final reference eval (actual run)

Pre-run checks:

```bash
export MODEL_NAME="Llama-3.2-1B-Instruct"
export NPO_TASK="UNLEARN_NPO_${MODEL_NAME}_forget10"
export RETAIN_LOGS="$PWD/saves/eval/tofu_${MODEL_NAME}_retain90/TOFU_EVAL.json"

ls -lah saves/unlearn/
ls -lah "saves/unlearn/${NPO_TASK}"

find "saves/unlearn/${NPO_TASK}" -maxdepth 4 -type f \
  \( -name "model.safetensors" -o -name "pytorch_model*.bin" \) -print

export NPO_MODEL_PATH="saves/unlearn/${NPO_TASK}/checkpoint-36"
test -f "${NPO_MODEL_PATH}/config.json" || echo "config.json not found in ${NPO_MODEL_PATH}"

export NPO_MODEL_PATH="saves/unlearn/${NPO_TASK}"
test -f "${NPO_MODEL_PATH}/config.json" || echo "config.json not found in ${NPO_MODEL_PATH}"
test -f "$RETAIN_LOGS" || echo "retain logs missing: $RETAIN_LOGS"
```

Observed:
- `checkpoint-36` did not contain `config.json`.
- Final model to evaluate is the run root:
  `saves/unlearn/UNLEARN_NPO_Llama-3.2-1B-Instruct_forget10`.

Final eval metrics with reference logs:

```text
root
extraction_strength 0.16343589598900693
forget_Q_A_Prob 0.2076463395729661
forget_Q_A_ROUGE 0.02471596921874099
forget_quality 2.241003819205479e-17
model_utility 0.13166610058079445
privleak -86.11690387542394
```
