# Part 1 runbook (offline / no internet) — OpenUnlearning + TOFU + (optional) MMLU

This runbook follows the standard OpenUnlearning pipeline:

1) (optional) fine-tune a base model on TOFU  
2) unlearn a forget split  
3) evaluate forgetting + retention (start with ROUGE-L recall)  
4) (optional) run a general benchmark like MMLU  

You said you already have locally (no HF internet needed):

- `tofu_Llama-3.2-1B-Instruct_full` (fine-tuned target model)
- `Llama-3.2-1B-Instruct` (base model)
- TOFU dataset cached locally
- MMLU dataset cached locally

---

## 0) Assumptions / variables

From the repo root (the cloned `open-unlearning`), set:

```bash
# ===== EDIT THESE PATHS =====
export BASE_LLAMA_PATH="/path/to/Llama-3.2-1B-Instruct"                  # local folder
export TOFU_FT_MODEL_PATH="/path/to/tofu_Llama-3.2-1B-Instruct_full"     # local folder (your finetuned target model)

# Attention backend for Hugging Face model loading.
# OpenUnlearning model configs default to flash_attention_2.
# Use flash_attention_3 on H100 if you installed FA3 only.
# Use sdpa if you do not want flash-attn dependencies.
export ATTN_IMPL="flash_attention_2"

# If your datasets are cached in a HuggingFace-style cache:
export HF_HOME="/path/to/hf_home"        # contains hub/ and datasets/
# Example:
#   $HF_HOME/hub/...
#   $HF_HOME/datasets/...

# If you also copied OpenUnlearning's precomputed retain logs (optional):
# export RETAIN_LOGS="/path/to/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
```

If you *do not* use an HF-style cache for datasets/models, you can still run as long as:
- `pretrained_model_name_or_path` points to a valid local model directory, and
- your code/configs load datasets from local disk.  
(For OpenUnlearning + datasets library, HF caches are easiest for true offline mode.)

---

## 1) Force “offline” mode (recommended)

```bash
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0   # choose GPU index(es), e.g. "0" or "0,1"

# Point everything to your local cache
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Avoid WANDB trying to connect
export WANDB_MODE=disabled
```

---

## 2) Quick sanity checks (GPU + local assets)

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -c "import os; from transformers import AutoTokenizer; model_path=os.environ.get('BASE_LLAMA_PATH') or os.environ.get('TOFU_FT_MODEL_PATH'); assert model_path, 'Set BASE_LLAMA_PATH or TOFU_FT_MODEL_PATH'; assert os.path.isdir(model_path), f'Model dir not found: {model_path}'; AutoTokenizer.from_pretrained(model_path, local_files_only=True); print('tokenizer OK:', model_path)"
```

Optional: confirm your datasets are visible in the HF cache (this is just a filesystem check):

```bash
ls -lah "$HF_HOME/datasets" | head
ls -lah "$HF_HOME/hub" | head
```

If TOFU loading fails with
`TypeError: must be called with a dataclass type or instance`
while using `datasets==3.0.1`, patch TOFU cache metadata once:

```bash
python scripts/fix_tofu_cache_datasets_3_0_1.py \
  --datasets-cache "$HF_DATASETS_CACHE" \
  --dataset-dir-name locuslab___tofu
```

---

## 3) Evaluate baselines on TOFU (ROUGE-L recall)

### 3.1 Evaluate the *base* model (Llama-3.2-1B-Instruct) on TOFU

```bash
export MODEL_NAME="Llama-3.2-1B-Instruct"

python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=${MODEL_NAME} \
  model.model_args.attn_implementation="${ATTN_IMPL}" \
  model.model_args.pretrained_model_name_or_path="${BASE_LLAMA_PATH}" \
  retain_logs_path=null \
  task_name="EVAL_base_${MODEL_NAME}_TOFU_forget10"
```

### 3.2 Evaluate the *fine-tuned* TOFU model (your target model before unlearning)

```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=${MODEL_NAME} \
  model.model_args.attn_implementation="${ATTN_IMPL}" \
  model.model_args.pretrained_model_name_or_path="${TOFU_FT_MODEL_PATH}" \
  retain_logs_path=null \
  task_name="EVAL_tofu_full_${MODEL_NAME}_TOFU_forget10"
```

**Notes**
- `retain_logs_path=null` is intentional for a minimal Part 1 setup (ROUGE-L recall only).
- If you want reference-based metrics like `forget_quality`, you will need retain logs and must set `retain_logs_path=...` (see §7).

---

## 4) Run unlearning (at least one method)

OpenUnlearning uses Hydra configs and `trainer=...` selects the unlearning algorithm.

### 4.1 GradAscent (simple baseline)

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 \
  trainer=GradAscent \
  model.model_args.attn_implementation="${ATTN_IMPL}" \
  model.model_args.pretrained_model_name_or_path="${TOFU_FT_MODEL_PATH}" \
  task_name="UNLEARN_GradAscent_${MODEL_NAME}_forget10"
```

### 4.2 (Optional) NPO (often more stable than pure GradAscent)

```bash
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 \
  trainer=NPO \
  model.model_args.attn_implementation="${ATTN_IMPL}" \
  model.model_args.pretrained_model_name_or_path="${TOFU_FT_MODEL_PATH}" \
  task_name="UNLEARN_NPO_${MODEL_NAME}_forget10"
```

---

## 5) Find the saved unlearned checkpoint path

Outputs are written to:

```bash
ls -lah saves/unlearn/
ls -lah "saves/unlearn/UNLEARN_GradAscent_${MODEL_NAME}_forget10"
```

If you don’t know which folder contains the final HF checkpoint, search:

```bash
find "saves/unlearn/UNLEARN_GradAscent_${MODEL_NAME}_forget10" -maxdepth 4 -type f \( -name "pytorch_model*.bin" -o -name "model.safetensors" \) -print
```

Pick the directory that contains `config.json`, tokenizer files, and model weights — that’s your `UNLEARNED_MODEL_PATH`.

Example:

```bash
export UNLEARNED_MODEL_PATH="saves/unlearn/UNLEARN_GradAscent_${MODEL_NAME}_forget10/<SOME_SUBFOLDER>"
```

---

## 6) Evaluate the unlearned model on TOFU

```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=${MODEL_NAME} \
  model.model_args.attn_implementation="${ATTN_IMPL}" \
  model.model_args.pretrained_model_name_or_path="${UNLEARNED_MODEL_PATH}" \
  retain_logs_path=null \
  task_name="EVAL_unlearned_GradAscent_${MODEL_NAME}_TOFU_forget10"
```

Repeat for NPO if you ran it.

---

## 7) (Optional) Enable reference-based TOFU metrics (needs retain logs)

OpenUnlearning can compute reference-based metrics (e.g., `forget_quality`) if you provide the retain model’s eval logs.

### Option A (recommended): download / generate retain logs on an internet machine
On a machine **with** internet:
- run `python setup_data.py --eval` inside the repo to populate `saves/eval/...`
- transfer `saves/eval/` to your offline GPU node

Then on the offline node, set:

```bash
export RETAIN_LOGS="saves/eval/tofu_${MODEL_NAME}_retain90/TOFU_EVAL.json"
```

And re-run evaluation with:

```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=${MODEL_NAME} \
  model.model_args.attn_implementation="${ATTN_IMPL}" \
  model.model_args.pretrained_model_name_or_path="${UNLEARNED_MODEL_PATH}" \
  retain_logs_path="${RETAIN_LOGS}" \
  task_name="EVAL_unlearned_with_retainlogs_${MODEL_NAME}_TOFU_forget10"
```

---

## 8) (Optional) Run MMLU after unlearning

### 8.1 Using lm-evaluation-harness CLI (direct)

If your environment has `lm-eval` installed (OpenUnlearning suggests `pip install .[lm-eval]`),
you can run MMLU directly on the unlearned model.

**Newer CLI style (lm-eval v0.4.10+):**
```bash
lm-eval run \
  --model hf \
  --model_args pretrained="${UNLEARNED_MODEL_PATH}",dtype=bfloat16 \
  --tasks mmlu \
  --device cuda \
  --output_path "saves/lm_eval/MMLU_unlearned_${MODEL_NAME}"
```

**Older CLI style (if your lm-eval is older):**
```bash
lm-eval \
  --model hf \
  --model_args pretrained="${UNLEARNED_MODEL_PATH}",dtype=bfloat16 \
  --tasks mmlu \
  --device cuda \
  --output_path "saves/lm_eval/MMLU_unlearned_${MODEL_NAME}"
```

For offline execution, it is critical that the MMLU dataset is already present in the HF datasets cache (`$HF_DATASETS_CACHE`) and that `HF_DATASETS_OFFLINE=1` is set.

---

## 9) Where to look for results (what to report)

### 9.1 Evaluation outputs (TOFU)

Evaluation outputs go under:
- `saves/eval/<task_name>/...`

Example:
```bash
ls -lah "saves/eval/EVAL_tofu_full_${MODEL_NAME}_TOFU_forget10"
```

Quick search for ROUGE values:
```bash
grep -Rni "rouge" "saves/eval/EVAL_tofu_full_${MODEL_NAME}_TOFU_forget10" | head -n 50
grep -Rni "rouge" "saves/eval/EVAL_unlearned_GradAscent_${MODEL_NAME}_TOFU_forget10" | head -n 50
```

### 9.2 Unlearning outputs (checkpoints + logs)

Unlearning outputs go under:
- `saves/unlearn/<task_name>/...`

Example:
```bash
ls -lah "saves/unlearn/UNLEARN_GradAscent_${MODEL_NAME}_forget10"
```

### 9.3 What to put in your “Part 1” table

At minimum, include rows for:
- base model (no tofu finetune)
- tofu finetuned model (before unlearning)
- unlearned model (after unlearning)

And columns for:
- TOFU forget split ROUGE-L recall
- TOFU holdout/retain split ROUGE-L recall (as “utility/retain”)
- (optional) MMLU accuracy (aggregate)
