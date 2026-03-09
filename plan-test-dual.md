# DualCF GPU Test Plan

This file is the step-by-step validation plan for running DualCF on the GPU
server later. Do not run these steps now.

## 1. Goal

Validate that DualCF is operationally correct before any real sweep:

1. artifact files load correctly
2. routed metadata reaches the trainer
3. one-step training works on GPU
4. DUET end-to-end eval works
5. only after that, expand to POPQA and RWKU

## 2. Required inputs before any run

Prepare these merged forget-side artifacts first:

- `DUET_DUALCF_JSONL=/abs/path/to/duet_dualcf.jsonl`
- `POPQA_DUALCF_JSONL=/abs/path/to/popqa_dualcf.jsonl`
- `RWKU_DUALCF_JSONL=/abs/path/to/rwku_dualcf.jsonl`

Each JSONL row must contain:

```json
{
  "index": 17,
  "question": "...",
  "answer": "...",
  "alternate": "...",
  "difficulty_score": 0.73,
  "attribution_score": 0.18
}
```

Required checks on the artifact before training:

1. `alternate` exists for every row
2. `difficulty_score` exists for every row
3. `attribution_score` exists for every row
4. `index` exists for every row
5. scores are numeric
6. `difficulty_score` follows "higher = harder"
7. `attribution_score` follows "higher = riskier"

## 3. Server preflight

Run these first on the GPU server:

```bash
cd /path/to/open-unlearning
git status --short
nvidia-smi
python -V
```

Confirm:

1. correct branch / working tree
2. expected GPU is visible
3. CUDA environment is the one used for the repo

## 4. Artifact sanity checks

Check that the JSONL files look correct before calling the trainer:

```bash
head -n 2 "${DUET_DUALCF_JSONL}"
head -n 2 "${POPQA_DUALCF_JSONL}"
head -n 2 "${RWKU_DUALCF_JSONL}"
```

Then run one full-file integrity scan:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

paths = [
    Path(os.environ["DUET_DUALCF_JSONL"]),
    Path(os.environ["POPQA_DUALCF_JSONL"]),
    Path(os.environ["RWKU_DUALCF_JSONL"]),
]
required = {"index", "question", "answer", "alternate", "difficulty_score", "attribution_score"}

for path in paths:
    seen_indices = set()
    duplicate_indices = set()
    bad_rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = json.loads(line)
            missing = sorted(required - set(row))
            if missing:
                bad_rows.append((line_no, f"missing keys: {missing}"))
                continue

            index = row["index"]
            if index in seen_indices:
                duplicate_indices.add(index)
            seen_indices.add(index)

            if not isinstance(row["alternate"], str) or not row["alternate"].strip():
                bad_rows.append((line_no, "empty alternate"))

            for score_key in ("difficulty_score", "attribution_score"):
                value = row[score_key]
                if not isinstance(value, (int, float)):
                    bad_rows.append((line_no, f"{score_key} is not numeric: {value!r}"))

    print(path)
    print("  rows=", len(seen_indices))
    print("  duplicates=", sorted(duplicate_indices))
    print("  bad_rows=", bad_rows[:10])
PY
```

Expected result:

1. `duplicates=[]` for each file
2. `bad_rows=[]` for each file

## 5. DUET direct 1-step smoke test

This is the first real test. Use a small 1B LoRA model and a 2-sample slice.

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=duet_dualcf_smoke \
  "forget_split=city_forget_rare_5[:2]" \
  "retain_split=city_fast_retain_500[:2]" \
  cf_dataset_path=json \
  cf_dataset_data_files="${DUET_DUALCF_JSONL}" \
  cf_dataset_split=train \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/duet_dualcf_smoke
```

Check immediately after the run:

1. `dualcf_cf_loss` appears in logs
2. `dualcf_neg_loss` appears in logs
3. `dualcf_forget_loss` appears in logs
4. `dualcf_alpha_eff` appears in logs
5. no `KeyError` for `difficulty_score`
6. no `KeyError` for `attribution_score`
7. no `_score_tensor` batch-size mismatch
8. adapter checkpoint exists under `/tmp/duet_dualcf_smoke`

## 6. DUET launcher smoke test

After the direct one-step command works, validate the actual launcher entrypoint.

```bash
CF_DATASET_PATH=json \
CF_DATASET_DATA_FILES="${DUET_DUALCF_JSONL}" \
CF_DATASET_SPLIT='train[:2]' \
USE_SFT_BASE=0 \
BASE_MODEL=Llama-3.2-1B-Instruct \
HF_BASE_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct \
MODEL_CONFIG=Llama-3.2-1B-Instruct-lora \
TOKENIZER_MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct \
NUM_EPOCHS=1 \
MAX_STEPS=1 \
PER_DEVICE_TRAIN_BS=1 \
GRAD_ACCUM=1 \
LRS=1e-5 \
BETAS=0.5 \
TAU_DS=0.5 \
TAU_AS=0.5 \
TEMP_DS=0.25 \
TEMP_AS=0.25 \
LAMBDA_NEG_MAXS=1.0 \
LAMBDA_RET_HIS=2.0 \
RISK_FORGET_SCALES=0.5 \
FORCE_RERUN=1 \
FORGET_SPLIT_OVERRIDE="city_forget_rare_5[:2]" \
RETAIN_SPLIT_OVERRIDE="city_fast_retain_500[:2]" \
FORGET_LABEL_OVERRIDE=city_forget_rare_5_smoke \
bash scripts/duet/dual_cf_duet.sh
```

Check:

1. script does not complain about placeholder artifact paths
2. script resolves local JSON mode with `cf_dataset_split='train[:2]'`
3. script uses the HF 1B base instead of the local 8B SFT base
4. script respects `MAX_STEPS=1`
5. training launches
6. evaluation launches

## 7. DUET small functional run

Only after the 1-step smoke passes, do a short DUET run with slightly larger
slices. This is a train step followed by an explicit eval step.

### Train

Suggested short train run:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=duet_dualcf_small \
  "forget_split=city_forget_rare_5[:32]" \
  "retain_split=city_fast_retain_500[:64]" \
  cf_dataset_path=json \
  cf_dataset_data_files="${DUET_DUALCF_JSONL}" \
  cf_dataset_split=train \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.num_train_epochs=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/duet_dualcf_small
```

### Eval

Then run eval separately:

```bash
python src/eval.py \
  experiment=eval/duet/default.yaml \
  model=Llama-3.2-1B-Instruct-lora \
  forget_split='city_forget_rare_5[:32]' \
  holdout_split='city_fast_retain_500[:64]' \
  task_name=duet_dualcf_small \
  model.model_args.pretrained_model_name_or_path=/tmp/duet_dualcf_small \
  model.model_args.base_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  model.tokenizer_args.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
  model.model_args.device_map=auto \
  model.model_args.low_cpu_mem_usage=true \
  paths.output_dir=/tmp/duet_dualcf_small/evals \
  retain_logs_path=null
```

Check after train + eval:

1. loss stays finite
2. no OOM
3. train output is written to `/tmp/duet_dualcf_small`
4. eval completes
5. eval files are written to `/tmp/duet_dualcf_small/evals`

## 8. DUET ablation checks

Run only after the small functional run works.

### Full DualCF

```bash
trainer.method_args.disable_difficulty_route=false \
trainer.method_args.disable_attribution_route=false
```

### Difficulty-only

```bash
trainer.method_args.disable_difficulty_route=false \
trainer.method_args.disable_attribution_route=true
```

### Attribution-only

```bash
trainer.method_args.disable_difficulty_route=true \
trainer.method_args.disable_attribution_route=false
```

### Uniform counterfactual baseline

Use the existing `DPO` path with the same counterfactual artifact.

Check for all ablations:

1. logs are present
2. training completes
3. no silent metadata drop

## 9. POPQA smoke sequence

Do not start here. Start only after DUET is stable.

### Direct one-step smoke

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/popqa/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=popqa_dualcf_smoke \
  "forget_split=rare_forget5_sum[:2]" \
  "retain_split=fast_retain_500[:2]" \
  cf_dataset_path=json \
  cf_dataset_data_files="${POPQA_DUALCF_JSONL}" \
  cf_dataset_split=train \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/popqa_dualcf_smoke
```

Check the same items as DUET.

## 10. RWKU smoke sequence

Only after DUET and POPQA pass.

### Direct one-step smoke

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/rwku/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=rwku_dualcf_smoke \
  "forget_split=forget_level2[:2]" \
  "retain_split=neighbor_level2[:2]" \
  cf_dataset_path=json \
  cf_dataset_data_files="${RWKU_DUALCF_JSONL}" \
  cf_dataset_name=null \
  cf_dataset_split=train \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/rwku_dualcf_smoke
```

Extra RWKU checks:

1. `question_key=query` is respected
2. proxy-retain attribution artifact is the one intended for RWKU
3. no accidental use of the benchmark retain split as if it were the true retain set

## 11. First real 8B DUET run

Only after all 1B smokes pass:

1. switch to the intended 8B model
2. keep batch size conservative
3. run one benchmark split only
4. save logs and summary

Recommended first target:

1. DUET
2. full DualCF
3. one forget split
4. one retain split

## 12. Sweep gating rules

Do not start sweeps until all of these are true:

1. DUET 1-step smoke passed
2. DUET small functional run passed
3. DUET eval completed
4. POPQA 1-step smoke passed
5. RWKU 1-step smoke passed
6. artifact schemas were manually checked
7. routed logs appeared in all runs

## 13. Failure triage order

If a smoke run fails, debug in this order:

1. `cf_dataset_split` is wrong for local JSON mode
2. `cf_dataset_data_files` points to the wrong file
3. artifact is missing `alternate`
4. artifact is missing `difficulty_score`
5. artifact is missing `attribution_score`
6. artifact is missing `index`
7. dataset/collator dropped metadata
8. batch-size mismatch reached `_score_tensor`
9. generated alternates are low quality or not truly counterfactual

## 14. What to save after each test

For every smoke or small run, keep:

1. exact command used
2. stdout/stderr log
3. output directory
4. checkpoint path
5. summary/eval JSON
6. note whether `dualcf_*` logs appeared
