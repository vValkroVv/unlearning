# DualCF GPU Test Plan

This file is the step-by-step validation and first-run plan for DualCF on the
GPU server later. Do not run these steps now.

## 1. Goal

Validate that DualCF is scientifically and operationally correct before any
real sweep:

1. artifact files are built for the exact benchmark slice being compared
2. artifact files pass stronger schema and provenance checks
3. routed metadata reaches the trainer
4. one-step training works on GPU
5. DUET end-to-end eval works on controlled slices first
6. only after DUET is stable, move to RWKU

## 2. Critical local JSON rule

In local JSON/JSONL mode, the DualCF forget training set is controlled by:

- `cf_dataset_path`
- `cf_dataset_data_files`
- `cf_dataset_split`

It is not controlled by `forget_split`.

That means:

1. slicing `forget_split=city_forget_rare_5[:2]` does not shrink DualCF training
   when `cf_dataset_path=json`
2. direct smoke tests must slice `cf_dataset_split`, for example
   `cf_dataset_split='train[:2]'`
3. small functional runs must also slice `cf_dataset_split`
4. `forget_split` still matters for benchmark identity and downstream eval config

Fair-comparison implication:

1. rare-only DUET comparison requires a rare-only DualCF artifact
2. popular-only DUET comparison requires a popular-only DualCF artifact
3. merged DUET comparison requires a merged DualCF artifact
4. never compare DualCF trained on merged JSONL against GA/NPO trained on
   rare-only or popular-only benchmark splits

## 3. Execution order

Use this order:

1. DUET rare-only
2. DUET popular-only
3. DUET merged rare+popular
4. RWKU

POPQA is optional for extra plumbing coverage, but it is not on the critical
path for the main DualCF diploma claim.

## 4. Recommended helper additions before main campaign

Add these before the main GPU campaign if possible:

1. `src/tools/validate_dual_cf_artifact.py`
2. `scripts/duet/prepare_dual_cf_duet.sh`
3. `scripts/rwku/prepare_dual_cf_rwku.sh`

Why:

1. artifact preparation belongs outside `DualCF.compute_loss()`
2. attribution scoring performs extra backward passes over a retain bank
3. validation should fail fast before expensive training starts
4. wrapper scripts reduce human error when preparing split-matched artifacts

If these helpers are not added before the first GPU pass, perform the same
checks manually and save all commands and logs.

## 5. Required artifact schema

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

Required semantic assumptions:

1. `alternate` is a real counterfactual and not just a copy of `answer`
2. `difficulty_score` follows "higher = harder"
3. `attribution_score` follows "higher = riskier"
4. `attribution_score` is min-max normalized and should sit in `[0, 1]`
5. `difficulty_score` should usually sit in `[0, 1]`, or its wider range must be
   documented if raw stage priors are used

## 6. Provenance sidecar for each artifact

For every final artifact, keep a sidecar JSON file such as:

- `duet_rare_dualcf.jsonl.meta.json`
- `duet_popular_dualcf.jsonl.meta.json`
- `duet_merged_dualcf.jsonl.meta.json`
- `rwku_dualcf.jsonl.meta.json`

Record at least:

1. source dataset path / name / split
2. `question_key`, `answer_key`, `answer_index`
3. counterfactual source type: column / jsonl / generator model
4. generator model id and tokenizer id
5. difficulty recipe: active columns and weights
6. attribution recipe: retain dataset path / name / split, `alignment`,
   `retain_max_steps`, `lora_only`
7. git commit
8. build timestamp

## 7. Server preflight

Run these first on the GPU server:

```bash
ln -sfn /data/home/vkropoti/unlearning/SwetieePawsss \
  /home/vkropoti/diploma/open-unlearning/SwetieePawsss

cd /home/vkropoti/diploma/open-unlearning
source /data/home/vkropoti/unlearning-venv/bin/activate

export HF_HOME=/data/home/vkropoti/unlearning/.hf_home
export HF_DATASETS_CACHE=/data/home/vkropoti/unlearning/.hf_datasets_cache
export TRITON_CACHE_DIR=/data/home/vkropoti/unlearning/.triton
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

git status --short
nvidia-smi
python -V
```

Confirm:

1. correct branch / working tree
2. expected GPU is visible
3. offline cache environment is the intended one
4. CUDA environment is the one used for the repo

## 8. Build DUET artifacts offline

Build three artifacts for fair DUET comparison:

1. `duet_rare_dualcf.jsonl`
2. `duet_popular_dualcf.jsonl`
3. `duet_merged_dualcf.jsonl`

Keep the artifact-generation model fixed across all DualCF ablations so the
artifact itself is not another moving part.

### 8.1 Rare-only DUET artifact

```bash
# 1) counterfactuals
python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/DUET \
  --split 'city_forget_rare_5' \
  --output-path /tmp/duet_rare_step1.jsonl \
  --question-key question \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct

# 2) difficulty
python src/tools/score_difficulty.py \
  --dataset-path json \
  --split train \
  --data-files /tmp/duet_rare_step1.jsonl \
  --output-path /tmp/duet_rare_step2.jsonl \
  --question-key question \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --popularity-column pop_sum \
  --w-pop 1.0 \
  --w-conf 1.0

# 3) attribution
python src/tools/score_attribution.py \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --forget-dataset-path json \
  --forget-split train \
  --forget-data-files /tmp/duet_rare_step2.jsonl \
  --retain-dataset-path SwetieePawsss/DUET \
  --retain-split city_fast_retain_500 \
  --output-path /tmp/duet_rare_dualcf.jsonl \
  --question-key question \
  --retain-max-steps 64
```

### 8.2 Popular-only DUET artifact

Repeat the same three stages with:

- source split `city_forget_popular_5`
- outputs `/tmp/duet_popular_step1.jsonl`,
  `/tmp/duet_popular_step2.jsonl`,
  `/tmp/duet_popular_dualcf.jsonl`

### 8.3 Merged DUET artifact

Repeat the same three stages with:

- source split `city_forget_rare_5+city_forget_popular_5`
- outputs `/tmp/duet_merged_step1.jsonl`,
  `/tmp/duet_merged_step2.jsonl`,
  `/tmp/duet_merged_dualcf.jsonl`

Use the merged artifact only for merged comparisons.

## 9. DUET artifact validation

### 9.1 Quick preview

```bash
head -n 2 /tmp/duet_rare_dualcf.jsonl
head -n 2 /tmp/duet_popular_dualcf.jsonl
head -n 2 /tmp/duet_merged_dualcf.jsonl
```

### 9.2 Full-file integrity scan

```bash
python - <<'PY'
import json
import math
from pathlib import Path

paths = [
    Path("/tmp/duet_rare_dualcf.jsonl"),
    Path("/tmp/duet_popular_dualcf.jsonl"),
    Path("/tmp/duet_merged_dualcf.jsonl"),
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

            for key in ("question", "answer", "alternate"):
                value = row[key]
                if not isinstance(value, str) or not value.strip():
                    bad_rows.append((line_no, f"empty or non-string {key}"))

            for score_key in ("difficulty_score", "attribution_score"):
                value = row[score_key]
                if not isinstance(value, (int, float)):
                    bad_rows.append((line_no, f"{score_key} is not numeric: {value!r}"))
                elif not math.isfinite(value):
                    bad_rows.append((line_no, f"{score_key} is not finite: {value!r}"))

            if (
                isinstance(row["answer"], str)
                and isinstance(row["alternate"], str)
                and row["answer"].strip().lower() == row["alternate"].strip().lower()
            ):
                bad_rows.append((line_no, "alternate matches answer after normalization"))

    print(path)
    print("  rows=", len(seen_indices))
    print("  duplicates=", sorted(duplicate_indices))
    print("  bad_rows=", bad_rows[:10])
PY
```

Expected result:

1. `duplicates=[]` for each file
2. `bad_rows=[]` for each file

### 9.3 Split-match and provenance checks

Before any DUET training:

1. rare-only artifact must contain only rare-target forget rows
2. popular-only artifact must contain only popular-target forget rows
3. merged artifact must be used only for merged comparison
4. sidecar provenance file must exist for each final artifact
5. artifact row count must match the intended training slice or full split
6. log difficulty min/max and attribution min/max for each artifact
7. fail if difficulty range looks surprising and is undocumented

## 10. DUET direct 1-step smoke test

This is the first real train test. Use the rare-only artifact first.

Important:

1. `cf_dataset_split='train[:2]'` controls the actual 2-sample train slice
2. `forget_split=city_forget_rare_5` keeps eval identity aligned with the
   benchmark target

```bash
DUET_DUALCF_JSONL=/tmp/duet_rare_dualcf.jsonl

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=duet_dualcf_smoke \
  forget_split=city_forget_rare_5 \
  retain_split=city_fast_retain_500 \
  cf_dataset_path=json \
  cf_dataset_data_files="${DUET_DUALCF_JSONL}" \
  "cf_dataset_split=train[:2]" \
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

## 11. DUET launcher smoke test

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
FORGET_SPLIT_OVERRIDE=city_forget_rare_5 \
RETAIN_SPLIT_OVERRIDE=city_fast_retain_500 \
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

## 12. DUET small functional run

Only after the 1-step smoke passes, do a short DUET run with slightly larger
slices.

Important:

1. keep `forget_split=city_forget_rare_5`
2. shrink actual training via `cf_dataset_split='train[:32]'`
3. evaluate on the benchmark split you want to report

### Train

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/duet/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=duet_dualcf_small \
  forget_split=city_forget_rare_5 \
  retain_split=city_fast_retain_500 \
  cf_dataset_path=json \
  cf_dataset_data_files="${DUET_DUALCF_JSONL}" \
  "cf_dataset_split=train[:32]" \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.num_train_epochs=1 \
  trainer.args.learning_rate=1e-5 \
  paths.output_dir=/tmp/duet_dualcf_small
```

### Eval

```bash
python src/eval.py \
  experiment=eval/duet/default.yaml \
  model=Llama-3.2-1B-Instruct-lora \
  forget_split=city_forget_rare_5[:32] \
  holdout_split=city_fast_retain_500[:64] \
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

## 13. DUET ablation and baseline matrix

Run only after the small functional run works.

Keep the artifact fixed across:

1. full DualCF
2. difficulty-only
3. attribution-only
4. uniform counterfactual baseline

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

### Negative-only baselines

Also run:

1. `GA`
2. `NPO`

Check for all ablations and baselines:

1. logs are present
2. training completes
3. no silent metadata drop
4. benchmark split and artifact split match the intended comparison

## 14. First real DUET 8B runs

Only after all 1B DUET smoke tests pass:

1. switch to the intended 8B model
2. keep batch size conservative
3. use the same base family for DualCF, GA, and NPO
4. keep LoRA rank / alpha / dropout aligned across methods
5. keep retain split aligned across methods
6. use the same eval config and seed list where possible

Recommended order:

1. `city_forget_rare_5`
2. `city_forget_popular_5`
3. merged rare+popular

## 15. DUET comparison outputs to save

For DUET, compare at least:

1. DualCF full
2. DualCF difficulty-only
3. DualCF attribution-only
4. DPO uniform counterfactual
5. GA
6. NPO

Compare:

1. `forget_qa_rouge`
2. `holdout_qa_rouge`
3. rare vs popular vs merged behavior
4. training stability notes from `dualcf_*` diagnostics

## 16. Build RWKU artifact offline

Move here only after DUET is stable.

RWKU defaults:

1. `question_key=query`
2. `forget_split=forget_level2`
3. `retain_split=neighbor_level2`

For the first plumbing-only RWKU artifact:

```bash
# 1) counterfactuals
python src/tools/make_counterfactuals.py \
  --dataset-path SwetieePawsss/exp_r \
  --dataset-name forget_level2 \
  --split test \
  --output-path /tmp/rwku_step1.jsonl \
  --question-key query \
  --answer-key answer \
  --model-cfg configs/model/Llama-3.2-1B-Instruct.yaml \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct
```

Then:

1. run `score_difficulty.py` only with documented proxies that actually exist in
   RWKU
2. do not blindly reuse DUET popularity weighting if `pop_sum` is absent
3. if needed, start with confidence-only difficulty scoring
4. write an intermediate difficulty artifact such as `/tmp/rwku_step2.jsonl`
5. run `score_attribution.py` with a documented proxy retain bank and write the
   final artifact to `/tmp/rwku_dualcf.jsonl`

For a plumbing smoke, `neighbor_level2` is acceptable as the proxy retain bank.
For the main RWKU result, prefer a separate proxy retain bank or disclose clearly
that the benchmark retain split was reused as the proxy retain source.

## 17. RWKU artifact validation

Before any RWKU training:

1. run the same schema and finite-score checks as DUET
2. assert `question_key=query`
3. assert the artifact corresponds exactly to `forget_level2`
4. record the proxy retain bank choice in provenance sidecar metadata
5. document whether benchmark eval retain data was reused as the attribution
   proxy bank

## 18. RWKU direct 1-step smoke test

Use a 1B LoRA override for plumbing only.

Important:

1. `cf_dataset_split='train[:2]'` controls the real train slice in local JSON mode
2. `forget_split=forget_level2` and `retain_split=neighbor_level2` keep benchmark
   identity aligned

```bash
RWKU_DUALCF_JSONL=/tmp/rwku_dualcf.jsonl

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/rwku/dual_cf_lora.yaml \
  trainer=DualCF \
  model=Llama-3.2-1B-Instruct-lora \
  task_name=rwku_dualcf_smoke \
  forget_split=forget_level2 \
  retain_split=neighbor_level2 \
  cf_dataset_path=json \
  cf_dataset_data_files="${RWKU_DUALCF_JSONL}" \
  cf_dataset_name=null \
  "cf_dataset_split=train[:2]" \
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
3. no accidental use of the benchmark retain split as if it were the true retain
   set without disclosure

## 19. First real RWKU 8B runs

Only after the RWKU smoke passes:

1. switch back to the intended 8B base
2. keep the same LoRA family for DualCF, GA, and NPO
3. start with a small identical LR shortlist
4. expand only after the first clean comparison run

Minimum RWKU comparison set:

1. DualCF full
2. GA
3. NPO

Repo note:

1. RWKU eval still writes `DUET_EVAL.json` and `DUET_SUMMARY.json`
2. that filename reuse is expected in this repo

## 20. Optional extra forget-metric pass for RWKU

After the first real RWKU runs, optionally run the extra forget-metric script:

```bash
FORGET_SPLIT=forget_level2 \
RETAIN_SPLIT=neighbor_level2 \
GPU_ID=1 \
BATCH_SIZE=16 \
AMP_MODE=bf16 \
bash scripts/forget_metrics/run_forget_metrics_rwku.sh
```

## 21. Sweep gating rules

Do not start larger sweeps until all of these are true:

1. DUET rare-only artifact was built
2. DUET rare-only artifact passed validation
3. DUET 1-step smoke passed
4. DUET launcher smoke passed
5. DUET small functional run passed
6. DUET eval completed
7. DUET ablation matrix is defined and split-matched
8. RWKU artifact passed validation
9. RWKU 1-step smoke passed
10. artifact provenance sidecars were saved
11. routed logs appeared in all DualCF runs

## 22. Failure triage order

If a smoke run fails, debug in this order:

1. `cf_dataset_split` is wrong for local JSON mode
2. `cf_dataset_data_files` points to the wrong file
3. artifact split does not match the intended benchmark target
4. artifact is missing `alternate`
5. artifact is missing `difficulty_score`
6. artifact is missing `attribution_score`
7. artifact is missing `index`
8. dataset / collator dropped metadata
9. batch-size mismatch reached `_score_tensor`
10. `alternate == answer` for many rows
11. difficulty or attribution range is broken
12. generated alternates are low quality or not truly counterfactual

## 23. What to save after each test

For every smoke or real run, keep:

1. exact command used
2. stdout / stderr log
3. output directory
4. checkpoint path
5. summary / eval JSON
6. note whether `dualcf_*` logs appeared
7. artifact path used
8. artifact provenance sidecar path
9. note whether the run was rare-only, popular-only, merged, or RWKU
