# Codex Counterfactual Generation On macOS

This runbook is now Codex-only.
The filename is historical, but the commands below use the dedicated Codex generator only.

It shows the exact local commands to:

- generate DUET Codex sidecars and Phase A artifacts
- generate RWKU Codex sidecars and Phase A artifacts
- rebuild merged DUET from existing `rare_codex_v3` + `popular_codex_v3`

Current repo path on this machine:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning
```

## 1. Common setup

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

unset CODEX_API_KEY
codex login status

export CODEX_MODEL="gpt-5.4-mini"
export CODEX_REASONING_EFFORT="medium"
export CODEX_CONCURRENT=1
export MAX_EXAMPLES=16
export NUM_ALTERNATES=4
export CODEX_TIMEOUT_SECONDS=900
```

Use `MAX_EXAMPLES=16` for a smoke run.

For a full run later:

```bash
export MAX_EXAMPLES=0
```

If a real `codex exec` run reports stale auth, refresh it once:

```bash
codex logout
codex login
```

## 2. DUET Codex generation

This runs the validated Codex Phase A wrapper for:

- `rare_codex_v3`
- `popular_codex_v3`
- `merged_codex_v3`

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

unset CODEX_API_KEY
export CODEX_MODEL="gpt-5.4-mini"
export CODEX_REASONING_EFFORT="medium"
export CODEX_CONCURRENT=1
export MAX_EXAMPLES=16
export NUM_ALTERNATES=4
export CODEX_TIMEOUT_SECONDS=900

bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex.sh
```

The wrapper calls:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/generate_codex_cf_sidecar.py`

That writes:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3`

## 3. RWKU Codex generation

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

unset CODEX_API_KEY
export CODEX_MODEL="gpt-5.4-mini"
export CODEX_REASONING_EFFORT="medium"
export CODEX_CONCURRENT=1
export MAX_EXAMPLES=16
export NUM_ALTERNATES=4
export CODEX_TIMEOUT_SECONDS=900

bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_rwku_phase_a_codex.sh
```

The wrapper calls:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/generate_codex_cf_sidecar.py`

That writes:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3`

## 4. Rebuild merged DUET from rare + popular Codex artifacts

If you do not want a separate merged Codex generation pass, first make sure these exist:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3`

Then rebuild merged from those two directories:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex_merged_from_parts.sh
```

You can override `RARE_DIR`, `POPULAR_DIR`, and `OUT_DIR` when the rare/popular
source directories are model-tagged or merged sidecars.

That writes:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/merged_input.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/api_sidecar.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/api_sidecar.jsonl.meta.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/api_sidecar.jsonl.summary.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/step0_candidate_bank.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/step1_counterfactuals_raw_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/step1b_counterfactuals_clean_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/step1b_clean_report.json`

## 5. Multi-model workflow: 8 counterfactuals from 4 models

The supported path is:

1. run sidecar-only generation separately for each model
2. generate `2` alternates per model
3. merge the resulting sidecars by `index`
4. rebuild merged DUET from the merged rare + merged popular sidecars

Do not raw-`cat` the files. Use the merge helper:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py`

### 5.1 Per-model sidecar-only runs

Use a distinct `RUN_TAG` for each model run so the output directories do not
overwrite each other.

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

unset CODEX_API_KEY
export MAX_EXAMPLES=16
export NUM_ALTERNATES=2
export STOP_AFTER_SIDECAR=1
export DUET_TARGETS="rare popular"
export CODEX_CONCURRENT=4
export CODEX_TIMEOUT_SECONDS=900
```

Example model run:

```bash
export CODEX_MODEL="gpt-5.4-mini"
export CODEX_REASONING_EFFORT="medium"
export RUN_TAG="m1_gpt54mini_medium"

bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex.sh
bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_rwku_phase_a_codex.sh
```

Repeat that block with three more model / reasoning combinations, for example:

```bash
export CODEX_MODEL="MODEL_2"
export CODEX_REASONING_EFFORT="medium"
export RUN_TAG="m2_model2_medium"
```

```bash
export CODEX_MODEL="MODEL_3"
export CODEX_REASONING_EFFORT="medium"
export RUN_TAG="m3_model3_medium"
```

```bash
export CODEX_MODEL="MODEL_4"
export CODEX_REASONING_EFFORT="medium"
export RUN_TAG="m4_model4_medium"
```

### 5.2 Merge rare sidecars to 8 total alternates

```bash
python /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__m1_gpt54mini_medium \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__m2_model2_medium \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__m3_model3_medium \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__m4_model4_medium \
  --output-path /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__mix/api_sidecar.jsonl \
  --max-alternates 8
```

### 5.3 Merge popular sidecars to 8 total alternates

```bash
python /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__m1_gpt54mini_medium \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__m2_model2_medium \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__m3_model3_medium \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__m4_model4_medium \
  --output-path /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__mix/api_sidecar.jsonl \
  --max-alternates 8
```

### 5.4 Merge RWKU sidecars to 8 total alternates

```bash
python /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__m1_gpt54mini_medium \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__m2_model2_medium \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__m3_model3_medium \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__m4_model4_medium \
  --output-path /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__mix/api_sidecar.jsonl \
  --max-alternates 8
```

### 5.5 Build merged DUET from the merged rare + popular sidecars

```bash
RARE_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__mix \
POPULAR_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__mix \
OUT_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3__mix \
bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex_merged_from_parts.sh
```

### 5.6 Reuse a merged sidecar for Phase A without regenerating

If you already wrote a merged or mixed sidecar into the final artifact
directory, you can skip sidecar generation and run only Phase A prep:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

export RUN_TAG="mix"
export SKIP_SIDECAR_GENERATION=1
export STOP_AFTER_SIDECAR=0
```

RWKU:

```bash
bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_rwku_phase_a_codex.sh
```

DUET rare only:

```bash
export DUET_TARGETS="rare"
bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex.sh
```

### 5.7 Exact smoke40 commands for the artifacts already on disk

If you already have these four source-model smoke directories:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g54mini_med`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g54mini_high`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g54_med`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g53codex_med`

and the matching `popular_codex_v3__smoke40_*` plus
`forget_level2_codex_v3__smoke40_*` directories, you do not need any new
generation. Use the commands below.

Merge rare to `8` alternates:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

python /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g54mini_med \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g54mini_high \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g54_med \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g53codex_med \
  --output-path /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_mix4/api_sidecar.jsonl \
  --max-alternates 8
```

Merge popular to `8` alternates:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

python /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_g54mini_med \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_g54mini_high \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_g54_med \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_g53codex_med \
  --output-path /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_mix4/api_sidecar.jsonl \
  --max-alternates 8
```

Merge RWKU to `8` alternates:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

python /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_g54mini_med \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_g54mini_high \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_g54_med \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_g53codex_med \
  --output-path /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_mix4/api_sidecar.jsonl \
  --max-alternates 8
```

Reuse the merged rare sidecar for Phase A:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

env MAX_EXAMPLES=40 NUM_ALTERNATES=8 SKIP_SIDECAR_GENERATION=1 STOP_AFTER_SIDECAR=0 RUN_TAG=smoke40_mix4 DUET_TARGETS="rare" \
  bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex.sh
```

Reuse the merged popular sidecar for Phase A:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

env MAX_EXAMPLES=40 NUM_ALTERNATES=8 SKIP_SIDECAR_GENERATION=1 STOP_AFTER_SIDECAR=0 RUN_TAG=smoke40_mix4 DUET_TARGETS="popular" \
  bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex.sh
```

Reuse the merged RWKU sidecar for Phase A:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

env MAX_EXAMPLES=40 NUM_ALTERNATES=8 SKIP_SIDECAR_GENERATION=1 STOP_AFTER_SIDECAR=0 RUN_TAG=smoke40_mix4 \
  bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_rwku_phase_a_codex.sh
```

Rebuild merged DUET from the smoke40 rare + popular mixed dirs:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

RARE_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_mix4 \
POPULAR_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_mix4 \
OUT_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3__smoke40_mix4 \
bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex_merged_from_parts.sh
```

These smoke40 outputs are already present on this machine and currently validate
cleanly:

- rare mixed: `40` rows
- popular mixed: `40` rows
- RWKU mixed: `40` rows
- merged DUET from parts: `80` rows

## 6. What should exist after success

DUET rare:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3/api_sidecar.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3/api_sidecar.jsonl.meta.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3/api_sidecar.jsonl.summary.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3/step0_candidate_bank.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3/step1_counterfactuals_raw_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3/step1b_counterfactuals_clean_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3/step1b_clean_report.json`

DUET popular:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3/api_sidecar.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3/api_sidecar.jsonl.meta.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3/api_sidecar.jsonl.summary.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3/step0_candidate_bank.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3/step1_counterfactuals_raw_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3/step1b_counterfactuals_clean_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3/step1b_clean_report.json`

DUET merged from parts:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/merged_input.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/api_sidecar.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/api_sidecar.jsonl.meta.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/api_sidecar.jsonl.summary.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/step0_candidate_bank.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/step1_counterfactuals_raw_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/step1b_counterfactuals_clean_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3/step1b_clean_report.json`

RWKU:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3/api_sidecar.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3/api_sidecar.jsonl.meta.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3/api_sidecar.jsonl.summary.json`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3/step1_counterfactuals_raw_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3/step1b_counterfactuals_clean_v3.jsonl`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3/step1b_clean_report.json`

## 7. Notes

- This file is Codex-only. No OpenAI API commands remain here.
- The active generator for DUET and RWKU is `scripts/api_cf/generate_codex_cf_sidecar.py`.
- You can pass `CODEX_REASONING_EFFORT=low|medium|high|xhigh` to the wrappers.
- You can pass `CODEX_CONCURRENT=N` to run up to `N` Codex batch requests in parallel.
- You can pass `RUN_TAG=...` to isolate outputs for different model runs.
- You can pass `STOP_AFTER_SIDECAR=1` to generate only `api_sidecar.jsonl` plus metadata.
- You can pass `SKIP_SIDECAR_GENERATION=1` to consume an already-merged sidecar without regenerating it.
- For partial smoke runs, merged-from-parts is the correct way to make merged equal rare + popular.
- Do not raw-`cat` sidecars. Use `merge_codex_sidecars.py` for model mixing and `run_duet_phase_a_codex_merged_from_parts.sh` for merged DUET reconstruction.
- `CODEX_TIMEOUT_SECONDS=900` is the safe local default for merged Codex runs on this machine.
- The generator shows a `tqdm` batch progress bar in terminal.
- Local live check on this machine: `codex-spark` is rejected under ChatGPT-login auth with `The 'codex-spark' model is not supported when using Codex with a ChatGPT account.`
