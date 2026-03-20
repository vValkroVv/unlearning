# Codex Counterfactual Generation Full Runbook On macOS

This runbook is Codex-first with verified ChatGPT Pro imports.
The filename is historical, but the commands below use the dedicated Codex generator plus canonical imported ChatGPT Pro sidecars, matching the exact multi-model workflow validated in `smoke_test_codex.md`.

It shows the full-run path to:

- generate DUET sidecars from the same three validated Codex source-model configurations
- add the verified ChatGPT Pro DUET and RWKU sidecars generated outside Codex CLI
- generate RWKU sidecars from the same three validated source-model configurations
- merge DUET and RWKU to `8` alternates per row
- run Phase A from the merged sidecars without regenerating them
- rebuild merged DUET from the merged rare + popular parts

Current repo path on this machine:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning
```

## 1. Validated source-model matrix

The smoke accepted these exact Codex model / reasoning pairs that this runbook still uses:

- `gpt-5.4-mini` with `medium`
- `gpt-5.4-mini` with `high`
- `gpt-5.4` with `low`

Additional verified ChatGPT Pro sources added outside Codex CLI:

- `gpt-5.4-pro` via ChatGPT Pro chat, normalized as `full_g54pro_xhigh`
- canonical rare import:
  `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_g54pro_xhigh`
- canonical popular import:
  `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_g54pro_xhigh`
- `gpt-5.4-pro` via ChatGPT Pro chat for RWKU, normalized as `full_g54pro_high`
- canonical RWKU import:
  `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_g54pro_high`

Each source-model run generates:

- `NUM_ALTERNATES=2` alternates per row
- sidecar only first
- separate tagged output directories

Then the three Codex sidecars plus the imported ChatGPT Pro sidecars are merged to:

- `8` alternates per row for DUET rare
- `8` alternates per row for DUET popular
- `8` alternates per row for DUET merged from parts
- `8` alternates per row for RWKU

## 2. Common setup

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

unset CODEX_API_KEY
codex login status

export CODEX_CONCURRENT=4
export CODEX_TIMEOUT_SECONDS=900
export CODEX_BATCH_SIZE=10
export CODEX_MAX_ATTEMPTS=3

export MAX_EXAMPLES=0
export NUM_ALTERNATES=2
export STOP_AFTER_SIDECAR=1
export SKIP_SIDECAR_GENERATION=0
```

`MAX_EXAMPLES=0` means full dataset, not smoke.

If auth is stale:

```bash
codex logout
codex login
```

Optional preflight model check:

```bash
python - <<'PY'
import subprocess, tempfile
pairs = [
    ("gpt-5.4-mini", "medium"),
    ("gpt-5.4-mini", "high"),
    ("gpt-5.4", "low"),
]
for model, effort in pairs:
    tmp = tempfile.mkdtemp()
    cmd = [
        "codex", "-s", "read-only", "-a", "never", "exec",
        "-C", tmp,
        "--skip-git-repo-check",
        "-m", model,
        "-c", f'model_reasoning_effort="{effort}"',
        "--color", "never",
        "Reply with OK only",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    combined = ((p.stdout or "") + "\n" + (p.stderr or "")).strip().splitlines()
    tail = combined[-1] if combined else ""
    print(f"{model}\t{effort}\t{p.returncode}\t{tail[:180]}")
PY
```

## 3. Generate sidecar-only outputs for the three validated Codex source runs

This is the smoke-tested approach, but with full-dataset settings and stable `RUN_TAG` names.

The naming convention is:

- `full_g54mini_med`
- `full_g54mini_high`
- `full_g54_low`

Run all three DUET + RWKU source passes:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning

configs=(
  "gpt-5.4-mini medium full_g54mini_med"
  "gpt-5.4-mini high full_g54mini_high"
  "gpt-5.4 low full_g54_low"
)

for cfg in "${configs[@]}"; do
  read -r model effort tag <<<"$cfg"

  env \
    CODEX_MODEL="$model" \
    CODEX_REASONING_EFFORT="$effort" \
    CODEX_CONCURRENT=4 \
    CODEX_TIMEOUT_SECONDS=900 \
    MAX_EXAMPLES=0 \
    NUM_ALTERNATES=2 \
    STOP_AFTER_SIDECAR=1 \
    DUET_TARGETS="rare popular" \
    RUN_TAG="$tag" \
    bash scripts/api_cf/run_duet_phase_a_codex.sh

  env \
    CODEX_MODEL="$model" \
    CODEX_REASONING_EFFORT="$effort" \
    CODEX_CONCURRENT=4 \
    CODEX_TIMEOUT_SECONDS=900 \
    MAX_EXAMPLES=0 \
    NUM_ALTERNATES=2 \
    STOP_AFTER_SIDECAR=1 \
    RUN_TAG="$tag" \
    bash scripts/api_cf/run_rwku_phase_a_codex.sh
done
```

This writes per-model source directories like:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_g54mini_med`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_g54mini_med`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_g54mini_med`

and the matching `__full_g54mini_high` and `__full_g54_low` directories.

Do not run DUET `merged` in this stage.
The supported path is rare + popular first, then merged from parts.

## 4. Verified ChatGPT Pro imports

### 4.1 DUET import

The extra DUET source came from ChatGPT Pro chat rather than the Codex CLI wrappers.
The verified source directories on this machine were:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts-gpt-pro-rare/dualcf_api_v3/duet/rare_codex_v3`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts-gpt-pro-popular/dualcf_api_v3/duet/popular_codex_v3`

The canonical copies were checked with:

```bash
python scripts/api_cf/check_phase_a_outputs.py \
  --dataset duet \
  --question-key question \
  --out-dir artifacts/dualcf_api_v3/duet/rare_codex_v3__full_g54pro_xhigh

python scripts/api_cf/check_phase_a_outputs.py \
  --dataset duet \
  --question-key question \
  --out-dir artifacts/dualcf_api_v3/duet/popular_codex_v3__full_g54pro_xhigh
```

The canonical imported directories used below are:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_g54pro_xhigh`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_g54pro_xhigh`

Their copied metadata keeps the actual chat backend (`chatgpt_manual` for rare,
`chatgpt_assistant` for popular) while the canonical tag records the intended
source label `g54pro_xhigh`.

### 4.2 RWKU import

The extra RWKU source also came from ChatGPT Pro chat rather than the Codex CLI wrappers.
The verified source directory on this machine was:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts-gpt-pro-rwku/dualcf_api_v3/rwku/forget_level2_codex_v3`

The canonical copy was checked with:

```bash
python scripts/api_cf/check_phase_a_outputs.py \
  --dataset rwku \
  --question-key query \
  --out-dir artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_g54pro_high
```

The canonical imported directory used below is:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_g54pro_high`

This RWKU copy has no `step0_candidate_bank.jsonl`, which is expected for RWKU.
Its copied metadata keeps the actual chat backend (`chatgpt_manual`) while the
canonical tag records the imported source label `g54pro_high`. The supplied RWKU
Pro metadata recorded `reasoning_effort=high`, so the canonical tag stays
`high` rather than inventing `xhigh`.

## 5. Merge the source sidecars

Use the merge helper.
Do not raw-`cat` the sidecars.
The helper now accepts the imported ChatGPT Pro sidecars even when their copied
metadata preserves `backend=chatgpt_manual` / `chatgpt_assistant` or a
different canonical dataset path than the local Codex-generated runs.

### 5.1 Merge DUET rare to 8 alternates

```bash
python /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_g54mini_med \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_g54mini_high \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_g54_low \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_g54pro_xhigh \
  --output-path /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_mix4/api_sidecar.jsonl \
  --max-alternates 8
```

### 5.2 Merge DUET popular to 8 alternates

```bash
python /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_g54mini_med \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_g54mini_high \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_g54_low \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_g54pro_xhigh \
  --output-path /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_mix4/api_sidecar.jsonl \
  --max-alternates 8
```

### 5.3 Merge RWKU to 8 alternates

```bash
python /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/merge_codex_sidecars.py \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_g54mini_med \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_g54mini_high \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_g54_low \
  --input-dir /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_g54pro_high \
  --output-path /Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_mix4/api_sidecar.jsonl \
  --max-alternates 8
```

## 6. Run Phase A from the merged sidecars

At this point the mixed sidecars already exist in the final artifact directories.
Run Phase A prep without regenerating the sidecars.
The reuse path now accepts merged metadata with `merged_from_sidecars=true`,
`model=multiple`, and local dataset-path aliases recorded in
`input_dataset_paths`.

### 6.1 DUET rare from merged sidecar

```bash
env \
  MAX_EXAMPLES=0 \
  NUM_ALTERNATES=8 \
  SKIP_SIDECAR_GENERATION=1 \
  STOP_AFTER_SIDECAR=0 \
  RUN_TAG=full_mix4 \
  DUET_TARGETS="rare" \
  bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex.sh
```

### 6.2 DUET popular from merged sidecar

```bash
env \
  MAX_EXAMPLES=0 \
  NUM_ALTERNATES=8 \
  SKIP_SIDECAR_GENERATION=1 \
  STOP_AFTER_SIDECAR=0 \
  RUN_TAG=full_mix4 \
  DUET_TARGETS="popular" \
  bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex.sh
```

### 6.3 RWKU from merged sidecar

```bash
env \
  MAX_EXAMPLES=0 \
  NUM_ALTERNATES=8 \
  SKIP_SIDECAR_GENERATION=1 \
  STOP_AFTER_SIDECAR=0 \
  RUN_TAG=full_mix4 \
  bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_rwku_phase_a_codex.sh
```

## 7. Rebuild merged DUET from the merged rare + popular parts

This keeps merged DUET exactly aligned with the already-prepared rare and popular artifacts.

```bash
RARE_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_mix4 \
POPULAR_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_mix4 \
OUT_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3__full_mix4 \
NUM_ALTERNATES=8 \
bash /Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/api_cf/run_duet_phase_a_codex_merged_from_parts.sh
```

## 8. Expected full-run outputs

After success, these directories should exist:

- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__full_mix4`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__full_mix4`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3__full_mix4`
- `/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__full_mix4`

For DUET rare, DUET popular, DUET merged, and RWKU mixed outputs, the successful Phase A directories should contain:

- `api_sidecar.jsonl`
- `api_sidecar.jsonl.meta.json`
- `api_sidecar.jsonl.summary.json`
- `step1_counterfactuals_raw_v3.jsonl`
- `step1b_counterfactuals_clean_v3.jsonl`
- `step1b_clean_report.json`

DUET directories should also contain:

- `step0_candidate_bank.jsonl`

Merged DUET from parts should also contain:

- `merged_input.jsonl`

DUET mixed outputs now carry `8` alternates per row.
RWKU mixed outputs now also carry `8` alternates per row.

## 9. Notes

- This file now documents the full run, not the `MAX_EXAMPLES=40` smoke.
- The generator for DUET and RWKU is `scripts/api_cf/generate_codex_cf_sidecar.py`.
- The merge helper is `scripts/api_cf/merge_codex_sidecars.py`.
- The merged DUET reconstruction path is `scripts/api_cf/run_duet_phase_a_codex_merged_from_parts.sh`.
- The extra DUET `gpt-5.4-pro` sources were generated via ChatGPT Pro chat and then normalized into the canonical `__full_g54pro_xhigh` directories under `artifacts/dualcf_api_v3/duet/`.
- The extra RWKU `gpt-5.4-pro` source was generated via ChatGPT Pro chat and then normalized into the canonical `__full_g54pro_high` directory under `artifacts/dualcf_api_v3/rwku/`.
- Mixed `full_mix4` sidecars now preserve `input_backends`,
  `input_dataset_paths`, and `input_models` in the merged metadata so the
  provenance of imported ChatGPT Pro rows is kept even though the merged sidecar
  itself records `model=multiple`.
- `RUN_TAG` becomes the `__suffix` on the artifact directory name, so the tags above are chosen to stay short and readable.
- `CODEX_REASONING_EFFORT` may be `low|medium|high|xhigh`, but this runbook keeps only the combinations that passed the smoke.
- `codex-spark` is not part of this workflow. The smoke showed it is rejected under ChatGPT-login auth.
- The DUET Phase A reuse path now works under `set -u`; the smoke exposed and validated that fix.
- The merged-sidecar reuse path now also works for mixed Codex + ChatGPT Pro
  sources under `SKIP_SIDECAR_GENERATION=1`; the validator accepts
  `model=multiple` if the requested `CODEX_MODEL` is one of the recorded
  `input_models`.
- RWKU `rwku_shared_fact_safe` selection now adds a low-relation source
  penalty for `forget_semantic_nn`, `retain_semantic_nn`, and
  `same_subject_same_type`, then rescues sub-`0.7` picks to the best valid
  candidate with relation `>=0.85`.
- After rerunning only RWKU Phase A reuse on
  `forget_level2_codex_v3__full_mix4`, the mean selected relation score moved
  from `0.8805` to `0.9098`; selected relation `<0.7` rows fell from `198` to
  `22`, `<0.8` rows fell from `344` to `189`, and there were no remaining
  `<0.7` rows with another valid candidate at `>=0.85`.
- Do not raw-`cat` the sidecars, including the imported ChatGPT Pro files. Keep using `merge_codex_sidecars.py`.
- Do not compare merged DualCF artifacts against rare-only or popular-only baselines. Keep split matching intact.
