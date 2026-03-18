# Codex Smoke Test Log

Repo:

```bash
cd /Users/valerii.kropotin/НОД/Diploma/open-unlearning
```

## Requested smoke configuration

- `MAX_EXAMPLES=40`
- `NUM_ALTERNATES=2` per source model run
- `CODEX_CONCURRENT=4`
- source models:
  - `gpt-5.4-mini` with `medium`
  - `gpt-5.4-mini` with `high`
  - `gpt-5.4` with `medium`
  - `gpt-5.3-codex` with `medium`
- merged sidecars:
  - rare: `40` rows, `8` alternates per row
  - popular: `40` rows, `8` alternates per row
  - RWKU: `40` rows, `8` alternates per row
  - merged DUET from rare+popular parts: `80` rows, `8` alternates per row

## Untimed preflight commands

These were run without `/usr/bin/time -p`.

```bash
ps -axo pid,ppid,stat,etime,command | rg 'generate_codex_cf_sidecar|codex exec|run_duet_phase_a_codex|run_rwku_phase_a_codex'
```

```bash
codex login status
```

```bash
python - <<'PY'
import subprocess, tempfile
models=['gpt-5.4-mini','gpt-5.4','gpt-5.3-codex']
for model in models:
    tmp=tempfile.mkdtemp()
    cmd=['codex','-s','read-only','-a','never','exec','-C',tmp,'--skip-git-repo-check','-m',model,'-c','model_reasoning_effort="medium"','--color','never','Reply with OK only']
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    combined=((p.stdout or '')+'\n'+(p.stderr or '')).strip().splitlines()
    tail=combined[-1] if combined else ''
    print(f'{model}\t{p.returncode}\t{tail[:180]}')
PY
```

Result:

- `gpt-5.4-mini` accepted
- `gpt-5.4` accepted
- `gpt-5.3-codex` accepted

## Timed commands

### 1. DUET sidecar-only: `gpt-5.4-mini / medium`

```bash
/usr/bin/time -p env CODEX_MODEL=gpt-5.4-mini CODEX_REASONING_EFFORT=medium CODEX_CONCURRENT=4 MAX_EXAMPLES=40 NUM_ALTERNATES=2 STOP_AFTER_SIDECAR=1 "DUET_TARGETS=rare popular" RUN_TAG=smoke40_g54mini_med CODEX_TIMEOUT_SECONDS=900 bash scripts/api_cf/run_duet_phase_a_codex.sh
```

- exit: `0`
- real: `94.14s`

### 2. RWKU sidecar-only: `gpt-5.4-mini / medium`

```bash
/usr/bin/time -p env CODEX_MODEL=gpt-5.4-mini CODEX_REASONING_EFFORT=medium CODEX_CONCURRENT=4 MAX_EXAMPLES=40 NUM_ALTERNATES=2 STOP_AFTER_SIDECAR=1 RUN_TAG=smoke40_g54mini_med CODEX_TIMEOUT_SECONDS=900 bash scripts/api_cf/run_rwku_phase_a_codex.sh
```

- exit: `0`
- real: `20.47s`

### 3. DUET sidecar-only: `gpt-5.4-mini / high`

```bash
/usr/bin/time -p env CODEX_MODEL=gpt-5.4-mini CODEX_REASONING_EFFORT=high CODEX_CONCURRENT=4 MAX_EXAMPLES=40 NUM_ALTERNATES=2 STOP_AFTER_SIDECAR=1 "DUET_TARGETS=rare popular" RUN_TAG=smoke40_g54mini_high CODEX_TIMEOUT_SECONDS=900 bash scripts/api_cf/run_duet_phase_a_codex.sh
```

- exit: `0`
- real: `119.30s`

### 4. RWKU sidecar-only: `gpt-5.4-mini / high`

```bash
/usr/bin/time -p env CODEX_MODEL=gpt-5.4-mini CODEX_REASONING_EFFORT=high CODEX_CONCURRENT=4 MAX_EXAMPLES=40 NUM_ALTERNATES=2 STOP_AFTER_SIDECAR=1 RUN_TAG=smoke40_g54mini_high CODEX_TIMEOUT_SECONDS=900 bash scripts/api_cf/run_rwku_phase_a_codex.sh
```

- exit: `0`
- real: `23.00s`

### 5. DUET sidecar-only: `gpt-5.4 / medium`

```bash
/usr/bin/time -p env CODEX_MODEL=gpt-5.4 CODEX_REASONING_EFFORT=medium CODEX_CONCURRENT=4 MAX_EXAMPLES=40 NUM_ALTERNATES=2 STOP_AFTER_SIDECAR=1 "DUET_TARGETS=rare popular" RUN_TAG=smoke40_g54_med CODEX_TIMEOUT_SECONDS=900 bash scripts/api_cf/run_duet_phase_a_codex.sh
```

- exit: `0`
- real: `138.83s`

### 6. RWKU sidecar-only: `gpt-5.4 / medium`

```bash
/usr/bin/time -p env CODEX_MODEL=gpt-5.4 CODEX_REASONING_EFFORT=medium CODEX_CONCURRENT=4 MAX_EXAMPLES=40 NUM_ALTERNATES=2 STOP_AFTER_SIDECAR=1 RUN_TAG=smoke40_g54_med CODEX_TIMEOUT_SECONDS=900 bash scripts/api_cf/run_rwku_phase_a_codex.sh
```

- exit: `0`
- real: `49.72s`

### 7. DUET sidecar-only: `gpt-5.3-codex / medium`

```bash
/usr/bin/time -p env CODEX_MODEL=gpt-5.3-codex CODEX_REASONING_EFFORT=medium CODEX_CONCURRENT=4 MAX_EXAMPLES=40 NUM_ALTERNATES=2 STOP_AFTER_SIDECAR=1 "DUET_TARGETS=rare popular" RUN_TAG=smoke40_g53codex_med CODEX_TIMEOUT_SECONDS=900 bash scripts/api_cf/run_duet_phase_a_codex.sh
```

- exit: `0`
- real: `114.86s`

### 8. RWKU sidecar-only: `gpt-5.3-codex / medium`

```bash
/usr/bin/time -p env CODEX_MODEL=gpt-5.3-codex CODEX_REASONING_EFFORT=medium CODEX_CONCURRENT=4 MAX_EXAMPLES=40 NUM_ALTERNATES=2 STOP_AFTER_SIDECAR=1 RUN_TAG=smoke40_g53codex_med CODEX_TIMEOUT_SECONDS=900 bash scripts/api_cf/run_rwku_phase_a_codex.sh
```

- exit: `0`
- real: `43.68s`

### 9. Merge rare sidecars

```bash
/usr/bin/time -p python scripts/api_cf/merge_codex_sidecars.py --input-dir artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g54mini_med --input-dir artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g54mini_high --input-dir artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g54_med --input-dir artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_g53codex_med --output-path artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_mix4/api_sidecar.jsonl --max-alternates 8
```

- exit: `0`
- real: `0.04s`

### 10. Merge popular sidecars

```bash
/usr/bin/time -p python scripts/api_cf/merge_codex_sidecars.py --input-dir artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_g54mini_med --input-dir artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_g54mini_high --input-dir artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_g54_med --input-dir artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_g53codex_med --output-path artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_mix4/api_sidecar.jsonl --max-alternates 8
```

- exit: `0`
- real: `0.04s`

### 11. Merge RWKU sidecars

```bash
/usr/bin/time -p python scripts/api_cf/merge_codex_sidecars.py --input-dir artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_g54mini_med --input-dir artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_g54mini_high --input-dir artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_g54_med --input-dir artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_g53codex_med --output-path artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_mix4/api_sidecar.jsonl --max-alternates 8
```

- exit: `0`
- real: `0.03s`

### 12. Mixed DUET rare Phase A reuse, first failed attempt

This exposed an empty-array shell bug in `scripts/duet/prepare_dual_cf_duet_v3.sh`.

```bash
/usr/bin/time -p env MAX_EXAMPLES=40 NUM_ALTERNATES=8 SKIP_SIDECAR_GENERATION=1 STOP_AFTER_SIDECAR=0 RUN_TAG=smoke40_mix4 "DUET_TARGETS=rare" bash scripts/api_cf/run_duet_phase_a_codex.sh
```

- exit: `1`
- real: `17.86s`
- failure: `dataset_extra_args[@]: unbound variable`

### 13. Mixed DUET rare Phase A reuse, second failed attempt

There was a second `dataset_extra_args[@]` expansion later in the same DUET prep script.

```bash
/usr/bin/time -p env MAX_EXAMPLES=40 NUM_ALTERNATES=8 SKIP_SIDECAR_GENERATION=1 STOP_AFTER_SIDECAR=0 RUN_TAG=smoke40_mix4 "DUET_TARGETS=rare" bash scripts/api_cf/run_duet_phase_a_codex.sh
```

- exit: `1`
- real: `17.90s`
- failure: `dataset_extra_args[@]: unbound variable`

### 14. Mixed DUET rare Phase A reuse, successful retry

```bash
/usr/bin/time -p env MAX_EXAMPLES=40 NUM_ALTERNATES=8 SKIP_SIDECAR_GENERATION=1 STOP_AFTER_SIDECAR=0 RUN_TAG=smoke40_mix4 "DUET_TARGETS=rare" bash scripts/api_cf/run_duet_phase_a_codex.sh
```

- exit: `0`
- real: `68.98s`

### 15. Mixed DUET popular Phase A reuse

```bash
/usr/bin/time -p env MAX_EXAMPLES=40 NUM_ALTERNATES=8 SKIP_SIDECAR_GENERATION=1 STOP_AFTER_SIDECAR=0 RUN_TAG=smoke40_mix4 "DUET_TARGETS=popular" bash scripts/api_cf/run_duet_phase_a_codex.sh
```

- exit: `0`
- real: `61.08s`

### 16. Mixed RWKU Phase A reuse

```bash
/usr/bin/time -p env MAX_EXAMPLES=40 NUM_ALTERNATES=8 SKIP_SIDECAR_GENERATION=1 STOP_AFTER_SIDECAR=0 RUN_TAG=smoke40_mix4 bash scripts/api_cf/run_rwku_phase_a_codex.sh
```

- exit: `0`
- real: `13.61s`

### 17. Rebuild merged DUET from mixed rare + popular

```bash
/usr/bin/time -p env RARE_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_mix4 POPULAR_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_mix4 OUT_DIR=/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3__smoke40_mix4 bash scripts/api_cf/run_duet_phase_a_codex_merged_from_parts.sh
```

- exit: `0`
- real: `38.71s`

## Patch applied during the smoke

The smoke required one repo fix before the mixed DUET Phase A reuse path could succeed:

- [prepare_dual_cf_duet_v3.sh](/Users/valerii.kropotin/НОД/Diploma/open-unlearning/scripts/duet/prepare_dual_cf_duet_v3.sh)

What changed:

- guarded the optional `dataset_extra_args` append in `make_cf_args`
- replaced a direct `build_duet_candidate_bank.py ... "${dataset_extra_args[@]}"` call with a guarded `candidate_bank_args` array

This was needed only because `set -u` treats empty optional arrays as unbound in this environment.

## Final artifact summary

### Source model sidecars

All four source-model rare sidecars are structurally clean:

- `rare_codex_v3__smoke40_g54mini_med`: `rows=40`, `unique_indices=40`, `bad_rows=0`
- `rare_codex_v3__smoke40_g54mini_high`: `rows=40`, `unique_indices=40`, `bad_rows=0`
- `rare_codex_v3__smoke40_g54_med`: `rows=40`, `unique_indices=40`, `bad_rows=0`
- `rare_codex_v3__smoke40_g53codex_med`: `rows=40`, `unique_indices=40`, `bad_rows=0`

The same pattern holds for the matching popular and RWKU source sidecars.

### Mixed artifacts

- [rare mixed](/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/rare_codex_v3__smoke40_mix4)
  - `api_sidecar.jsonl.summary.json`: `rows=40`, `bad_rows=0`
  - `step1b_clean_report.json`: `rows=40`, `valid_row_rate=1.0`, `exact_match_count=0`, `gold_substring_count=0`, `invalid_reason_counts={}`
- [popular mixed](/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/popular_codex_v3__smoke40_mix4)
  - `api_sidecar.jsonl.summary.json`: `rows=40`, `bad_rows=0`
  - `step1b_clean_report.json`: `rows=40`, `valid_row_rate=1.0`, `exact_match_count=0`, `gold_substring_count=0`, `invalid_reason_counts={}`
- [rwku mixed](/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/rwku/forget_level2_codex_v3__smoke40_mix4)
  - `api_sidecar.jsonl.summary.json`: `rows=40`, `bad_rows=0`
  - `step1b_clean_report.json`: `rows=40`, `valid_row_rate=1.0`, `exact_match_count=0`, `gold_substring_count=0`, `invalid_reason_counts={}`
- [merged mixed](/Users/valerii.kropotin/НОД/Diploma/open-unlearning/artifacts/dualcf_api_v3/duet/merged_codex_v3__smoke40_mix4)
  - `api_sidecar.jsonl.summary.json`: `rows=80`, `bad_rows=0`
  - `step1b_clean_report.json`: `rows=80`, `valid_row_rate=1.0`, `exact_match_count=0`, `gold_substring_count=0`, `invalid_reason_counts={}`

## Important note about merged row count

The merged DUET artifact is `80` rows, not `40`, because it was built from the
union of:

- `40` rare rows
- `40` popular rows

via `run_duet_phase_a_codex_merged_from_parts.sh`.
