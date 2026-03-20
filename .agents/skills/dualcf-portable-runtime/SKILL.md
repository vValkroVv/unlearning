---
name: dualcf-portable-runtime
description: Use for robust code, config, and launcher work that must run on both local GPUs and VAST or H100 environments while preserving split matching, offline artifacts, and eval cadence.
---

# DualCF Portable Runtime

## When to use
- Editing training launchers, campaign wrappers, prep scripts, or eval scripts.
- Hardening path resolution, offline mirrors, cache handling, or env defaults.
- Fixing a bug that appears on VAST, H100, or a local GPU but not all environments.
- Making code changes that must remain robust under real runs, not just local inspection.

## Read first
- `AGENTS.md`
- `dual_cf_v3_integration_diff.md`
- `prod-run-dual-gpu.md`
- `prod-run-dual-vast.md`
- the exact launcher, config, and Python module being changed

## Workflow
1. Confirm the current runtime surface: env vars, paths, splits, artifacts, and model configs.
2. Patch the smallest file set that removes the real failure or portability hazard.
3. Keep local and remote paths overrideable through env vars.
4. Preserve train -> eval -> checkpoint eval -> utility eval -> cleanup flow unless the task explicitly changes it.
5. Run or specify small local smoke commands before any campaign-scale command.
6. Hand off to `dualcf-doc-sync` after the code path is stable.

## Guardrails
- Do not hard-code one machine layout if an env override is possible.
- Keep artifact preparation offline.
- Preserve split matching across DualCF, DPO, GA, NPO, NPO-SAM, and LoKU.
- Keep old configs working when possible.
- Fail fast with clear path, split, and artifact errors.

## Output standard
Always return:
- exact files changed,
- why the patch is safe,
- exact local smoke commands,
- exact VAST or remote smoke commands,
- expected logs, metrics, or behavior changes,
- any remaining risks.
