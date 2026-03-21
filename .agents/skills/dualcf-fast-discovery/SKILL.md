---
name: dualcf-fast-discovery
description: Use first when you need fast discovery of docs, code paths, launchers, configs, and recent repo changes before implementation. Optimize for exact file triage and command discovery.
---

# DualCF Fast Discovery

## When to use
- A task starts with ambiguity about where the logic lives.
- You need the fastest path to the right docs, scripts, configs, or recent commits.
- You want to identify the smallest safe read set before touching code.
- You need a short implementation plan grounded in the current repo state.

## Read first
- `AGENTS.md`
- `README.md`
- `dual_cf_v3_integration_diff.md`
- `prod-run-dual-gpu-v3.md`
- the exact file family the task mentions

## Workflow
1. Find the exact code path with `rg`, `git log`, `git show`, and focused file reads.
2. Separate docs, runtime scripts, trainer code, and artifact tools.
3. Identify hard invariants that apply to the task.
4. Produce the minimum file shortlist required for safe implementation.
5. If the task continues into coding, hand off to `dualcf-portable-runtime`.

## Guardrails
- Keep context small and precise.
- Prefer exact file references over broad summaries.
- Surface path portability risks, eval-flow risks, and split-matching risks early.
- Do not broaden the task into an unsolicited refactor.

## Output standard
Always return:
- exact files to read or edit next,
- exact commands to inspect or smoke-test,
- relevant recent commits or diffs,
- the main risks to preserve during implementation.
