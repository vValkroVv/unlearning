# DualCF Codex starter kit

This bundle is a repo-local Codex harness for your DualCF workflow.

What is inside:
- `AGENTS.root.md`: the repo-root `AGENTS.md` you should install or symlink into the repository root.
- `install_root_agents.sh`: copies `AGENTS.root.md` to `../AGENTS.md`.
- `config.toml`: shared project-scoped Codex config.
- `agents/`: official custom subagent definitions in TOML.

Official layout used by Codex:
- `.codex/config.toml`
- `.codex/agents/<agent>.toml`
- `.agents/skills/<skill>/SKILL.md`
- `.agents/skills/<skill>/agents/openai.yaml` for optional UI metadata only

Model selection:
- put `model` and `model_reasoning_effort` in `.codex/agents/*.toml`
- optionally add named `[profiles.<name>]` in `.codex/config.toml`
- do not put model selection in `AGENTS.md`, skill markdown, or `agents/openai.yaml`

Recommended install:
1. Unzip this archive at the repository root.
2. Run `bash .codex/install_root_agents.sh`.
3. Restart Codex so the root `AGENTS.md` is picked up.
4. If your Codex build auto-discovers local skills, use them directly. If not, keep them as repo-local instructions and reference them from `AGENTS.md`.

Notes:
- `AGENTS.md` is the officially recognized persistent repo guidance file.
- Skills follow the `SKILL.md` plus optional `agents/openai.yaml` pattern.
- `agents/openai.yaml` is for UI metadata, invocation policy, and dependencies, not model routing.
