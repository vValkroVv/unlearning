#!/usr/bin/env bash
set -euo pipefail
repo_root="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
src="${repo_root}/.codex/AGENTS.root.md"
dst="${repo_root}/AGENTS.md"
if [[ ! -f "$src" ]]; then
  echo "Missing $src" >&2
  exit 1
fi
if [[ -f "$dst" ]]; then
  backup="${dst}.bak.$(date +%Y%m%d%H%M%S)"
  cp "$dst" "$backup"
  echo "Backed up existing AGENTS.md to $backup"
fi
cp "$src" "$dst"
echo "Installed $dst from $src"
