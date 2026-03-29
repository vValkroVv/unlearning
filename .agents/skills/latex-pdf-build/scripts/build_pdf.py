#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_tex_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile a LaTeX source file in this workspace into a PDF."
    )
    parser.add_argument("tex_path", help="Path to the .tex file, absolute or relative to the repo root")
    args = parser.parse_args()

    engine = shutil.which("tectonic")
    if engine is None:
        print(
            "tectonic is not installed. Install it with: brew install tectonic",
            file=sys.stderr,
        )
        return 1

    tex_path = resolve_tex_path(args.tex_path)
    if not tex_path.exists():
        print(f"TeX file not found: {tex_path}", file=sys.stderr)
        return 1
    if tex_path.suffix.lower() != ".tex":
        print(f"Expected a .tex file, got: {tex_path}", file=sys.stderr)
        return 1

    cmd = [engine, tex_path.name]
    result = subprocess.run(cmd, cwd=tex_path.parent)
    if result.returncode != 0:
        return result.returncode

    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        print(f"Build finished but PDF was not created: {pdf_path}", file=sys.stderr)
        return 1

    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
