---
name: latex-pdf-build
description: Compile LaTeX sources into PDF files inside this open-unlearning workspace. Use when asked to build, rebuild, or verify a PDF from a `.tex` file under `/Users/valerii.kropotin/НОД/Diploma/open-unlearning`, especially generated slide decks such as `metrics-new/results-combine-v2_5/combined_tables_slides.tex`.
---

# LaTeX PDF Build

Compile the requested `.tex` file with the repo-local helper instead of rebuilding the workflow from scratch. Prefer `tectonic` because it handles package fetching and works well in this workspace.

## Workflow

1. Resolve the target `.tex` path relative to the workspace root `/Users/valerii.kropotin/НОД/Diploma/open-unlearning` unless the user gave an absolute path.
2. Run `scripts/build_pdf.py <path-to-tex>` from this skill.
3. Read the compiler output and report either the generated PDF path or the first actionable LaTeX error.
4. If the compiler fails because of a source issue, fix the document and rerun the same command.
5. If `tectonic` is missing, install it with Homebrew: `brew install tectonic`, then rerun the build.

## Quick Start

Use the helper directly:

```bash
python3 .agents/skills/latex-pdf-build/scripts/build_pdf.py metrics-new/results-combine-v2_5/combined_tables_slides.tex
```

## Rules

- Invoke the helper from the repo root or pass an absolute `.tex` path.
- Let the helper run in the source file's directory so LaTeX relative includes keep working.
- Do not assume an existing PDF is current; rebuild it when asked.
- Prefer fixing actual LaTeX/package errors over switching engines.
- Keep edits minimal when the build fails because of document issues.

## Resource

- `scripts/build_pdf.py`: Resolve repo-relative paths, verify `tectonic` exists, run the build in the correct directory, and print the resulting PDF path.
