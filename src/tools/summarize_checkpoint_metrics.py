#!/usr/bin/env python3
"""Summarize checkpoint evaluation metrics for DualCF trajectory runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-name", default="DUET_SUMMARY.json")
    return parser.parse_args()


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**18, path.name


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    eval_dirs = []
    seen = set()

    checkpoint_eval_root = run_dir / "checkpoint_evals"
    if checkpoint_eval_root.exists():
        for candidate in checkpoint_eval_root.iterdir():
            summary_path = candidate / args.summary_name
            if summary_path.exists():
                eval_dirs.append((candidate, summary_path))
                seen.add(candidate.resolve())
    else:
        for candidate in run_dir.glob("checkpoint-*"):
            summary_path = candidate / "evals" / args.summary_name
            if summary_path.exists():
                eval_dirs.append((candidate, summary_path))
                seen.add(candidate.resolve())

    final_summary = run_dir / "evals" / args.summary_name
    if final_summary.exists() and run_dir.resolve() not in seen:
        eval_dirs.append((run_dir, final_summary))

    rows = []
    for ckpt_dir, summary_path in sorted(eval_dirs, key=lambda item: checkpoint_sort_key(item[0])):
        with summary_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        step_match = re.search(r"checkpoint-(\d+)$", ckpt_dir.name)
        step = int(step_match.group(1)) if step_match else -1
        rows.append(
            {
                "checkpoint": ckpt_dir.name,
                "step": step,
                "forget_qa_rouge": metrics.get("forget_qa_rouge"),
                "holdout_qa_rouge": metrics.get("holdout_qa_rouge"),
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("checkpoint\tstep\tforget_qa_rouge\tholdout_qa_rouge\n")
        for row in rows:
            handle.write(
                f"{row['checkpoint']}\t{row['step']}\t{row['forget_qa_rouge']}\t{row['holdout_qa_rouge']}\n"
            )


if __name__ == "__main__":
    main()
