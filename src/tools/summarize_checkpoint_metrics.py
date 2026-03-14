#!/usr/bin/env python3
"""Summarize checkpoint evaluation metrics for DualCF trajectory runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoint_summary_utils import collect_eval_summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-name", default="DUET_SUMMARY.json")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    rows = []
    for row in collect_eval_summaries(
        run_dir=run_dir,
        eval_root_name="checkpoint_evals",
        summary_name=args.summary_name,
    ):
        summary_path = Path(row["summary_path"])
        with summary_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "label": row["label"],
                "checkpoint": row["label"],
                "step": row["step"],
                "epoch": row["epoch"],
                "forget_qa_rouge": metrics.get("forget_qa_rouge"),
                "holdout_qa_rouge": metrics.get("holdout_qa_rouge"),
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "label\tcheckpoint\tstep\tepoch\tforget_qa_rouge\tholdout_qa_rouge\n"
        )
        for row in rows:
            handle.write(
                f"{row['label']}\t{row['checkpoint']}\t"
                f"{'' if row['step'] is None else row['step']}\t"
                f"{'' if row['epoch'] is None else row['epoch']}\t"
                f"{row['forget_qa_rouge']}\t{row['holdout_qa_rouge']}\n"
            )


if __name__ == "__main__":
    main()
