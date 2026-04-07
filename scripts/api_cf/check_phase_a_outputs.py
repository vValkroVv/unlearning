#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_sidecar(path: Path) -> None:
    rows = load_jsonl(path)
    seen: set[int] = set()

    for line_no, row in enumerate(rows, start=1):
        idx = int(row["index"])
        if idx in seen:
            fail(f"{path}: duplicate index={idx}")
        seen.add(idx)

        alternates = row.get("alternates")
        if not isinstance(alternates, list) or not alternates:
            fail(f"{path}: line {line_no} has empty alternates")

        for key in ("scores", "relation_scores", "shared_fact_scores", "candidate_sources"):
            value = row.get(key)
            if not isinstance(value, list):
                fail(f"{path}: line {line_no} {key} is not a list")
            if len(value) != len(alternates):
                fail(
                    f"{path}: line {line_no} {key} len={len(value)} "
                    f"!= alternates len={len(alternates)}"
                )

    print(f"[ok] sidecar rows={len(rows)} path={path}")


def check_clean(path: Path, question_key: str) -> None:
    rows = load_jsonl(path)
    seen: set[int] = set()
    failures: list[tuple[int, str]] = []

    for line_no, row in enumerate(rows, start=1):
        idx = int(row["index"])
        if idx in seen:
            failures.append((line_no, f"duplicate index={idx}"))
        seen.add(idx)

        for key in (question_key, "answer", "alternate"):
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                failures.append((line_no, f"empty/non-string {key}"))

        alternate = str(row.get("alternate", "")).strip().lower()
        answer = str(row.get("answer", "")).strip().lower()
        if alternate == answer:
            failures.append((line_no, "alternate == answer"))

        if row.get("cf_invalid_reason") not in (None, "", "null", "None"):
            failures.append((line_no, f"cf_invalid_reason={row.get('cf_invalid_reason')!r}"))

        if "cf_is_valid" in row and row["cf_is_valid"] is not True:
            failures.append((line_no, "cf_is_valid != True"))

    if failures:
        print(failures[:20], file=sys.stderr)
        fail(f"{path}: found {len(failures)} bad rows")

    print(f"[ok] clean rows={len(rows)} path={path}")


def check_report(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"[report] {path}")
    for key in (
        "rows",
        "valid_row_rate",
        "exact_match_count",
        "gold_substring_count",
        "invalid_reason_counts",
        "repair_source_counts",
        "sidecar_coverage_rate",
        "relation_metadata_coverage_rate",
        "shared_fact_metadata_coverage_rate",
    ):
        if key in report:
            print(f" {key}: {report[key]}")

    if report.get("valid_row_rate") != 1.0:
        fail(f"{path}: valid_row_rate != 1.0")
    if report.get("exact_match_count", 0) != 0:
        fail(f"{path}: exact_match_count != 0")
    if report.get("gold_substring_count", 0) != 0:
        fail(f"{path}: gold_substring_count != 0")
    if report.get("invalid_reason_counts"):
        fail(f"{path}: invalid_reason_counts not empty")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("duet", "rwku"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--question-key", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    sidecar = out_dir / "api_sidecar.jsonl"
    check_sidecar(sidecar)

    summary_path = out_dir / "api_sidecar.jsonl.summary.json"
    if summary_path.exists():
        print(f"[ok] exists: {summary_path}")

    if args.dataset == "duet":
        candidate_bank = out_dir / "step0_candidate_bank.jsonl"
        if not candidate_bank.exists():
            fail(f"missing {candidate_bank}")
        print(f"[ok] candidate bank exists: {candidate_bank}")

    raw_cf = out_dir / "step1_counterfactuals_raw_v3.jsonl"
    clean_cf = out_dir / "step1b_counterfactuals_clean_v3.jsonl"
    clean_report = out_dir / "step1b_clean_report.json"

    for path in (raw_cf, clean_cf, clean_report):
        if not path.exists():
            fail(f"missing {path}")
        print(f"[ok] exists: {path}")

    check_clean(clean_cf, args.question_key)
    check_report(clean_report)

    print("[done] all phase-A checks passed")


if __name__ == "__main__":
    main()
