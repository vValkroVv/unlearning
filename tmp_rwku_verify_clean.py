#!/usr/bin/env python3
"""Verify that RWKU clean/final artifacts contain no invalid counterfactuals."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (  # noqa: E402
    clean_counterfactual_text,
    counterfactual_invalid_reason,
    read_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fix-file", default=str(REPO_ROOT / "tmp_rwku_fix.txt"))
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--alternate-key", default="alternate")
    parser.add_argument("--question-key", default="query")
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    parser.add_argument("--preview", type=int, default=10)
    return parser.parse_args()


def load_fix_file(path: Path) -> dict[int, str]:
    fixes: dict[int, str] = {}
    if not path.exists():
        return fixes
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "|" not in stripped:
                raise ValueError(
                    f"Invalid fix line {line_no} in {path}: expected `index|alternate`."
                )
            index_text, alternate = stripped.split("|", 1)
            fixes[int(index_text.strip())] = clean_counterfactual_text(alternate.strip())
    return fixes


def build_index_map(rows: list[dict]) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    for row in rows:
        index = int(row["index"])
        if index in mapping:
            raise ValueError(f"Duplicate index in artifact: {index}")
        mapping[index] = row
    return mapping


def validate_rows(rows: list[dict], args) -> tuple[list[dict], Counter[str]]:
    invalid_rows: list[dict] = []
    reason_counts: Counter[str] = Counter()
    seen = set()

    for row in rows:
        index = int(row["index"])
        if index in seen:
            invalid_rows.append(
                {
                    "index": index,
                    "reason": "duplicate_index",
                    "answer": row.get(args.answer_key, ""),
                    "alternate": row.get(args.alternate_key, ""),
                    "question": row.get(args.question_key, ""),
                }
            )
            reason_counts["duplicate_index"] += 1
            continue
        seen.add(index)

        reason = counterfactual_invalid_reason(
            row.get(args.alternate_key, ""),
            row.get(args.answer_key, ""),
            reject_gold_substring=True,
            max_overlap_ratio=args.max_overlap_ratio,
            require_short_answer=True,
            max_alt_length_chars=args.max_alt_length_chars,
        )
        if reason is not None:
            invalid_rows.append(
                {
                    "index": index,
                    "reason": reason,
                    "answer": row.get(args.answer_key, ""),
                    "alternate": row.get(args.alternate_key, ""),
                    "question": row.get(args.question_key, ""),
                }
            )
            reason_counts[reason] += 1

    return invalid_rows, reason_counts


def find_final_path(out_dir: Path) -> Path | None:
    final_candidates = sorted(
        path for path in out_dir.glob("dualcf_*.jsonl") if path.is_file()
    )
    if not final_candidates:
        return None
    return max(final_candidates, key=lambda path: (path.stat().st_mtime, path.name))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    raw_path = out_dir / "step1_counterfactuals_raw.jsonl"
    clean_path = out_dir / "step1b_counterfactuals_clean.jsonl"
    final_path = find_final_path(out_dir)
    fix_file = Path(args.fix_file).expanduser().resolve()

    if not out_dir.exists():
        raise SystemExit(f"OUT_DIR does not exist: {out_dir}")
    if not raw_path.exists():
        raise SystemExit(f"Missing raw path: {raw_path}")
    if not clean_path.exists():
        raise SystemExit(f"Missing clean path: {clean_path}")

    raw_rows = read_jsonl(str(raw_path))
    clean_rows = read_jsonl(str(clean_path))
    final_rows = read_jsonl(str(final_path)) if final_path is not None else []

    raw_map = build_index_map(raw_rows)
    clean_map = build_index_map(clean_rows)
    final_map = build_index_map(final_rows) if final_rows else {}
    fixes = load_fix_file(fix_file)

    raw_indices = set(raw_map)
    clean_indices = set(clean_map)
    final_indices = set(final_map)
    fix_indices = set(fixes)

    missing_from_clean = sorted(raw_indices - clean_indices)
    extra_in_clean = sorted(clean_indices - raw_indices)
    missing_fix_indices = sorted(fix_indices - clean_indices)
    mismatched_fix_values = []
    for index, expected_alternate in fixes.items():
        if index not in clean_map:
            continue
        actual = clean_counterfactual_text(clean_map[index].get(args.alternate_key, ""))
        if actual != expected_alternate:
            mismatched_fix_values.append(
                {
                    "index": index,
                    "expected": expected_alternate,
                    "actual": actual,
                }
            )

    clean_invalid_rows, clean_reason_counts = validate_rows(clean_rows, args)
    final_invalid_rows, final_reason_counts = validate_rows(final_rows, args) if final_rows else ([], Counter())

    final_missing_from_clean = sorted(clean_indices - final_indices) if final_rows else []
    final_extra_vs_clean = sorted(final_indices - clean_indices) if final_rows else []

    passed = (
        not missing_from_clean
        and not extra_in_clean
        and not missing_fix_indices
        and not mismatched_fix_values
        and not clean_invalid_rows
        and (not final_rows or (not final_missing_from_clean and not final_extra_vs_clean and not final_invalid_rows))
    )

    print(f"OUT_DIR={out_dir}")
    print(f"raw_rows={len(raw_rows)}")
    print(f"clean_rows={len(clean_rows)}")
    print(f"final_rows={len(final_rows)}")
    print(f"fix_rows={len(fixes)} from {fix_file}")
    print(f"missing_from_clean={len(missing_from_clean)}")
    print(f"extra_in_clean={len(extra_in_clean)}")
    print(f"missing_fix_indices={len(missing_fix_indices)}")
    print(f"mismatched_fix_values={len(mismatched_fix_values)}")
    print(f"clean_invalid_rows={len(clean_invalid_rows)} reason_counts={dict(sorted(clean_reason_counts.items()))}")
    if final_rows:
        print(f"final_missing_from_clean={len(final_missing_from_clean)}")
        print(f"final_extra_vs_clean={len(final_extra_vs_clean)}")
        print(f"final_invalid_rows={len(final_invalid_rows)} reason_counts={dict(sorted(final_reason_counts.items()))}")

    preview = max(0, int(args.preview))
    if missing_from_clean[:preview]:
        print(f"missing_from_clean_sample={missing_from_clean[:preview]}")
    if missing_fix_indices[:preview]:
        print(f"missing_fix_indices_sample={missing_fix_indices[:preview]}")
    if extra_in_clean[:preview]:
        print(f"extra_in_clean_sample={extra_in_clean[:preview]}")
    if mismatched_fix_values[:preview]:
        print("mismatched_fix_values_sample=" + json.dumps(mismatched_fix_values[:preview], ensure_ascii=False))
    if clean_invalid_rows[:preview]:
        print("clean_invalid_sample=" + json.dumps(clean_invalid_rows[:preview], ensure_ascii=False))
    if final_invalid_rows[:preview]:
        print("final_invalid_sample=" + json.dumps(final_invalid_rows[:preview], ensure_ascii=False))

    print(f"verification={'passed' if passed else 'failed'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
