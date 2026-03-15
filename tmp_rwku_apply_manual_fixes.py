#!/usr/bin/env python3
"""Apply manual RWKU counterfactual fixes into the clean DualCF artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (  # noqa: E402
    clean_counterfactual_text,
    counterfactual_invalid_reason,
    read_jsonl,
    save_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fix-file", default=str(REPO_ROOT / "tmp_rwku_fix.txt"))
    parser.add_argument("--question-key", default="query")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--alternate-key", default="alternate")
    parser.add_argument("--raw-path", default=None)
    parser.add_argument("--clean-path", default=None)
    parser.add_argument("--backup-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    return parser.parse_args()


def load_fix_file(path: Path) -> dict[int, str]:
    fixes: dict[int, str] = {}
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
            index = int(index_text.strip())
            alternate = clean_counterfactual_text(alternate.strip())
            if not alternate:
                raise ValueError(f"Empty alternate on line {line_no} in {path}")
            if index in fixes:
                raise ValueError(f"Duplicate index {index} in {path}")
            fixes[index] = alternate
    return fixes


def build_index_map(rows: list[dict]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for row in rows:
        index = int(row["index"])
        if index in result:
            raise ValueError(f"Duplicate index in JSONL rows: {index}")
        result[index] = row
    return result


def validate_rows(rows: list[dict], args) -> list[dict]:
    invalid_rows = []
    seen_indices = set()
    for row in rows:
        index = int(row["index"])
        if index in seen_indices:
            invalid_rows.append({"index": index, "reason": "duplicate_index"})
            continue
        seen_indices.add(index)

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
                }
            )
    return invalid_rows


def main():
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    raw_path = (
        Path(args.raw_path).expanduser().resolve()
        if args.raw_path
        else out_dir / "step1_counterfactuals_raw.jsonl"
    )
    clean_path = (
        Path(args.clean_path).expanduser().resolve()
        if args.clean_path
        else out_dir / "step1b_counterfactuals_clean.jsonl"
    )
    backup_path = (
        Path(args.backup_path).expanduser().resolve()
        if args.backup_path
        else out_dir / "step1b_counterfactuals_clean.before_manual_fix.jsonl"
    )
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else out_dir / "rwku_manual_fix_summary.json"
    )
    fix_file = Path(args.fix_file).expanduser().resolve()

    if not out_dir.exists():
        raise SystemExit(f"OUT_DIR does not exist: {out_dir}")
    if not raw_path.exists():
        raise SystemExit(f"Missing raw path: {raw_path}")
    if not clean_path.exists():
        raise SystemExit(f"Missing clean path: {clean_path}")
    if not fix_file.exists():
        raise SystemExit(f"Missing fix file: {fix_file}")

    fixes = load_fix_file(fix_file)
    raw_rows = read_jsonl(str(raw_path))
    clean_rows = read_jsonl(str(clean_path))
    raw_map = build_index_map(raw_rows)
    clean_map = build_index_map(clean_rows)

    raw_indices = [int(row["index"]) for row in raw_rows]
    raw_index_set = set(raw_indices)
    clean_index_set = set(clean_map)
    fix_index_set = set(fixes)

    unknown_fix_indices = sorted(fix_index_set - raw_index_set)
    if unknown_fix_indices:
        raise SystemExit(
            f"Fix file contains indices not present in raw JSONL: {unknown_fix_indices}"
        )

    missing_from_clean = sorted(raw_index_set - clean_index_set)
    unexpected_missing = sorted(set(missing_from_clean) - fix_index_set)
    if unexpected_missing:
        raise SystemExit(
            "Clean JSONL has dropped rows that are not covered by the fix file: "
            f"{unexpected_missing}"
        )

    fixed_rows_by_index = {index: dict(row) for index, row in clean_map.items()}
    applied_indices: list[int] = []

    for index, alternate in fixes.items():
        base_row = dict(clean_map.get(index) or raw_map[index])
        base_row[args.alternate_key] = alternate
        base_row["cf_source"] = "manual_fix"
        base_row["cf_generator_backend"] = "manual_fix"
        base_row["cf_answer_type"] = "manual_fix"
        base_row["cf_manual_fix"] = True
        base_row["cf_manual_fix_value"] = alternate
        base_row["cf_invalid_reason"] = None
        base_row["cf_is_valid"] = True
        if "cf_raw_alternate" not in base_row:
            base_row["cf_raw_alternate"] = raw_map[index].get(args.alternate_key, "")
        fixed_rows_by_index[index] = base_row
        applied_indices.append(index)

    rebuilt_clean_rows = []
    for index in raw_indices:
        if index not in fixed_rows_by_index:
            raise SystemExit(f"Index {index} is missing from rebuilt clean rows")
        rebuilt_clean_rows.append(fixed_rows_by_index[index])

    invalid_rows = validate_rows(rebuilt_clean_rows, args)
    if invalid_rows:
        raise SystemExit(
            "Manual fixes still contain invalid rows: "
            + json.dumps(invalid_rows[:10], ensure_ascii=False)
        )

    if len(rebuilt_clean_rows) != len(raw_rows):
        raise SystemExit(
            f"Rebuilt clean row count {len(rebuilt_clean_rows)} != raw row count {len(raw_rows)}"
        )

    if not backup_path.exists():
        shutil.copyfile(clean_path, backup_path)

    save_jsonl(rebuilt_clean_rows, str(clean_path))

    summary = {
        "out_dir": str(out_dir),
        "fix_file": str(fix_file),
        "raw_path": str(raw_path),
        "clean_path": str(clean_path),
        "backup_path": str(backup_path),
        "summary_path": str(summary_path),
        "raw_rows": len(raw_rows),
        "clean_rows_before": len(clean_rows),
        "clean_rows_after": len(rebuilt_clean_rows),
        "manual_fix_count": len(fixes),
        "missing_from_clean_before": len(missing_from_clean),
        "applied_indices": sorted(applied_indices),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"fix_file={fix_file}")
    print(f"raw_path={raw_path}")
    print(f"clean_path={clean_path}")
    print(f"backup_path={backup_path}")
    print(f"summary_path={summary_path}")
    print(f"clean_rows_before={len(clean_rows)}")
    print(f"clean_rows_after={len(rebuilt_clean_rows)}")
    print(f"manual_fix_count={len(fixes)}")
    print(f"missing_from_clean_before={len(missing_from_clean)}")
    print("validation=passed")


if __name__ == "__main__":
    main()
