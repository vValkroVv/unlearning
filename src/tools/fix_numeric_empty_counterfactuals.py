#!/usr/bin/env python3
"""Fill empty counterfactual rows with a deterministic numeric fallback."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Optional

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (  # noqa: E402
    clean_counterfactual_text,
    counterfactual_invalid_reason,
    read_jsonl,
    save_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--alternate-key", default="alternate")
    parser.add_argument("--mapping-key", default="index")
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    return parser.parse_args()


def invalid_reason(args: argparse.Namespace, alternate: str, answer: Any) -> Optional[str]:
    return counterfactual_invalid_reason(
        alternate,
        answer,
        reject_gold_substring=args.reject_gold_substring,
        max_overlap_ratio=args.max_overlap_ratio,
        require_short_answer=args.require_short_answer,
        max_alt_length_chars=args.max_alt_length_chars,
    )


def ordinal_suffix(value: int) -> str:
    if 10 <= (value % 100) <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def perturb_int_text(text: str, seed: int) -> Optional[str]:
    match = re.fullmatch(r"([+-]?)(\d+)", text)
    if not match:
        return None
    sign, digits = match.groups()
    value = int(f"{sign}{digits}")
    step = 1 if seed % 2 == 0 else -1
    if value == 0:
        step = 1
    if value > 0 and value + step <= 0:
        step = 1
    return str(value + step)


def perturb_decimal_text(text: str, seed: int) -> Optional[str]:
    match = re.fullmatch(r"([+-]?\d+)(\.\d+)", text)
    if not match:
        return None
    value = float(text)
    step = 0.1 if seed % 2 == 0 else -0.1
    decimals = len(match.group(2)) - 1
    return f"{value + step:.{decimals}f}"


def perturb_decade_text(text: str, seed: int) -> Optional[str]:
    match = re.fullmatch(r"(\d{3,4})s", text)
    if not match:
        return None
    value = int(match.group(1))
    step = 10 if seed % 2 == 0 else -10
    return f"{max(0, value + step)}s"


def perturb_ordinal_text(text: str, seed: int) -> Optional[str]:
    match = re.fullmatch(r"(\d+)(st|nd|rd|th)", text.lower())
    if not match:
        return None
    value = int(match.group(1))
    step = 1 if seed % 2 == 0 else -1
    if value <= 1 and step < 0:
        step = 1
    candidate = value + step
    return f"{candidate}{ordinal_suffix(candidate)}"


def numeric_fallback(answer: Any, seed: int) -> Optional[str]:
    text = str(answer).strip()
    if not text:
        return None

    percent_suffix = "%" if text.endswith("%") else ""
    normalized = text.replace(",", "").replace("%", "").strip()

    for builder in (
        perturb_ordinal_text,
        perturb_decade_text,
        perturb_decimal_text,
        perturb_int_text,
    ):
        candidate = builder(normalized, seed)
        if candidate is not None:
            return f"{candidate}{percent_suffix}"
    return None


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)

    patched = 0
    skipped_non_numeric = 0
    still_invalid = 0

    for row in rows:
        raw_alternate = str(row.get("cf_raw_alternate", row.get(args.alternate_key, "")))
        cleaned_alternate = clean_counterfactual_text(raw_alternate)
        reason = invalid_reason(args, cleaned_alternate, row.get(args.answer_key, ""))

        if reason != "empty":
            continue

        seed = int(row.get(args.mapping_key, 0) or 0)
        fallback = numeric_fallback(row.get(args.answer_key, ""), seed)
        if fallback is None:
            skipped_non_numeric += 1
            continue

        fallback_reason = invalid_reason(args, fallback, row.get(args.answer_key, ""))
        if fallback_reason is not None:
            still_invalid += 1
            continue

        row["cf_raw_alternate"] = raw_alternate
        row[args.alternate_key] = fallback
        row["cf_invalid_reason"] = None
        row["cf_is_valid"] = True
        row["cf_source"] = "numeric_rule_fallback"
        row["cf_answer_type"] = "numeric_rule_fallback"
        row["cf_same_relation"] = True
        row["cf_rule_fallback_from_answer"] = str(row.get(args.answer_key, ""))
        patched += 1

    save_jsonl(rows, args.output_path)
    print(
        "[fix_numeric_empty_counterfactuals] "
        f"rows={len(rows)} patched={patched} "
        f"skipped_non_numeric={skipped_non_numeric} still_invalid={still_invalid}",
        flush=True,
    )


if __name__ == "__main__":
    main()
