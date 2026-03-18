#!/usr/bin/env python3
"""Validate a DualCF JSONL artifact before training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (
    BAD_CF_PREFIXES,
    build_artifact_quality_report,
    clean_counterfactual_text,
    counterfactual_invalid_reason,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a DualCF JSONL artifact.")
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--question-key", choices=("question", "query"), default="question")
    parser.add_argument("--max-bad-rows", type=int, default=10)
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--max-alt-length-chars", type=int, default=None)
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--check-overlap-ratio", type=float, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def _count_duplicate_candidates(values):
    seen = set()
    duplicates = 0
    for value in values if isinstance(values, list) else []:
        cleaned = clean_counterfactual_text(value)
        if not cleaned:
            continue
        if cleaned in seen:
            duplicates += 1
            continue
        seen.add(cleaned)
    return duplicates


def _list_length_valid(values, expected_length: int) -> bool:
    if not isinstance(values, list):
        return False
    return len(values) in (0, expected_length)


def main():
    args = parse_args()
    artifact_path = args.input_path or args.artifact_path
    if not artifact_path:
        raise ValueError("Pass --artifact-path or --input-path")
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    required = {
        "index",
        args.question_key,
        "answer",
        "alternate",
        "difficulty_score",
        "attribution_score",
    }
    seen_indices = set()
    duplicate_indices = set()
    bad_rows = []
    ranges = {
        "difficulty_score": [float("inf"), float("-inf")],
        "attribution_score": [float("inf"), float("-inf")],
    }
    optional_numeric_keys = (
        "difficulty_score_raw",
        "attribution_score_raw",
        "attribution_score_raw_global",
        "attribution_score_raw_syntax",
        "attribution_score_raw_semantic",
        "attribution_score_raw_utility",
    )
    invalid_reason_counts = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                bad_rows.append((line_no, "empty line"))
                continue

            row = json.loads(line)
            missing = sorted(required - set(row))
            if missing:
                bad_rows.append((line_no, f"missing keys: {missing}"))
                continue

            index = row["index"]
            if index in seen_indices:
                duplicate_indices.add(index)
            seen_indices.add(index)

            for key in (args.question_key, "answer", "alternate"):
                value = row.get(key)
                if not isinstance(value, str) or not value.strip():
                    bad_rows.append((line_no, f"empty or non-string {key}"))

            for score_key in ("difficulty_score", "attribution_score"):
                value = row.get(score_key)
                if not isinstance(value, (int, float)):
                    bad_rows.append((line_no, f"{score_key} is not numeric: {value!r}"))
                    continue
                if not math.isfinite(value):
                    bad_rows.append((line_no, f"{score_key} is not finite: {value!r}"))
                    continue
                ranges[score_key][0] = min(ranges[score_key][0], float(value))
                ranges[score_key][1] = max(ranges[score_key][1], float(value))
                if args.strict and not (0.0 <= float(value) <= 1.0):
                    bad_rows.append((line_no, f"{score_key} out of [0,1]: {value!r}"))

            for score_key in optional_numeric_keys:
                if score_key not in row:
                    continue
                value = row.get(score_key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    bad_rows.append((line_no, f"{score_key} is not finite numeric: {value!r}"))

            invalid_reason = counterfactual_invalid_reason(
                row["alternate"],
                row["answer"],
                reject_gold_substring=args.reject_gold_substring,
                max_overlap_ratio=args.check_overlap_ratio,
                require_short_answer=args.require_short_answer,
                max_alt_length_chars=args.max_alt_length_chars,
            )
            if invalid_reason is not None:
                bad_rows.append((line_no, f"invalid alternate: {invalid_reason}"))
                invalid_reason_counts[invalid_reason] = (
                    invalid_reason_counts.get(invalid_reason, 0) + 1
                )

            alternate_lower = row["alternate"].strip().lower()
            for prefix in BAD_CF_PREFIXES:
                if alternate_lower.startswith(prefix):
                    bad_rows.append((line_no, f"alternate kept banned prefix `{prefix}`"))
                    break

            if args.strict:
                for key in ("difficulty_components", "attribution_components"):
                    if key in row and not isinstance(row[key], dict):
                        bad_rows.append((line_no, f"{key} is not a dict"))
                for key in (
                    "candidate_answers",
                    "candidate_relation_scores",
                    "candidate_shared_fact_scores",
                    "candidate_sources",
                    "external_alternates",
                    "external_alternate_scores",
                    "external_alternate_relation_scores",
                    "external_alternate_shared_fact_scores",
                    "external_alternate_sources",
                    "belief_candidates",
                ):
                    if key in row and not isinstance(row[key], list):
                        bad_rows.append((line_no, f"{key} is not a list"))
                for key in ("cf_pick_meta", "belief_pick_meta", "cf_provenance"):
                    if key in row and not isinstance(row[key], dict):
                        bad_rows.append((line_no, f"{key} is not a dict"))
                for key in ("cf_is_valid",):
                    if key in row and not isinstance(row[key], bool):
                        bad_rows.append((line_no, f"{key} is not a bool"))
                for key in ("belief_alternate",):
                    if key in row and not isinstance(row[key], str):
                        bad_rows.append((line_no, f"{key} is not a string"))

            if "external_alternates" in row and "external_alternate_scores" in row:
                alternates = row.get("external_alternates")
                scores = row.get("external_alternate_scores")
                if isinstance(alternates, list) and scores is not None and not isinstance(scores, list):
                    bad_rows.append((line_no, "external_alternate_scores is not a list"))
                if isinstance(alternates, list) and isinstance(scores, list):
                    if len(scores) not in (0, len(alternates)):
                        bad_rows.append(
                            (
                                line_no,
                                "external_alternate_scores length does not match external_alternates",
                            )
                        )
                for meta_key in (
                    "external_alternate_relation_scores",
                    "external_alternate_shared_fact_scores",
                    "external_alternate_sources",
                ):
                    meta_values = row.get(meta_key)
                    if meta_values is not None and isinstance(alternates, list):
                        if not _list_length_valid(meta_values, len(alternates)):
                            bad_rows.append(
                                (
                                    line_no,
                                    f"{meta_key} length does not match external_alternates",
                                )
                            )
            if "candidate_answers" in row:
                candidate_answers = row.get("candidate_answers")
                if isinstance(candidate_answers, list):
                    for meta_key in (
                        "candidate_relation_scores",
                        "candidate_shared_fact_scores",
                        "candidate_sources",
                    ):
                        meta_values = row.get(meta_key)
                        if meta_values is not None and not _list_length_valid(
                            meta_values,
                            len(candidate_answers),
                        ):
                            bad_rows.append(
                                (
                                    line_no,
                                    f"{meta_key} length does not match candidate_answers",
                                )
                            )
            if "belief_candidates" in row and "belief_alternate" in row:
                beliefs = row.get("belief_candidates")
                belief_alt = row.get("belief_alternate")
                if isinstance(beliefs, list) and isinstance(belief_alt, str):
                    if belief_alt and belief_alt not in beliefs:
                        bad_rows.append(
                            (line_no, "belief_alternate is not present in belief_candidates")
                        )
            pick_meta = row.get("cf_pick_meta")
            if isinstance(pick_meta, dict):
                selected_text = str(
                    pick_meta.get("selected_candidate")
                    or pick_meta.get("selected_candidate_text")
                    or ""
                ).strip()
                if selected_text and clean_counterfactual_text(selected_text) != clean_counterfactual_text(
                    row.get("alternate", "")
                ):
                    bad_rows.append(
                        (line_no, "cf_pick_meta selected_candidate_text does not match alternate")
                    )
                selected_index = pick_meta.get("selected_candidate_index")
                if selected_index not in (None, "") and not isinstance(selected_index, int):
                    bad_rows.append((line_no, "cf_pick_meta selected_candidate_index is not an int"))
                candidate_pool_size = pick_meta.get("candidate_pool_size")
                if candidate_pool_size not in (None, ""):
                    if not isinstance(candidate_pool_size, int) or candidate_pool_size < 0:
                        bad_rows.append(
                            (line_no, "cf_pick_meta candidate_pool_size is invalid")
                        )
                    elif isinstance(selected_index, int) and not (
                        0 <= selected_index < candidate_pool_size
                    ):
                        bad_rows.append(
                            (
                                line_no,
                                "cf_pick_meta selected_candidate_index is out of range",
                            )
                        )
                duplicate_removed = pick_meta.get("duplicate_candidates_removed")
                if duplicate_removed not in (None, "") and (
                    not isinstance(duplicate_removed, int) or duplicate_removed < 0
                ):
                    bad_rows.append(
                        (
                            line_no,
                            "cf_pick_meta duplicate_candidates_removed is invalid",
                        )
                    )
                if args.strict and "selected_source" not in pick_meta:
                    bad_rows.append((line_no, "cf_pick_meta missing selected_source"))
            provenance = row.get("cf_provenance")
            if isinstance(provenance, dict) and args.strict:
                for key in ("generator_backend", "prompt_family", "candidate_count", "prompt_version"):
                    if key not in provenance:
                        bad_rows.append((line_no, f"cf_provenance missing `{key}`"))

    report = build_artifact_quality_report(
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()],
        question_key=args.question_key,
        answer_key="answer",
        alternate_key="alternate",
    )
    print(f"artifact={path}")
    print(f"rows={len(seen_indices)}")
    print(f"duplicate_indices={sorted(duplicate_indices)}")
    print(f"bad_rows_count={len(bad_rows)}")
    print(f"bad_rows_sample={bad_rows[: max(0, int(args.max_bad_rows))]}")
    print(f"invalid_reason_counts={invalid_reason_counts}")
    print(
        "ranges="
        + str({key: tuple(value) for key, value in ranges.items() if value[0] != float("inf")})
    )
    print(f"artifact_quality={report}")

    if args.report_path not in (None, "", "null", "None"):
        payload = {
            "artifact": str(path),
            "rows": len(seen_indices),
            "duplicate_indices": sorted(duplicate_indices),
            "bad_rows_count": len(bad_rows),
            "bad_rows_sample": bad_rows[: max(0, int(args.max_bad_rows))],
            "invalid_reason_counts": invalid_reason_counts,
            "ranges": {
                key: tuple(value)
                for key, value in ranges.items()
                if value[0] != float("inf")
            },
            "artifact_quality": report,
        }
        with open(args.report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    if duplicate_indices or bad_rows:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
