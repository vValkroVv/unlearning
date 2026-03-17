#!/usr/bin/env python3
"""Clean and optionally repair DualCF counterfactual artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (
    build_answer_type_fallback_candidates,
    clean_counterfactual_text,
    counterfactual_invalid_reason,
    load_keyed_jsonish,
    pick_best_counterfactual_v3,
    read_jsonl,
    save_jsonl,
)


def log(message: str) -> None:
    print(f"[clean_counterfactuals] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--alternate-key", default="alternate")
    parser.add_argument("--mapping-key", default="index")
    parser.add_argument("--candidate-bank", default=None)
    parser.add_argument("--repair-invalid", action="store_true")
    parser.add_argument("--drop-invalid", action="store_true")
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    return parser.parse_args()


def _filter_scored_candidates(candidates, scores):
    filtered_candidates = []
    filtered_scores = []
    for index, candidate in enumerate(candidates):
        candidate_text = clean_counterfactual_text(candidate)
        if not candidate_text:
            continue
        filtered_candidates.append(candidate_text)
        filtered_scores.append(scores[index] if index < len(scores) else None)
    return filtered_candidates, filtered_scores


def main():
    args = parse_args()
    rows = read_jsonl(args.input_path)
    candidate_bank = (
        load_keyed_jsonish(args.candidate_bank, key_field=args.mapping_key)
        if args.candidate_bank not in (None, "", "null", "None")
        else {}
    )

    cleaned_rows = []
    repaired = 0
    dropped = 0
    still_invalid = 0
    for row in rows:
        updated = dict(row)
        seed = int(updated.get(args.mapping_key, 0) or 0)
        raw_alternate = str(updated.get(args.alternate_key, ""))
        updated["cf_raw_alternate"] = raw_alternate
        updated[args.alternate_key] = clean_counterfactual_text(raw_alternate)

        invalid_reason = counterfactual_invalid_reason(
            updated[args.alternate_key],
            updated.get(args.answer_key, ""),
            reject_gold_substring=args.reject_gold_substring,
            max_overlap_ratio=args.max_overlap_ratio,
            require_short_answer=args.require_short_answer,
            max_alt_length_chars=args.max_alt_length_chars,
        )
        if invalid_reason and args.repair_invalid:
            bank_row = candidate_bank.get(str(updated.get(args.mapping_key, "")), {})
            row_candidates = list(
                updated.get("candidate_answers") or bank_row.get("candidate_answers", [])
            )
            external_candidates = list(updated.get("external_alternates", []))
            external_scores = updated.get("external_alternate_scores")
            candidate_pool = [updated[args.alternate_key]]
            score_pool = [None]
            candidate_pool.extend(external_candidates)
            if isinstance(external_scores, list):
                score_pool.extend(list(external_scores))
            else:
                score_pool.extend([None] * len(external_candidates))
            candidate_pool.extend(row_candidates)
            score_pool.extend([None] * len(row_candidates))
            fallback_candidates = build_answer_type_fallback_candidates(
                updated.get(args.answer_key, ""),
                seed=seed,
            )
            candidate_pool.extend(fallback_candidates)
            score_pool.extend([None] * len(fallback_candidates))
            candidate_pool, score_pool = _filter_scored_candidates(candidate_pool, score_pool)
            repaired_alternate, pick_meta = pick_best_counterfactual_v3(
                question=str(updated.get("question") or updated.get("query") or ""),
                answer=str(updated.get(args.answer_key, "")),
                candidates=candidate_pool,
                candidate_answers=row_candidates,
                external_scores=score_pool,
                reject_gold_substring=args.reject_gold_substring,
                max_overlap_ratio=args.max_overlap_ratio,
                require_short_answer=args.require_short_answer,
                max_alt_length_chars=args.max_alt_length_chars,
            )
            if repaired_alternate:
                updated[args.alternate_key] = repaired_alternate
                updated["cf_pick_meta"] = pick_meta
                invalid_reason = pick_meta.get("invalid_reason")
                if invalid_reason is None:
                    repaired += 1

        updated["cf_invalid_reason"] = invalid_reason
        updated["cf_is_valid"] = invalid_reason is None

        if invalid_reason is not None:
            still_invalid += 1
            if args.drop_invalid:
                dropped += 1
                continue

        cleaned_rows.append(updated)

    save_jsonl(cleaned_rows, args.output_path)
    log(
        "Done. "
        f"rows_in={len(rows)} rows_out={len(cleaned_rows)} "
        f"repaired={repaired} dropped={dropped} still_invalid={still_invalid}"
    )


if __name__ == "__main__":
    main()
