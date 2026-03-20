#!/usr/bin/env python3
"""Clean and optionally repair DualCF counterfactual artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (
    build_artifact_quality_report,
    build_answer_type_fallback_candidates,
    build_low_confidence_fallback_candidates,
    clean_counterfactual_text,
    counterfactual_invalid_reason,
    dedupe_candidate_metadata,
    duplicate_candidate_count,
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
    parser.add_argument("--allow-low-confidence-fallback", action="store_true")
    parser.add_argument("--drop-invalid", action="store_true")
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


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
            row_relation_scores = list(
                updated.get("candidate_relation_scores")
                or bank_row.get("candidate_relation_scores", [])
            )
            row_shared_fact_scores = list(
                updated.get("candidate_shared_fact_scores")
                or bank_row.get("candidate_shared_fact_scores", [])
            )
            row_sources = list(
                updated.get("candidate_sources")
                or bank_row.get("candidate_sources", [])
            )
            external_candidates = list(updated.get("external_alternates", []))
            external_scores = updated.get("external_alternate_scores")
            external_relation_scores = updated.get("external_alternate_relation_scores")
            external_shared_fact_scores = updated.get("external_alternate_shared_fact_scores")
            external_sources = updated.get("external_alternate_sources")
            candidate_pool = [updated[args.alternate_key]]
            score_pool = [None]
            relation_score_pool = [None]
            shared_fact_score_pool = [None]
            source_pool = ["existing_alternate"]
            candidate_pool.extend(external_candidates)
            if isinstance(external_scores, list):
                score_pool.extend(list(external_scores))
            else:
                score_pool.extend([None] * len(external_candidates))
            if isinstance(external_relation_scores, list):
                relation_score_pool.extend(list(external_relation_scores))
            else:
                relation_score_pool.extend([None] * len(external_candidates))
            if isinstance(external_shared_fact_scores, list):
                shared_fact_score_pool.extend(list(external_shared_fact_scores))
            else:
                shared_fact_score_pool.extend([None] * len(external_candidates))
            if isinstance(external_sources, list):
                source_pool.extend(
                    [str(source).strip() if str(source).strip() else "external" for source in external_sources]
                )
            else:
                source_pool.extend(["external"] * len(external_candidates))
            candidate_pool.extend(row_candidates)
            score_pool.extend([None] * len(row_candidates))
            relation_score_pool.extend(row_relation_scores[: len(row_candidates)])
            if len(row_relation_scores) < len(row_candidates):
                relation_score_pool.extend([None] * (len(row_candidates) - len(row_relation_scores)))
            shared_fact_score_pool.extend(row_shared_fact_scores[: len(row_candidates)])
            if len(row_shared_fact_scores) < len(row_candidates):
                shared_fact_score_pool.extend([None] * (len(row_candidates) - len(row_shared_fact_scores)))
            source_pool.extend(
                [str(source).strip() if str(source).strip() else "candidate_bank" for source in row_sources[: len(row_candidates)]]
            )
            if len(row_sources) < len(row_candidates):
                source_pool.extend(["candidate_bank"] * (len(row_candidates) - len(row_sources)))
            fallback_candidates = build_answer_type_fallback_candidates(
                updated.get(args.answer_key, ""),
                seed=seed,
            )
            candidate_pool.extend(fallback_candidates)
            score_pool.extend([None] * len(fallback_candidates))
            relation_score_pool.extend([None] * len(fallback_candidates))
            shared_fact_score_pool.extend([None] * len(fallback_candidates))
            source_pool.extend(["typed_fallback"] * len(fallback_candidates))
            duplicate_candidates_removed = duplicate_candidate_count(candidate_pool)
            (
                candidate_pool,
                score_pool,
                relation_score_pool,
                shared_fact_score_pool,
                source_pool,
            ) = dedupe_candidate_metadata(
                candidate_pool,
                scores=score_pool,
                relation_scores=relation_score_pool,
                shared_fact_scores=shared_fact_score_pool,
                candidate_sources=source_pool,
            )
            low_confidence_candidates = (
                build_low_confidence_fallback_candidates(updated.get(args.answer_key, ""))
                if args.allow_low_confidence_fallback
                else []
            )
            default_relation_score = None
            if updated.get("cf_same_relation") not in (None, "", "null", "None"):
                default_relation_score = 1.0 if bool(updated.get("cf_same_relation")) else 0.0
            repaired_alternate, pick_meta = pick_best_counterfactual_v3(
                question=str(updated.get("question") or updated.get("query") or ""),
                answer=str(updated.get(args.answer_key, "")),
                candidates=candidate_pool,
                candidate_answers=row_candidates,
                external_scores=score_pool,
                relation_scores=relation_score_pool,
                shared_fact_scores=shared_fact_score_pool,
                candidate_sources=source_pool,
                default_relation_score=default_relation_score,
                prompt_family=str(updated.get("cf_prompt_family") or "") or None,
                reject_gold_substring=args.reject_gold_substring,
                max_overlap_ratio=args.max_overlap_ratio,
                require_short_answer=args.require_short_answer,
                max_alt_length_chars=args.max_alt_length_chars,
            )
            if (
                pick_meta.get("invalid_reason") is not None
                and low_confidence_candidates
            ):
                candidate_pool.extend(low_confidence_candidates)
                score_pool.extend([None] * len(low_confidence_candidates))
                relation_score_pool.extend([None] * len(low_confidence_candidates))
                shared_fact_score_pool.extend([None] * len(low_confidence_candidates))
                source_pool.extend(["low_confidence_fallback"] * len(low_confidence_candidates))
                duplicate_candidates_removed = duplicate_candidate_count(candidate_pool)
                (
                    candidate_pool,
                    score_pool,
                    relation_score_pool,
                    shared_fact_score_pool,
                    source_pool,
                ) = dedupe_candidate_metadata(
                    candidate_pool,
                    scores=score_pool,
                    relation_scores=relation_score_pool,
                    shared_fact_scores=shared_fact_score_pool,
                    candidate_sources=source_pool,
                )
                repaired_alternate, pick_meta = pick_best_counterfactual_v3(
                    question=str(updated.get("question") or updated.get("query") or ""),
                    answer=str(updated.get(args.answer_key, "")),
                    candidates=candidate_pool,
                    candidate_answers=row_candidates,
                    external_scores=score_pool,
                    relation_scores=relation_score_pool,
                    shared_fact_scores=shared_fact_score_pool,
                    candidate_sources=source_pool,
                    default_relation_score=default_relation_score,
                    prompt_family=str(updated.get("cf_prompt_family") or "") or None,
                    reject_gold_substring=args.reject_gold_substring,
                    max_overlap_ratio=args.max_overlap_ratio,
                    require_short_answer=args.require_short_answer,
                    max_alt_length_chars=args.max_alt_length_chars,
                )
            if repaired_alternate:
                updated[args.alternate_key] = repaired_alternate
                pick_meta["selected_candidate"] = pick_meta.get(
                    "selected_candidate_text",
                    repaired_alternate,
                )
                pick_meta["candidate_pool_size"] = len(candidate_pool)
                pick_meta["duplicate_candidates_removed"] = int(
                    duplicate_candidates_removed
                )
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
    if args.report_path not in (None, "", "null", "None"):
        report = build_artifact_quality_report(
            cleaned_rows,
            answer_key=args.answer_key,
            alternate_key=args.alternate_key,
        )
        with open(args.report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)
        log(f"Saved artifact-quality report to {args.report_path}")
    log(
        "Done. "
        f"rows_in={len(rows)} rows_out={len(cleaned_rows)} "
        f"repaired={repaired} dropped={dropped} still_invalid={still_invalid}"
    )


if __name__ == "__main__":
    main()
