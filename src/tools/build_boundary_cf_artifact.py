#!/usr/bin/env python3
"""Build a sibling BoundaryCF artifact with local retain matches."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (  # noqa: E402
    counterfactual_invalid_reason,
    lexical_overlap_ratio,
    load_dataset_split,
    load_keyed_jsonish,
    normalize_text,
    read_jsonl,
    save_jsonl,
)


def log(message: str) -> None:
    print(f"[build_boundary_cf_artifact] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--proxy-map-path", required=True)
    parser.add_argument("--retain-dataset-path", required=True)
    parser.add_argument("--retain-split", required=True)
    parser.add_argument("--retain-dataset-name", default=None)
    parser.add_argument("--retain-data-files", default=None)
    parser.add_argument("--retain-question-key", default="question")
    parser.add_argument("--retain-answer-key", default="answer")
    parser.add_argument("--candidate-bank", default=None)
    parser.add_argument("--mapping-key", default="index")
    parser.add_argument("--candidate-field", default="candidate_answers")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--alternate-key", default="alternate")
    parser.add_argument("--external-alternates-key", default="external_alternates")
    parser.add_argument("--external-scores-key", default="external_alternate_scores")
    parser.add_argument(
        "--external-relation-scores-key",
        default="external_alternate_relation_scores",
    )
    parser.add_argument(
        "--external-shared-fact-scores-key",
        default="external_alternate_shared_fact_scores",
    )
    parser.add_argument("--external-sources-key", default="external_alternate_sources")
    parser.add_argument("--external-sidecar-path", default=None)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.35)
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--min-relation-score", type=float, default=0.8)
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    parser.add_argument("--sidecar-path", default=None)
    return parser.parse_args()


def load_rows(path: str):
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise TypeError(f"Expected a JSON list at {path_obj}, got {type(payload)}")
        return payload
    return read_jsonl(path)


def infer_answer_type(text: str) -> str:
    normalized = normalize_text(text)
    compact = normalized.replace(",", "").replace(".", "")
    if compact.isdigit():
        return "number"
    if re.search(r"\b\d{4}\b", normalized):
        return "year"
    if re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        normalized,
    ):
        return "date"
    token_count = len([token for token in normalized.split(" ") if token])
    if token_count <= 1:
        return "token"
    if token_count <= 3:
        return "short_phrase"
    return "phrase"


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_optional_float(value: Any):
    if value in (None, "", "null", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_external_sidecar_path(args) -> str | None:
    if args.external_sidecar_path not in (None, "", "null", "None"):
        return str(Path(args.external_sidecar_path))
    candidate = Path(args.input_path).resolve().parent / "api_sidecar.jsonl"
    if candidate.exists():
        return str(candidate)
    return None


def _load_optional_keyed_jsonish(path: str | None, key_field: str):
    if path in (None, "", "null", "None"):
        return {}
    return load_keyed_jsonish(path, key_field=key_field)


def _add_candidate(candidates, seen, *, text, source, source_rank, relation_score, shared_fact_score, external_score):
    if text is None:
        return
    cleaned = str(text).strip()
    if not cleaned:
        return
    key = normalize_text(cleaned)
    if key in seen:
        return
    seen.add(key)
    candidates.append(
        {
            "text": cleaned,
            "source": source,
            "source_rank": int(source_rank),
            "relation_score": _coerce_optional_float(relation_score),
            "shared_fact_score": _coerce_optional_float(shared_fact_score),
            "external_score": _coerce_optional_float(external_score),
        }
    )


def build_external_candidate_pool(row, sidecar_row, args):
    external_alternates = _coerce_list(row.get(args.external_alternates_key))
    if not external_alternates and sidecar_row:
        external_alternates = _coerce_list(sidecar_row.get("alternates"))
    if not external_alternates:
        return []

    scores = _coerce_list(row.get(args.external_scores_key))
    relation_scores = _coerce_list(row.get(args.external_relation_scores_key))
    shared_fact_scores = _coerce_list(row.get(args.external_shared_fact_scores_key))
    sources = _coerce_list(row.get(args.external_sources_key))
    if sidecar_row:
        if not scores:
            scores = _coerce_list(sidecar_row.get("scores"))
        if not relation_scores:
            relation_scores = _coerce_list(sidecar_row.get("relation_scores"))
        if not shared_fact_scores:
            shared_fact_scores = _coerce_list(sidecar_row.get("shared_fact_scores"))
        if not sources:
            sources = _coerce_list(sidecar_row.get("candidate_sources"))

    candidates = []
    seen = set()
    for idx, text in enumerate(external_alternates):
        _add_candidate(
            candidates,
            seen,
            text=text,
            source=sources[idx] if idx < len(sources) and sources[idx] not in (None, "") else "external_alternates",
            source_rank=idx,
            relation_score=relation_scores[idx] if idx < len(relation_scores) else None,
            shared_fact_score=shared_fact_scores[idx] if idx < len(shared_fact_scores) else None,
            external_score=scores[idx] if idx < len(scores) else None,
        )
    return candidates


def build_legacy_candidate_pool(row, bank_row, candidate_field: str, alternate_key: str):
    candidates = []
    seen = set()

    _add_candidate(
        candidates,
        seen,
        text=row.get(alternate_key),
        source="artifact_top1",
        source_rank=0,
        relation_score=None,
        shared_fact_score=None,
        external_score=None,
    )

    for idx, text in enumerate(_coerce_list(row.get(candidate_field))):
        relation_scores = _coerce_list(row.get("candidate_relation_scores"))
        shared_fact_scores = _coerce_list(row.get("candidate_shared_fact_scores"))
        sources = _coerce_list(row.get("candidate_sources"))
        _add_candidate(
            candidates,
            seen,
            text=text,
            source=sources[idx] if idx < len(sources) and sources[idx] not in (None, "") else "artifact_candidates",
            source_rank=idx,
            relation_score=relation_scores[idx] if idx < len(relation_scores) else None,
            shared_fact_score=shared_fact_scores[idx] if idx < len(shared_fact_scores) else None,
            external_score=None,
        )

    bank_candidates = _coerce_list(bank_row.get(candidate_field)) if bank_row else []
    bank_relation_scores = _coerce_list(bank_row.get("candidate_relation_scores")) if bank_row else []
    bank_shared_fact_scores = _coerce_list(bank_row.get("candidate_shared_fact_scores")) if bank_row else []
    bank_sources = _coerce_list(bank_row.get("candidate_sources")) if bank_row else []
    for idx, text in enumerate(bank_candidates):
        _add_candidate(
            candidates,
            seen,
            text=text,
            source=bank_sources[idx] if idx < len(bank_sources) and bank_sources[idx] not in (None, "") else "candidate_bank",
            source_rank=idx,
            relation_score=bank_relation_scores[idx] if idx < len(bank_relation_scores) else None,
            shared_fact_score=bank_shared_fact_scores[idx] if idx < len(bank_shared_fact_scores) else None,
            external_score=None,
        )
    return candidates


def score_boundary_candidate(candidate, gold_answer: str):
    lexical_overlap = lexical_overlap_ratio(candidate["text"], gold_answer)
    shared_fact = candidate["shared_fact_score"]
    if shared_fact is None:
        shared_fact = difflib.SequenceMatcher(
            None,
            normalize_text(candidate["text"]),
            normalize_text(gold_answer),
        ).ratio()
    relation_score = candidate["relation_score"]
    if relation_score is None:
        relation_score = shared_fact
    type_match = 1.0 if infer_answer_type(candidate["text"]) == infer_answer_type(gold_answer) else 0.0
    boundary_score = (
        0.35 * float(relation_score)
        + 0.20 * float(shared_fact)
        + 0.15 * type_match
        + 0.30 * lexical_overlap
    )
    payload = {
        "boundary_score": boundary_score,
        "boundary_relation": float(relation_score),
        "boundary_relation_score": float(relation_score),
        "boundary_shared_fact_score": float(shared_fact),
        "boundary_type_match": type_match,
        "boundary_overlap": lexical_overlap,
        "boundary_lexical_overlap": lexical_overlap,
    }
    if candidate["external_score"] is not None:
        payload["boundary_external_score"] = float(candidate["external_score"])
    return payload


def select_boundary_candidate(row, candidates, args):
    strict_scored = []
    relation_type_scored = []
    relation_only_scored = []
    valid_scored = []
    for candidate in candidates:
        invalid_reason = counterfactual_invalid_reason(
            candidate["text"],
            row["answer"],
            reject_gold_substring=args.reject_gold_substring,
            max_overlap_ratio=args.max_overlap_ratio,
            require_short_answer=args.require_short_answer,
            max_alt_length_chars=args.max_alt_length_chars,
        )
        if invalid_reason is not None:
            continue

        score_payload = score_boundary_candidate(candidate, row["answer"])
        relation_ok = score_payload["boundary_relation"] >= args.min_relation_score
        type_ok = score_payload["boundary_type_match"] > 0.0
        overlap_ok = score_payload["boundary_overlap"] >= args.min_overlap_ratio
        bucket = (score_payload["boundary_score"], candidate, score_payload)
        valid_scored.append(bucket)
        if relation_ok:
            relation_only_scored.append(bucket)
        if relation_ok and type_ok:
            relation_type_scored.append(bucket)
        if relation_ok and type_ok and overlap_ok:
            strict_scored.append(bucket)

    for selection_mode, scored in (
        ("strict_overlap", strict_scored),
        ("relation_type_fallback", relation_type_scored),
        ("relation_only_fallback", relation_only_scored),
        ("valid_fallback", valid_scored),
    ):
        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        candidate, score_payload = scored[0][1], dict(scored[0][2])
        score_payload["boundary_selection_mode"] = selection_mode
        score_payload["boundary_passed_overlap_gate"] = selection_mode == "strict_overlap"
        score_payload["boundary_passed_relation_gate"] = selection_mode != "valid_fallback"
        score_payload["boundary_passed_type_gate"] = selection_mode in (
            "strict_overlap",
            "relation_type_fallback",
        )
        return candidate, score_payload

    if not valid_scored:
        raise ValueError(
            f"No boundary candidate survived for index={row.get('index')} "
            f"question={row.get(args.question_key)!r}"
        )
    raise RuntimeError("Boundary candidate selection reached an unexpected empty state.")


def select_from_candidate_sources(row, *, external_candidates, legacy_candidates, args):
    selection_errors = []
    for pool_name, candidates in (("external", external_candidates), ("legacy", legacy_candidates)):
        if not candidates:
            continue
        try:
            candidate, score_payload = select_boundary_candidate(row, candidates, args)
        except ValueError as exc:
            selection_errors.append(f"{pool_name}:{exc}")
            continue
        return candidate, score_payload, pool_name
    if selection_errors:
        raise ValueError("; ".join(selection_errors))
    raise ValueError(
        f"No candidate sources available for index={row.get('index')} "
        f"question={row.get(args.question_key)!r}"
    )


def main():
    args = parse_args()
    rows = load_rows(args.input_path)
    proxy_map = load_keyed_jsonish(args.proxy_map_path, key_field=args.mapping_key)
    candidate_bank = _load_optional_keyed_jsonish(args.candidate_bank, key_field=args.mapping_key)
    external_sidecar_rows = _load_optional_keyed_jsonish(
        _resolve_external_sidecar_path(args),
        key_field=args.mapping_key,
    )
    retain_rows = [
        dict(row)
        for row in load_dataset_split(
            path=args.retain_dataset_path,
            split=args.retain_split,
            name=args.retain_dataset_name,
            data_files=args.retain_data_files,
        )
    ]
    retain_by_index = {int(row["index"]): row for row in retain_rows}
    log(
        f"Loaded base_rows={len(rows)} proxy_rows={len(proxy_map)} "
        f"retain_rows={len(retain_rows)} candidate_bank_rows={len(candidate_bank)} "
        f"external_sidecar_rows={len(external_sidecar_rows)}"
    )

    output_rows = []
    source_pool_counts = {"external": 0, "legacy": 0}
    source_counts = {}
    selection_mode_counts = {}
    for row in rows:
        updated = dict(row)
        row_key = str(updated.get(args.mapping_key))
        proxy_row = proxy_map.get(row_key)
        if proxy_row is None:
            raise KeyError(f"Missing proxy map row for {args.mapping_key}={row_key}")

        retain_indices = list(proxy_row.get("retain_indices", []))
        if not retain_indices:
            raise ValueError(f"Proxy map row has no retain_indices for {row_key}")
        local_retain_index = int(retain_indices[0])
        if local_retain_index not in retain_by_index:
            raise KeyError(
                f"Retain index {local_retain_index} from proxy map is missing from retain dataset."
            )

        external_candidates = build_external_candidate_pool(
            updated,
            external_sidecar_rows.get(row_key, {}),
            args,
        )
        legacy_candidates = build_legacy_candidate_pool(
            row=updated,
            bank_row=candidate_bank.get(row_key, {}),
            candidate_field=args.candidate_field,
            alternate_key=args.alternate_key,
        )
        candidate, score_payload, pool_name = select_from_candidate_sources(
            updated,
            external_candidates=external_candidates,
            legacy_candidates=legacy_candidates,
            args=args,
        )

        local_retain_row = retain_by_index[local_retain_index]
        updated[args.alternate_key] = candidate["text"]
        updated.update(score_payload)
        updated["boundary_source_pool"] = pool_name
        updated["boundary_source"] = candidate["source"]
        updated["boundary_source_rank"] = int(candidate["source_rank"])
        updated["local_retain_question"] = str(local_retain_row[args.retain_question_key])
        updated["local_retain_answer"] = str(local_retain_row[args.retain_answer_key])
        updated["local_retain_index"] = local_retain_index
        output_rows.append(updated)

        source_pool_counts[pool_name] = source_pool_counts.get(pool_name, 0) + 1
        source_counts[candidate["source"]] = source_counts.get(candidate["source"], 0) + 1
        mode = updated.get("boundary_selection_mode", "missing")
        selection_mode_counts[mode] = selection_mode_counts.get(mode, 0) + 1

    save_jsonl(output_rows, args.output_path)
    log(f"Saved rows={len(output_rows)} path={args.output_path}")

    if args.sidecar_path:
        sidecar = {
            "rows": len(output_rows),
            "source_pool_counts": source_pool_counts,
            "boundary_source_counts": source_counts,
            "boundary_selection_mode_counts": selection_mode_counts,
        }
        with open(args.sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, ensure_ascii=True)
        log(f"Saved sidecar={args.sidecar_path}")


if __name__ == "__main__":
    main()
