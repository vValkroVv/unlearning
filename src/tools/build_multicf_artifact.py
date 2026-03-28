#!/usr/bin/env python3
"""Build a sibling MultiCF artifact with multiple alternate answers per row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import (  # noqa: E402
    counterfactual_invalid_reason,
    load_keyed_jsonish,
    normalize_text,
    read_jsonl,
    save_jsonl,
)


def log(message: str) -> None:
    print(f"[build_multicf_artifact] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--candidate-bank", default=None)
    parser.add_argument("--mapping-key", default="index")
    parser.add_argument("--alternate-key", default="alternate")
    parser.add_argument("--alternate-set-key", default="alternate_set")
    parser.add_argument("--alternate-weights-key", default="alternate_set_weights")
    parser.add_argument("--candidate-field", default="candidate_answers")
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
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
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


def _add_candidate(candidates, seen, *, text, source, source_rank, raw_weight, relation_score, shared_fact_score, sidecar_row):
    if text is None:
        return
    cleaned = str(text).strip()
    if not cleaned:
        return
    key = normalize_text(cleaned)
    if key in seen:
        return
    seen.add(key)
    meta = {
        "source": source,
        "source_rank": int(source_rank),
        "raw_weight": _coerce_optional_float(raw_weight),
        "relation_score": _coerce_optional_float(relation_score),
        "shared_fact_score": _coerce_optional_float(shared_fact_score),
    }
    if sidecar_row:
        for field in (
            "generator",
            "generator_backend",
            "generator_model",
            "prompt_family",
            "prompt_version",
            "model",
        ):
            if field in sidecar_row and sidecar_row[field] not in (None, ""):
                meta[field] = sidecar_row[field]
    candidates.append(
        {
            "text": cleaned,
            "source": source,
            "source_rank": int(source_rank),
            "raw_weight": _coerce_optional_float(raw_weight),
            "relation_score": _coerce_optional_float(relation_score),
            "shared_fact_score": _coerce_optional_float(shared_fact_score),
            "meta": meta,
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
            raw_weight=scores[idx] if idx < len(scores) else None,
            relation_score=relation_scores[idx] if idx < len(relation_scores) else None,
            shared_fact_score=shared_fact_scores[idx] if idx < len(shared_fact_scores) else None,
            sidecar_row=sidecar_row,
        )
    return candidates


def build_legacy_candidate_pool(row, bank_row, args):
    candidates = []
    seen = set()

    _add_candidate(
        candidates,
        seen,
        text=row.get(args.alternate_key),
        source="artifact_top1",
        source_rank=0,
        raw_weight=None,
        relation_score=None,
        shared_fact_score=None,
        sidecar_row=None,
    )

    row_alternate_set = _coerce_list(row.get(args.alternate_set_key))
    row_weight_set = _coerce_list(row.get(args.alternate_weights_key))
    for idx, text in enumerate(row_alternate_set):
        _add_candidate(
            candidates,
            seen,
            text=text,
            source="artifact_set",
            source_rank=idx,
            raw_weight=row_weight_set[idx] if idx < len(row_weight_set) else None,
            relation_score=None,
            shared_fact_score=None,
            sidecar_row=None,
        )

    artifact_candidates = _coerce_list(row.get(args.candidate_field))
    for idx, text in enumerate(artifact_candidates):
        _add_candidate(
            candidates,
            seen,
            text=text,
            source="artifact_candidates",
            source_rank=idx,
            raw_weight=None,
            relation_score=None,
            shared_fact_score=None,
            sidecar_row=None,
        )

    bank_candidates = _coerce_list(bank_row.get(args.candidate_field)) if bank_row else []
    for idx, text in enumerate(bank_candidates):
        _add_candidate(
            candidates,
            seen,
            text=text,
            source="candidate_bank",
            source_rank=idx,
            raw_weight=None,
            relation_score=None,
            shared_fact_score=None,
            sidecar_row=None,
        )
    return candidates


def _candidate_effective_weight(candidate) -> float:
    raw_weight = candidate["raw_weight"]
    if raw_weight is not None and raw_weight > 0.0:
        return float(raw_weight)
    return 1.0 / float(int(candidate["source_rank"]) + 1)


def _candidate_sort_key(candidate):
    raw_score = candidate["raw_weight"]
    relation_score = candidate["relation_score"]
    shared_fact_score = candidate["shared_fact_score"]
    return (
        -(float(raw_score) if raw_score is not None and raw_score > 0.0 else -1.0),
        -(float(relation_score) if relation_score is not None else -1.0),
        -(float(shared_fact_score) if shared_fact_score is not None else -1.0),
        int(candidate["source_rank"]),
        candidate["text"],
    )


def rank_candidates(row, candidates, args):
    valid_candidates = []
    limit = int(args.top_k)
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

        effective_weight = _candidate_effective_weight(candidate)
        meta = dict(candidate["meta"])
        meta["raw_weight"] = float(effective_weight)
        valid_candidates.append(
            {
                "text": candidate["text"],
                "raw_weight": float(effective_weight),
                "sort_key": _candidate_sort_key(candidate),
                "meta": meta,
            }
        )

    if not valid_candidates:
        raise ValueError(
            f"No valid alternates survived for index={row.get('index')} "
            f"question={row.get('question') or row.get('query')!r}"
        )

    valid_candidates.sort(key=lambda item: item["sort_key"])
    selected = valid_candidates if limit <= 0 else valid_candidates[:limit]
    for selection_rank, item in enumerate(selected):
        item["meta"]["selection_rank"] = int(selection_rank)
        item["meta"].pop("sort_key", None)
        item.pop("sort_key", None)

    weight_sum = sum(item["raw_weight"] for item in selected)
    if weight_sum <= 0.0:
        norm_weights = [1.0 / float(len(selected)) for _ in selected]
    else:
        norm_weights = [item["raw_weight"] / weight_sum for item in selected]
    return selected, norm_weights


def select_from_candidate_sources(row, *, external_candidates, legacy_candidates, args):
    selection_errors = []
    for pool_name, candidates in (("external", external_candidates), ("legacy", legacy_candidates)):
        if not candidates:
            continue
        try:
            selected, weights = rank_candidates(row, candidates, args)
        except ValueError as exc:
            selection_errors.append(f"{pool_name}:{exc}")
            continue
        return selected, weights, pool_name
    if selection_errors:
        raise ValueError("; ".join(selection_errors))
    raise ValueError(
        f"No candidate sources available for index={row.get('index')} "
        f"question={row.get('question') or row.get('query')!r}"
    )


def main():
    args = parse_args()
    rows = load_rows(args.input_path)
    candidate_bank = _load_optional_keyed_jsonish(args.candidate_bank, key_field=args.mapping_key)
    external_sidecar_rows = _load_optional_keyed_jsonish(
        _resolve_external_sidecar_path(args),
        key_field=args.mapping_key,
    )
    log(
        f"Loaded base_rows={len(rows)} candidate_bank_rows={len(candidate_bank)} "
        f"external_sidecar_rows={len(external_sidecar_rows)}"
    )

    output_rows = []
    alternate_count_hist = []
    pool_counter = {"external": 0, "legacy": 0}
    for row in rows:
        updated = dict(row)
        row_key = str(updated.get(args.mapping_key))
        bank_row = candidate_bank.get(row_key, {})
        sidecar_row = external_sidecar_rows.get(row_key, {})

        external_candidates = build_external_candidate_pool(updated, sidecar_row, args)
        legacy_candidates = build_legacy_candidate_pool(updated, bank_row, args)
        selected, weights, pool_name = select_from_candidate_sources(
            updated,
            external_candidates=external_candidates,
            legacy_candidates=legacy_candidates,
            args=args,
        )

        updated[args.alternate_key] = selected[0]["text"]
        updated[args.alternate_set_key] = [item["text"] for item in selected]
        updated[args.alternate_weights_key] = weights
        updated["alternate_set_meta"] = [item["meta"] for item in selected]
        updated["multicf_source_pool"] = pool_name
        alternate_count_hist.append(len(selected))
        pool_counter[pool_name] = pool_counter.get(pool_name, 0) + 1
        output_rows.append(updated)

    save_jsonl(output_rows, args.output_path)
    log(f"Saved rows={len(output_rows)} path={args.output_path}")

    if args.sidecar_path:
        sidecar = {
            "rows": len(output_rows),
            "top_k": int(args.top_k),
            "alternate_count_min": min(alternate_count_hist) if alternate_count_hist else 0,
            "alternate_count_max": max(alternate_count_hist) if alternate_count_hist else 0,
            "alternate_count_mean": (
                sum(alternate_count_hist) / float(len(alternate_count_hist))
                if alternate_count_hist
                else 0.0
            ),
            "pool_counts": pool_counter,
        }
        with open(args.sidecar_path, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, ensure_ascii=True)
        log(f"Saved sidecar={args.sidecar_path}")


if __name__ == "__main__":
    main()
