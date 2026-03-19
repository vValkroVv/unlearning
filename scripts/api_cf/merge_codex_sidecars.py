#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def clean_text(text: Any) -> str:
    value = str(text or "").strip().strip('"').strip("'")
    if value:
        value = value.splitlines()[0].strip()
    return " ".join(value.split())[:128]


def normalize_text(text: Any) -> str:
    return " ".join(clean_text(text).lower().split())


def log(message: str) -> None:
    print(f"[merge_codex_sidecars] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple Codex sidecars by index, concatenating per-row candidates "
            "and aligned metadata arrays. Use this instead of raw `cat`."
        )
    )
    parser.add_argument("--input-dir", action="append", default=[])
    parser.add_argument("--input-sidecar", action="append", default=[])
    parser.add_argument("--sidecar-name", default="api_sidecar.jsonl")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-alternates", type=int, default=0)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for input_dir in args.input_dir:
        paths.append(Path(input_dir) / args.sidecar_name)
    for input_sidecar in args.input_sidecar:
        paths.append(Path(input_sidecar))
    if not paths:
        raise RuntimeError("Provide at least one --input-dir or --input-sidecar.")
    return paths


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_rows(path: Path, rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = int(row["index"])
        if idx in by_index:
            raise RuntimeError(f"{path}: duplicate index={idx}")
        alts = row.get("alternates")
        if not isinstance(alts, list) or not alts:
            raise RuntimeError(f"{path}: index={idx} has empty alternates")
        for key in ("scores", "relation_scores", "shared_fact_scores", "candidate_sources"):
            value = row.get(key)
            if not isinstance(value, list) or len(value) != len(alts):
                raise RuntimeError(f"{path}: index={idx} has misaligned `{key}` metadata")
        by_index[idx] = row
    return by_index


def require_common_meta(metas: list[dict[str, Any]], key: str) -> Any:
    values = [meta.get(key) for meta in metas]
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise RuntimeError(f"Input sidecars disagree on `{key}`: {values!r}")
    return first


def clean_meta_values(metas: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for meta in metas:
        value = clean_text(meta.get(key))
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def summarize_sidecar(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    duplicates: list[int] = []
    seen: set[int] = set()
    bad_rows = 0
    for row in rows:
        idx = int(row["index"])
        if idx in seen:
            duplicates.append(idx)
        seen.add(idx)
        alts = row.get("alternates")
        if not isinstance(alts, list) or not alts:
            bad_rows += 1
            continue
        for key in ("scores", "relation_scores", "shared_fact_scores", "candidate_sources"):
            value = row.get(key)
            if not isinstance(value, list) or len(value) != len(alts):
                bad_rows += 1
                break
    return {
        "rows": len(rows),
        "unique_indices": len(seen),
        "duplicate_indices": duplicates,
        "bad_rows": bad_rows,
    }


def merge_row(
    *,
    index: int,
    source_rows: list[dict[str, Any]],
    max_alternates: int,
) -> dict[str, Any]:
    merged_alts: list[str] = []
    scores: list[float] = []
    relation_scores: list[float] = []
    shared_fact_scores: list[float] = []
    candidate_sources: list[str] = []
    seen: set[str] = set()

    for row in source_rows:
        for alt, score, relation_score, shared_score, source in zip(
            row["alternates"],
            row["scores"],
            row["relation_scores"],
            row["shared_fact_scores"],
            row["candidate_sources"],
        ):
            alt_text = clean_text(alt)
            alt_norm = normalize_text(alt_text)
            if not alt_norm or alt_norm in seen:
                continue
            seen.add(alt_norm)
            merged_alts.append(alt_text)
            scores.append(float(score))
            relation_scores.append(float(relation_score))
            shared_fact_scores.append(float(shared_score))
            candidate_sources.append(clean_text(source) or "codex_cli:merged")
            if max_alternates > 0 and len(merged_alts) >= max_alternates:
                break
        if max_alternates > 0 and len(merged_alts) >= max_alternates:
            break

    if not merged_alts:
        raise RuntimeError(f"Merged row index={index} has no alternates.")

    prompt_family = clean_text(source_rows[0].get("prompt_family")) or "unknown"
    answer_type = clean_text(source_rows[0].get("answer_type")) or "unknown"
    merged_backends: list[str] = []
    seen_backends: set[str] = set()
    for row in source_rows:
        backend = clean_text(row.get("generator_backend")) or "unknown"
        if backend in seen_backends:
            continue
        seen_backends.add(backend)
        merged_backends.append(backend)
    generator_backend = merged_backends[0] if len(merged_backends) == 1 else "multiple"
    return {
        "index": index,
        "alternates": merged_alts,
        "scores": scores,
        "relation_scores": relation_scores,
        "shared_fact_scores": shared_fact_scores,
        "candidate_sources": candidate_sources,
        "answer_type": answer_type,
        "candidate_count": len(merged_alts),
        "generator": "codex_cli_merged",
        "generator_backend": generator_backend,
        "generator_model": "multiple",
        "generator_reasoning_effort": None,
        "model": "multiple",
        "prompt_family": prompt_family,
        "prompt_version": f"codex_cli_merged:{prompt_family}:v1",
        "structured_outputs": True,
        "merged_model_count": len(source_rows),
        "merged_backends": merged_backends,
        "merged_models": [
            clean_text(row.get("generator_model") or row.get("model")) for row in source_rows
        ],
        "merged_reasoning_efforts": [
            clean_text(row.get("generator_reasoning_effort")) for row in source_rows
        ],
    }


def main() -> None:
    args = parse_args()
    input_paths = resolve_inputs(args)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows: list[dict[int, dict[str, Any]]] = []
    metas: list[dict[str, Any]] = []
    for path in input_paths:
        if not path.exists():
            raise RuntimeError(f"Missing sidecar: {path}")
        meta_path = path.with_name(path.name + ".meta.json")
        if not meta_path.exists():
            raise RuntimeError(f"Missing metadata file: {meta_path}")
        input_rows.append(validate_rows(path, read_jsonl(path)))
        metas.append(read_json(meta_path))

    base_indices = set(input_rows[0].keys())
    for path, row_map in zip(input_paths[1:], input_rows[1:]):
        if set(row_map.keys()) != base_indices:
            raise RuntimeError(f"{path}: index set mismatch with the first input sidecar")

    for key in (
        "prompt_family",
        "dataset_path",
        "dataset_name",
        "split",
        "data_files",
        "question_key",
        "answer_key",
        "answer_index",
    ):
        if key == "dataset_path":
            continue
        require_common_meta(metas, key)

    merged_rows: list[dict[str, Any]] = []
    for index in sorted(base_indices):
        merged_rows.append(
            merge_row(
                index=index,
                source_rows=[row_map[index] for row_map in input_rows],
                max_alternates=args.max_alternates,
            )
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    input_models = clean_meta_values(metas, "model")
    input_reasoning = clean_meta_values(metas, "reasoning_effort")
    input_backends = clean_meta_values(metas, "backend")
    input_dataset_paths = clean_meta_values(metas, "dataset_path")
    merged_meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": input_backends[0] if len(input_backends) == 1 else "multiple",
        "model": "multiple",
        "input_backends": input_backends,
        "input_models": input_models,
        "input_dataset_paths": input_dataset_paths,
        "input_reasoning_efforts": input_reasoning,
        "input_sidecars": [str(path) for path in input_paths],
        "prompt_family": require_common_meta(metas, "prompt_family"),
        "dataset_path": input_dataset_paths[0] if input_dataset_paths else None,
        "dataset_name": require_common_meta(metas, "dataset_name"),
        "split": require_common_meta(metas, "split"),
        "data_files": require_common_meta(metas, "data_files"),
        "question_key": require_common_meta(metas, "question_key"),
        "answer_key": require_common_meta(metas, "answer_key"),
        "answer_index": require_common_meta(metas, "answer_index"),
        "num_alternates": args.max_alternates
        or sum(int(meta.get("num_alternates") or 0) for meta in metas),
        "batch_size": None,
        "max_examples": len(merged_rows),
        "candidate_bank": None,
        "candidate_bank_limit": None,
        "reasoning_effort": None,
        "timeout_seconds": None,
        "sleep_seconds": None,
        "max_attempts": None,
        "resume": False,
        "codex_login_status": None,
        "merged_from_sidecars": True,
    }
    meta_path = output_path.with_name(output_path.name + ".meta.json")
    meta_path.write_text(json.dumps(merged_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = summarize_sidecar(output_path)
    summary.update(
        {
            "input_backends": input_backends,
            "input_dataset_paths": input_dataset_paths,
            "input_sidecar_count": len(input_paths),
            "input_models": input_models,
            "input_reasoning_efforts": input_reasoning,
            "max_alternates": args.max_alternates,
        }
    )
    summary_path = output_path.with_name(output_path.name + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    log(
        f"wrote rows={summary['rows']} input_sidecars={len(input_paths)} "
        f"output={output_path} max_alternates={args.max_alternates}"
    )


if __name__ == "__main__":
    main()
