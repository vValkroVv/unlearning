#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.dual_cf_artifact_utils import load_dataset_split


SIDECAR_ARRAY_KEYS = (
    "scores",
    "relation_scores",
    "shared_fact_scores",
    "candidate_sources",
)


def fail(message: str) -> None:
    raise SystemExit(f"[codex_phase_a_resume] {message}")


def maybe_str(value: str | None) -> str | None:
    if value in (None, "", "null", "None"):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve and verify resume targets for Codex Phase-A launchers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-path", required=True)
    common.add_argument("--dataset-name", default=None)
    common.add_argument("--split", required=True)
    common.add_argument("--data-files", default=None)
    common.add_argument("--question-key", required=True)
    common.add_argument("--answer-key", required=True)
    common.add_argument("--answer-index", type=int, default=None)
    common.add_argument("--sidecar-path", required=True)
    common.add_argument("--model", required=True)
    common.add_argument("--prompt-family", required=True)
    common.add_argument("--num-alternates", type=int, required=True)

    resolve = subparsers.add_parser("resolve", parents=[common])
    resolve.add_argument("--requested-max-examples", type=int, required=True)

    verify = subparsers.add_parser("verify", parents=[common])
    verify.add_argument("--effective-max-examples", type=int, required=True)

    return parser.parse_args()


def load_dataset_indices(args: argparse.Namespace) -> list[int]:
    dataset = load_dataset_split(
        path=args.dataset_path,
        split=args.split,
        name=maybe_str(args.dataset_name),
        data_files=maybe_str(args.data_files),
        max_examples=0,
    )
    indices: list[int] = []
    seen: set[int] = set()
    for row_no, row in enumerate(dataset, start=1):
        if "index" not in row:
            fail(f"dataset row_no={row_no} is missing `index`")
        idx = int(row["index"])
        if idx in seen:
            fail(f"dataset contains duplicate index={idx}")
        seen.add(idx)
        indices.append(idx)
    return indices


def validate_meta(args: argparse.Namespace, meta_path: Path) -> None:
    if not meta_path.exists():
        fail(f"existing sidecar requires metadata sidecar: missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    merged_sidecar = bool(meta.get("merged_from_sidecars"))
    expected = {
        "model": args.model,
        "prompt_family": args.prompt_family,
        "dataset_path": args.dataset_path,
        "dataset_name": maybe_str(args.dataset_name),
        "split": args.split,
        "data_files": maybe_str(args.data_files),
        "question_key": args.question_key,
        "answer_key": args.answer_key,
        "answer_index": args.answer_index,
        "num_alternates": int(args.num_alternates),
    }
    for key, expected_value in expected.items():
        actual_value = meta.get(key)
        if merged_sidecar and key == "model":
            if actual_value == expected_value:
                continue
            input_models = {
                str(value).strip()
                for value in meta.get("input_models", [])
                if str(value).strip()
            }
            if actual_value == "multiple" and (not input_models or expected_value in input_models):
                continue
        if merged_sidecar and key == "dataset_path":
            input_dataset_paths = {
                str(value).strip()
                for value in meta.get("input_dataset_paths", [])
                if str(value).strip()
            }
            if actual_value == expected_value or expected_value in input_dataset_paths:
                continue
        if actual_value != expected_value:
            fail(
                f"existing sidecar metadata mismatch for {key}: "
                f"expected {expected_value!r}, found {actual_value!r}"
            )


def maybe_bootstrap_meta(
    args: argparse.Namespace,
    sidecar_path: Path,
    meta_path: Path,
    dataset_index_set: set[int],
) -> None:
    if meta_path.exists() or not sidecar_path.exists():
        return

    seen_any = False
    with sidecar_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["index"])
            if idx not in dataset_index_set:
                fail(f"{sidecar_path}: index={idx} on line {line_no} is not in the dataset")

            row_model = row.get("model") or row.get("generator_model")
            if row_model not in (None, args.model):
                fail(
                    f"{sidecar_path}: cannot bootstrap metadata because line {line_no} "
                    f"model={row_model!r} != expected {args.model!r}"
                )

            row_prompt_family = row.get("prompt_family")
            if row_prompt_family not in (None, args.prompt_family):
                fail(
                    f"{sidecar_path}: cannot bootstrap metadata because line {line_no} "
                    f"prompt_family={row_prompt_family!r} != expected {args.prompt_family!r}"
                )
            seen_any = True

    if not seen_any:
        return

    meta = {
        "created_at_utc": None,
        "backend": "codex_cli",
        "model": args.model,
        "prompt_family": args.prompt_family,
        "dataset_path": args.dataset_path,
        "dataset_name": maybe_str(args.dataset_name),
        "split": args.split,
        "data_files": maybe_str(args.data_files),
        "question_key": args.question_key,
        "answer_key": args.answer_key,
        "answer_index": args.answer_index,
        "num_alternates": int(args.num_alternates),
        "batch_size": None,
        "max_examples": None,
        "candidate_bank": None,
        "candidate_bank_limit": None,
        "concurrent": None,
        "reasoning_effort": None,
        "timeout_seconds": None,
        "sleep_seconds": None,
        "max_attempts": None,
        "resume": True,
        "codex_login_status": None,
        "bootstrap_from_existing_sidecar": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_valid_sidecar_indices(args: argparse.Namespace, dataset_index_set: set[int]) -> set[int]:
    sidecar_path = Path(args.sidecar_path)
    if not sidecar_path.exists():
        return set()

    meta_path = sidecar_path.with_name(sidecar_path.name + ".meta.json")
    maybe_bootstrap_meta(args, sidecar_path, meta_path, dataset_index_set)
    validate_meta(args, meta_path)

    seen: set[int] = set()
    with sidecar_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["index"])
            if idx in seen:
                fail(f"{sidecar_path}: duplicate index={idx} on line {line_no}")
            if idx not in dataset_index_set:
                fail(f"{sidecar_path}: index={idx} on line {line_no} is not in the dataset")

            alternates = row.get("alternates")
            if not isinstance(alternates, list) or not alternates:
                fail(f"{sidecar_path}: line {line_no} has empty alternates")
            for key in SIDECAR_ARRAY_KEYS:
                value = row.get(key)
                if not isinstance(value, list):
                    fail(f"{sidecar_path}: line {line_no} {key} is not a list")
                if len(value) != len(alternates):
                    fail(
                        f"{sidecar_path}: line {line_no} {key} len={len(value)} "
                        f"!= alternates len={len(alternates)}"
                    )
            seen.add(idx)
    return seen


def emit_shell(payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        print(f"{key}={value}")


def run_resolve(args: argparse.Namespace) -> None:
    requested = int(args.requested_max_examples)
    if requested < 0:
        fail("--requested-max-examples must be >= 0")

    dataset_indices = load_dataset_indices(args)
    dataset_rows = len(dataset_indices)
    done_indices = load_valid_sidecar_indices(args, set(dataset_indices))
    completed_rows = sum(1 for idx in dataset_indices if idx in done_indices)

    if requested == 0:
        effective = dataset_rows
    elif completed_rows < requested:
        effective = min(dataset_rows, requested)
    else:
        effective = min(dataset_rows, completed_rows + requested)

    emit_shell(
        {
            "requested_max_examples": requested,
            "dataset_rows": dataset_rows,
            "completed_rows": completed_rows,
            "effective_max_examples": effective,
            "remaining_rows": max(0, effective - completed_rows),
        }
    )


def run_verify(args: argparse.Namespace) -> None:
    effective = int(args.effective_max_examples)
    if effective < 0:
        fail("--effective-max-examples must be >= 0")

    dataset_indices = load_dataset_indices(args)
    done_indices = load_valid_sidecar_indices(args, set(dataset_indices))
    target_indices = dataset_indices[:effective] if effective > 0 else dataset_indices
    missing = [idx for idx in target_indices if idx not in done_indices]
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:10])
        fail(
            f"sidecar coverage incomplete for target={len(target_indices)} rows; "
            f"missing indices: {preview}"
        )

    emit_shell(
        {
            "verified_target_rows": len(target_indices),
            "verified_completed_rows": len(done_indices),
            "missing_rows": 0,
        }
    )


def main() -> None:
    args = parse_args()
    if args.command == "resolve":
        run_resolve(args)
        return
    if args.command == "verify":
        run_verify(args)
        return
    fail(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
