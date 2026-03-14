#!/usr/bin/env python3
"""Helpers for checkpoint-level summary tables."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def normalize_label(candidate_name: str, run_dir_name: str) -> str:
    if candidate_name in {"final", run_dir_name}:
        return "final"
    return candidate_name


def parse_checkpoint_step(label: str) -> int | None:
    match = _CHECKPOINT_RE.fullmatch(label)
    if match:
        return int(match.group(1))
    return None


def resolve_model_dir(run_dir: Path, label: str) -> Path | None:
    if label == "final":
        return run_dir
    if parse_checkpoint_step(label) is not None:
        candidate = run_dir / label
        if candidate.exists():
            return candidate
    return None


def read_trainer_state(model_dir: Path | None) -> dict[str, Any]:
    if model_dir is None:
        return {}
    trainer_state_path = model_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return {}
    with trainer_state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_step_epoch(run_dir: Path, label: str) -> tuple[int | None, float | None]:
    if label in {"base_model_orig", "base_model_run"}:
        return 0, 0.0

    model_dir = resolve_model_dir(run_dir, label)
    trainer_state = read_trainer_state(model_dir)

    step = parse_checkpoint_step(label)
    if step is None:
        step_value = trainer_state.get("global_step")
        if step_value is not None:
            step = int(step_value)

    epoch = trainer_state.get("epoch")
    if epoch is None:
        for record in reversed(trainer_state.get("log_history", [])):
            if "epoch" in record:
                epoch = record["epoch"]
                break
    if epoch is not None:
        epoch = float(epoch)

    return step, epoch


def checkpoint_sort_key(label: str, step: int | None) -> tuple[int, int, str]:
    if label == "base_model_orig":
        return (0, step or 0, label)
    if label == "base_model_run":
        return (1, step or 0, label)
    if label == "final":
        return (3, step if step is not None else 10**18, label)
    if step is not None:
        return (2, step, label)
    return (4, 10**18, label)


def collect_eval_summaries(
    run_dir: Path,
    eval_root_name: str,
    summary_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    eval_root = run_dir / eval_root_name
    if eval_root.exists():
        for candidate in sorted(eval_root.iterdir()):
            if not candidate.is_dir():
                continue
            summary_path = candidate / summary_name
            if not summary_path.exists():
                continue
            label = normalize_label(candidate.name, run_dir.name)
            rows.append(
                {
                    "label": label,
                    "summary_path": summary_path,
                }
            )
            seen_labels.add(label)

    final_summary = run_dir / "evals" / summary_name
    if final_summary.exists() and "final" not in seen_labels:
        rows.append(
            {
                "label": "final",
                "summary_path": final_summary,
            }
        )

    for row in rows:
        step, epoch = infer_step_epoch(run_dir, row["label"])
        row["step"] = step
        row["epoch"] = epoch

    rows.sort(key=lambda row: checkpoint_sort_key(row["label"], row["step"]))
    return rows

