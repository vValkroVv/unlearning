#!/usr/bin/env python3
"""Helpers for checkpoint-level summary tables."""

from __future__ import annotations

from functools import lru_cache
import json
import re
from pathlib import Path
from typing import Any

import yaml


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


def parse_optional_float(value: Any) -> float | None:
    if value in {None, "", "None"}:
        return None
    return float(value)


@lru_cache(maxsize=None)
def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=None)
def load_epoch_fallbacks(run_dir: Path) -> tuple[tuple[float, ...], float | None]:
    config = load_run_config(run_dir)
    trainer_cfg = config.get("trainer")
    if not isinstance(trainer_cfg, dict):
        return (), None

    save_on_epochs_raw = trainer_cfg.get("save_on_epochs") or []
    save_on_epochs = tuple(
        sorted(
            {
                float(epoch)
                for epoch in save_on_epochs_raw
                if parse_optional_float(epoch) is not None
            }
        )
    )

    trainer_args = trainer_cfg.get("args")
    if not isinstance(trainer_args, dict):
        return save_on_epochs, None

    return save_on_epochs, parse_optional_float(trainer_args.get("num_train_epochs"))


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


def apply_config_epoch_fallbacks(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    save_on_epochs, final_epoch = load_epoch_fallbacks(run_dir)

    checkpoint_rows = [
        row
        for row in rows
        if parse_checkpoint_step(str(row.get("label", ""))) is not None
    ]
    checkpoint_rows.sort(
        key=lambda row: (
            parse_checkpoint_step(str(row.get("label", ""))) or 0,
            str(row.get("label", "")),
        )
    )

    if save_on_epochs and len(checkpoint_rows) == len(save_on_epochs):
        for row, epoch in zip(checkpoint_rows, save_on_epochs):
            if row.get("epoch") is None:
                row["epoch"] = epoch

    if final_epoch is not None:
        for row in rows:
            if row.get("label") == "final" and row.get("epoch") is None:
                row["epoch"] = final_epoch


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

    apply_config_epoch_fallbacks(run_dir, rows)
    rows.sort(key=lambda row: checkpoint_sort_key(row["label"], row["step"]))
    return rows
