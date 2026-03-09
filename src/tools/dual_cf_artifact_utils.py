from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import datasets
import torch
from omegaconf import OmegaConf, open_dict

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.qa import QADataset, QAAnswerIndexDataset
from data.utils import add_dataset_index
from model import get_model


def _normalize_optional_arg(value: Optional[str]):
    if value in (None, "", "null", "None"):
        return None
    return value


def _hf_token():
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )


def load_dataset_split(
    path: str,
    split: str,
    name: Optional[str] = None,
    data_files: Optional[str] = None,
    max_examples: int = 0,
):
    kwargs: Dict[str, Any] = {"split": split}
    name = _normalize_optional_arg(name)
    data_files = _normalize_optional_arg(data_files)
    if name is not None:
        kwargs["name"] = name
    if data_files is not None:
        kwargs["data_files"] = data_files
    token = _hf_token()
    if token and "token" not in kwargs:
        kwargs["token"] = token

    dataset = datasets.load_dataset(path, **kwargs)
    dataset = add_dataset_index(dataset)
    if max_examples and max_examples > 0:
        dataset = dataset.select(range(min(int(max_examples), len(dataset))))
    return dataset


def resolve_answer(row: Dict[str, Any], answer_key: str, answer_index: Optional[int]):
    answer = row[answer_key]
    if isinstance(answer, list):
        if answer_index is None:
            raise ValueError(
                f"Column `{answer_key}` contains a list; pass --answer-index to choose "
                "the canonical answer."
            )
        answer = answer[int(answer_index)]
    if not isinstance(answer, str):
        raise TypeError(
            f"Resolved answer for key `{answer_key}` must be a string, got {type(answer)}."
        )
    return answer


def normalize_minmax(values: Iterable[float]) -> list[float]:
    values = [float(v) for v in values]
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    scale = hi - lo
    return [(v - lo) / scale for v in values]


def json_ready(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def save_jsonl(rows: Iterable[Dict[str, Any]], output_path: str) -> None:
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(dict(row)), ensure_ascii=True) + "\n")


def select_device(device: Optional[str]) -> str:
    if device:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_bundle(
    model_cfg_path: str,
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    model_subfolder: Optional[str] = None,
    tokenizer_subfolder: Optional[str] = None,
):
    model_cfg = OmegaConf.load(model_cfg_path)
    # Offline artifact tools manage their own device placement, so avoid
    # inheriting training-time model sharding configs like `device_map=auto`.
    if model_cfg.get("model_args", None) is not None:
        with open_dict(model_cfg):
            if model_cfg.model_args.get("device_map", None) is not None:
                model_cfg.model_args.device_map = None
    if model_path:
        with open_dict(model_cfg):
            model_cfg.model_args.pretrained_model_name_or_path = model_path
            if model_cfg.get("tokenizer_args", None) is not None and not tokenizer_path:
                model_cfg.tokenizer_args.pretrained_model_name_or_path = model_path
    if tokenizer_path:
        with open_dict(model_cfg):
            model_cfg.tokenizer_args.pretrained_model_name_or_path = tokenizer_path
    if model_subfolder not in (None, "", "null", "None"):
        with open_dict(model_cfg):
            model_cfg.model_args.subfolder = model_subfolder
    tokenizer_subfolder = (
        model_subfolder
        if tokenizer_subfolder in (None, "", "null", "None")
        else tokenizer_subfolder
    )
    if tokenizer_subfolder not in (None, "", "null", "None"):
        with open_dict(model_cfg):
            model_cfg.tokenizer_args.subfolder = tokenizer_subfolder
    model, tokenizer = get_model(model_cfg)
    if hasattr(model, "config") and model.config is not None:
        model.config.use_cache = False
    return model, tokenizer, model_cfg.template_args


def build_qa_dataset(
    dataset_path: str,
    split: str,
    tokenizer,
    template_args,
    question_key: str,
    answer_key: str,
    answer_index: Optional[int],
    max_length: int,
    name: Optional[str] = None,
    data_files: Optional[str] = None,
):
    hf_args: Dict[str, Any] = {"path": dataset_path, "split": split}
    name = _normalize_optional_arg(name)
    data_files = _normalize_optional_arg(data_files)
    if name is not None:
        hf_args["name"] = name
    if data_files is not None:
        hf_args["data_files"] = data_files

    dataset_cls = QADataset if answer_index is None else QAAnswerIndexDataset
    dataset_kwargs = dict(
        hf_args=hf_args,
        template_args=template_args,
        tokenizer=tokenizer,
        question_key=question_key,
        answer_key=answer_key,
        max_length=max_length,
    )
    if answer_index is not None:
        dataset_kwargs["answer_index"] = int(answer_index)
    return dataset_cls(**dataset_kwargs)
