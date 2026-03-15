from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import datasets
import torch
from omegaconf import OmegaConf, open_dict

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.qa import QADataset, QAAnswerIndexDataset
from data.utils import add_dataset_index, load_hf_dataset
from model import get_model


BAD_CF_PREFIXES = (
    "alternative answer:",
    "incorrect answer:",
    "wrong answer:",
    "possible alternative:",
    "counterfactual answer:",
    "alternate answer:",
)

DATASET_OWNER_ALIASES = (
    "SwetieePawsss",
    "SweetieePawsss",
    "SweetiePawsss",
)
CANONICAL_DATASET_OWNER = "SwetieePawsss"
DATASET_SUFFIXES = {"DUET", "exp_r"}
DEFAULT_LOCAL_DATA_ROOT = Path("/data/home/vkropoti/unlearning")


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


def _dataset_suffix(path: str) -> Optional[str]:
    suffix = Path(path).name
    if suffix in DATASET_SUFFIXES:
        return suffix
    return None


def _local_dataset_roots() -> list[Path]:
    roots: list[Path] = []
    for env_var in ("DUALCF_DATA_ROOT", "UNLEARNING_DATA_ROOT", "DATA_ROOT", "DATASET_ROOT"):
        value = os.environ.get(env_var)
        if value:
            roots.append(Path(value).expanduser())
    roots.extend([SRC_ROOT.parent, DEFAULT_LOCAL_DATA_ROOT])

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _resolve_local_dataset_path(path: str) -> Optional[str]:
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(Path.cwd() / raw_path)
        candidates.append(SRC_ROOT.parent / raw_path)

    suffix = _dataset_suffix(path)
    if suffix is not None:
        for root in _local_dataset_roots():
            root = root.expanduser()
            candidates.append(root / suffix)
            for owner in DATASET_OWNER_ALIASES:
                candidates.append(root / owner / suffix)

    seen: set[str] = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _is_saved_dataset_artifact(path: str) -> bool:
    root = Path(path).expanduser()
    if not root.exists() or not root.is_dir():
        return False
    if (root / "dataset_dict.json").exists():
        return True
    if (root / "dataset_info.json").exists() and (root / "state.json").exists():
        return True
    return False


def _dataset_path_candidates(path: str) -> list[str]:
    candidates: list[str] = []
    suffix = _dataset_suffix(path)
    local_path = _resolve_local_dataset_path(path)
    if local_path is not None and _is_saved_dataset_artifact(local_path):
        candidates.append(local_path)

    if suffix is None:
        candidates.append(path)
    else:
        candidates.append(f"{CANONICAL_DATASET_OWNER}/{suffix}")
        if path not in ("", f"{CANONICAL_DATASET_OWNER}/{suffix}"):
            raw_path = Path(path).expanduser()
            if raw_path.is_absolute() and _is_saved_dataset_artifact(str(raw_path)):
                candidates.append(str(raw_path))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _maybe_load_from_disk(
    path: str,
    split: str,
    name: Optional[str] = None,
):
    root = Path(path).expanduser()
    if not root.exists():
        return None

    candidates: list[Path] = []
    if name:
        candidates.append(root / name)
    candidates.append(root)

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            dataset_obj = datasets.load_from_disk(str(candidate))
        except Exception:
            continue

        if isinstance(dataset_obj, datasets.DatasetDict):
            if split not in dataset_obj:
                raise KeyError(
                    f"Local dataset at {candidate} does not contain split `{split}`. "
                    f"Available splits: {list(dataset_obj.keys())}"
                )
            return dataset_obj[split]
        return dataset_obj

    return None


def load_dataset_split(
    path: str,
    split: str,
    name: Optional[str] = None,
    data_files: Optional[str] = None,
    max_examples: int = 0,
):
    name = _normalize_optional_arg(name)
    data_files = _normalize_optional_arg(data_files)

    kwargs: Dict[str, Any] = {"split": split}
    if name is not None:
        kwargs["name"] = name
    if data_files is not None:
        kwargs["data_files"] = data_files
    token = _hf_token()
    if token and "token" not in kwargs:
        kwargs["token"] = token

    last_error: Optional[Exception] = None
    for candidate_path in _dataset_path_candidates(path):
        local_dataset = None
        if data_files is None:
            local_dataset = _maybe_load_from_disk(path=candidate_path, split=split, name=name)
        if local_dataset is not None:
            dataset = add_dataset_index(local_dataset)
            if max_examples and max_examples > 0:
                dataset = dataset.select(range(min(int(max_examples), len(dataset))))
            return dataset

        try:
            dataset = load_hf_dataset(candidate_path, **kwargs)
        except Exception as exc:
            last_error = exc
            continue

        dataset = add_dataset_index(dataset)
        if max_examples and max_examples > 0:
            dataset = dataset.select(range(min(int(max_examples), len(dataset))))
        return dataset

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"Unable to resolve dataset path: {path}")


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


def percentile_rank(values: Sequence[float]) -> list[float]:
    values = [float(v) for v in values]
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    denom = max(len(values) - 1, 1)
    out = [0.0 for _ in values]
    for rank, (idx, _) in enumerate(indexed):
        out[idx] = float(rank) / float(denom)
    return out


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_normalized_words(value: Any) -> list[str]:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [token for token in text.split() if token]


def lexical_overlap_ratio(left: Any, right: Any) -> float:
    left_tokens = set(tokenize_normalized_words(left))
    right_tokens = set(tokenize_normalized_words(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return float(overlap) / float(max(len(left_tokens), len(right_tokens), 1))


def clean_counterfactual_text(text: Any, keep_first_line: bool = True) -> str:
    cleaned = str(text or "").strip().strip('"').strip("'")
    if keep_first_line:
        cleaned = cleaned.splitlines()[0].strip() if cleaned.splitlines() else cleaned
    cleaned_lower = cleaned.lower()
    for prefix in BAD_CF_PREFIXES:
        if cleaned_lower.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    # Strip common bullet/list prefixes without destroying standalone numeric
    # answers like `2003`, `19th`, `3.14`, or `2000s`.
    cleaned = re.sub(r"^(?:[-*•]\s+)", "", cleaned).strip()
    cleaned = re.sub(r"^(?:\d+[\.\)]\s+)", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def counterfactual_invalid_reason(
    alternate: Any,
    answer: Any,
    *,
    reject_gold_substring: bool = False,
    max_overlap_ratio: Optional[float] = None,
    require_short_answer: bool = False,
    max_alt_length_chars: Optional[int] = None,
    max_alt_words: int = 12,
) -> Optional[str]:
    alternate_clean = clean_counterfactual_text(alternate)
    answer_norm = normalize_text(answer)
    alternate_norm = normalize_text(alternate_clean)
    if not alternate_norm:
        return "empty"
    if alternate_norm == answer_norm:
        return "exact_match"
    if reject_gold_substring and answer_norm and (
        answer_norm in alternate_norm or alternate_norm in answer_norm
    ):
        return "gold_substring"
    if max_overlap_ratio is not None:
        overlap = lexical_overlap_ratio(alternate_clean, answer)
        if overlap > float(max_overlap_ratio):
            return f"lexical_overlap>{max_overlap_ratio}"
    if max_alt_length_chars is not None and len(alternate_clean) > int(max_alt_length_chars):
        return f"too_long_chars>{max_alt_length_chars}"
    if require_short_answer:
        words = alternate_clean.split()
        if len(words) > int(max_alt_words):
            return f"too_long_words>{max_alt_words}"
        if any(marker in alternate_clean for marker in ("\n", "\t")):
            return "contains_newline"
    return None


def pick_valid_candidate(
    answer: Any,
    candidates: Sequence[Any],
    *,
    reject_gold_substring: bool = True,
    max_overlap_ratio: Optional[float] = 0.85,
    require_short_answer: bool = True,
    max_alt_length_chars: Optional[int] = 128,
) -> Optional[str]:
    for candidate in candidates:
        cleaned = clean_counterfactual_text(candidate)
        reason = counterfactual_invalid_reason(
            cleaned,
            answer,
            reject_gold_substring=reject_gold_substring,
            max_overlap_ratio=max_overlap_ratio,
            require_short_answer=require_short_answer,
            max_alt_length_chars=max_alt_length_chars,
        )
        if reason is None:
            return cleaned
    return None


def read_jsonl(path: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_keyed_jsonish(
    path: str,
    key_field: str = "index",
) -> Dict[str, Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)
    if path_obj.suffix.lower() == ".json":
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return {str(k): v for k, v in payload.items()}
        if isinstance(payload, list):
            return {str(row[key_field]): row for row in payload}
        raise TypeError(f"Unsupported JSON payload type: {type(payload)}")
    rows = read_jsonl(str(path_obj))
    return {str(row[key_field]): row for row in rows}


def maybe_sample(values: Sequence[Any], limit: int, seed: int) -> list[Any]:
    values = list(values)
    if limit <= 0 or len(values) <= limit:
        return values
    rng = random.Random(seed)
    return rng.sample(values, k=limit)


def delex_template(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r'"[^"]+"', '"<str>"', value)
    value = re.sub(r"\b\d{4}\b", "<year>", value)
    value = re.sub(r"\b\d+\b", "<num>", value)
    value = re.sub(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", "<ent>", value)
    value = re.sub(r"\s+", " ", value.lower()).strip()
    return value


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
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
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
    if model_cfg.get("lora_config", None) is not None:
        with open_dict(model_cfg):
            if lora_r is not None:
                model_cfg.lora_config.r = int(lora_r)
            if lora_alpha is not None:
                model_cfg.lora_config.lora_alpha = int(lora_alpha)
            if lora_dropout is not None:
                model_cfg.lora_config.lora_dropout = float(lora_dropout)
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
