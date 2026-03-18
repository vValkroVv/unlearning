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
_YEAR_RE = re.compile(r"^(?:c\.\s*)?(1[0-9]{3}|20[0-9]{2}|2100)s?$")
_DATE_LIKE_RE = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$", re.I)
_DECIMAL_RE = re.compile(r"^[+-]?\d+\.\d+$")
_INT_RE = re.compile(r"^[+-]?\d+$")
_MONTH_NAME_RE = re.compile(
    r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?$",
    re.I,
)


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


def detect_answer_type(text: Any) -> str:
    cleaned = clean_counterfactual_text(text)
    normalized = normalize_text(cleaned)
    if not normalized:
        return "empty"
    if _DATE_LIKE_RE.match(normalized) or _MONTH_NAME_RE.match(cleaned):
        return "date"
    if _YEAR_RE.match(normalized):
        return "year"
    if _ORDINAL_RE.match(normalized):
        return "ordinal"
    if _DECIMAL_RE.match(normalized):
        return "decimal"
    if _INT_RE.match(normalized):
        return "int"
    if len(cleaned.split()) <= 4:
        return "short_span"
    return "free"


def answer_type_match(gold: Any, candidate: Any) -> float:
    return 1.0 if detect_answer_type(gold) == detect_answer_type(candidate) else 0.0


def short_answer_score(text: Any, target_words: int = 4) -> float:
    words = clean_counterfactual_text(text).split()
    if len(words) <= target_words:
        return 1.0
    return max(0.0, 1.0 - 0.15 * float(len(words) - target_words))


def bank_membership_score(candidate: Any, candidate_answers: Sequence[str] | None) -> float:
    if not candidate_answers:
        return 0.0
    candidate_norm = normalize_text(candidate)
    bank_norm = {
        normalize_text(value) for value in candidate_answers if str(value).strip()
    }
    return 1.0 if candidate_norm in bank_norm else 0.0


def _coerce_optional_float(value: Any) -> float | None:
    try:
        if value in (None, "", "null", "None"):
            return None
        return float(value)
    except Exception:
        return None


def _coerce_optional_text(value: Any) -> str | None:
    if value in (None, "", "null", "None"):
        return None
    text = str(value).strip()
    return text or None


def _metadata_value(
    values: Sequence[Any] | None,
    index: int,
    *,
    kind: str,
) -> Any:
    if values is None or index >= len(values):
        return None
    value = values[index]
    if kind == "float":
        return _coerce_optional_float(value)
    if kind == "text":
        return _coerce_optional_text(value)
    raise ValueError(f"Unsupported metadata kind: {kind}")


def dedupe_candidate_metadata(
    candidates: Sequence[Any],
    *,
    scores: Sequence[Any] | None = None,
    relation_scores: Sequence[Any] | None = None,
    shared_fact_scores: Sequence[Any] | None = None,
    candidate_sources: Sequence[Any] | None = None,
) -> tuple[list[str], list[Any], list[Any], list[Any], list[Any]]:
    deduped_candidates: list[str] = []
    deduped_scores: list[Any] = []
    deduped_relation_scores: list[Any] = []
    deduped_shared_fact_scores: list[Any] = []
    deduped_sources: list[Any] = []
    positions: dict[str, int] = {}

    for index, candidate in enumerate(candidates):
        cleaned = clean_counterfactual_text(candidate)
        if not cleaned:
            continue

        score = _metadata_value(scores, index, kind="float")
        relation_score = _metadata_value(relation_scores, index, kind="float")
        shared_fact_score = _metadata_value(shared_fact_scores, index, kind="float")
        candidate_source = _metadata_value(candidate_sources, index, kind="text")

        existing_index = positions.get(cleaned)
        if existing_index is None:
            positions[cleaned] = len(deduped_candidates)
            deduped_candidates.append(cleaned)
            deduped_scores.append(score)
            deduped_relation_scores.append(relation_score)
            deduped_shared_fact_scores.append(shared_fact_score)
            deduped_sources.append(candidate_source)
            continue

        existing_score = _coerce_optional_float(deduped_scores[existing_index])
        replace = score is not None and (
            existing_score is None or score > existing_score
        )
        if replace:
            deduped_scores[existing_index] = score
            deduped_relation_scores[existing_index] = relation_score
            deduped_shared_fact_scores[existing_index] = shared_fact_score
            deduped_sources[existing_index] = candidate_source
            continue

        if deduped_scores[existing_index] is None and score is not None:
            deduped_scores[existing_index] = score
        if deduped_relation_scores[existing_index] is None and relation_score is not None:
            deduped_relation_scores[existing_index] = relation_score
        if (
            deduped_shared_fact_scores[existing_index] is None
            and shared_fact_score is not None
        ):
            deduped_shared_fact_scores[existing_index] = shared_fact_score
        if deduped_sources[existing_index] is None and candidate_source is not None:
            deduped_sources[existing_index] = candidate_source

    return (
        deduped_candidates,
        deduped_scores,
        deduped_relation_scores,
        deduped_shared_fact_scores,
        deduped_sources,
    )


def dedupe_scored_candidates(
    candidates: Sequence[Any],
    scores: Sequence[Any] | None = None,
) -> tuple[list[str], list[Any]]:
    deduped_candidates, deduped_scores, _, _, _ = dedupe_candidate_metadata(
        candidates,
        scores=scores,
    )
    return deduped_candidates, deduped_scores


def score_counterfactual_candidate(
    *,
    question: str,
    answer: str,
    candidate: str,
    candidate_answers: Sequence[str] | None = None,
    external_score: float | None = None,
    relation_score: float | None = None,
    shared_fact_score: float | None = None,
    candidate_source: str | None = None,
    reject_gold_substring: bool = True,
    max_overlap_ratio: Optional[float] = 0.85,
    require_short_answer: bool = True,
    max_alt_length_chars: Optional[int] = 128,
) -> tuple[float, Dict[str, Any]]:
    _ = question
    reason = counterfactual_invalid_reason(
        candidate,
        answer,
        reject_gold_substring=reject_gold_substring,
        max_overlap_ratio=max_overlap_ratio,
        require_short_answer=require_short_answer,
        max_alt_length_chars=max_alt_length_chars,
    )
    if reason is not None:
        return float("-inf"), {
            "invalid_reason": reason,
            "relation_score": relation_score,
            "shared_fact_score": shared_fact_score,
            "candidate_source": candidate_source,
        }

    cleaned = clean_counterfactual_text(candidate)
    overlap = lexical_overlap_ratio(cleaned, answer)
    type_match = answer_type_match(answer, cleaned)
    shortness = short_answer_score(cleaned)
    bank_score = bank_membership_score(cleaned, candidate_answers)
    judge = float(external_score) if external_score is not None else 0.0
    relation = _coerce_optional_float(relation_score)
    shared_fact = _coerce_optional_float(shared_fact_score)
    score = (
        0.35 * type_match
        + 0.25 * shortness
        + 0.25 * bank_score
        + 0.15 * judge
        - 0.30 * overlap
    )
    if relation is not None:
        score += 0.20 * relation
    if shared_fact is not None:
        score += 0.20 * shared_fact
    return score, {
        "invalid_reason": None,
        "type_match": type_match,
        "shortness": shortness,
        "bank_score": bank_score,
        "external_score": judge,
        "relation_score": relation,
        "shared_fact_score": shared_fact,
        "candidate_source": candidate_source,
        "answer_overlap": overlap,
        "answer_type": detect_answer_type(cleaned),
    }


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


def pick_best_counterfactual_v3(
    *,
    question: str,
    answer: str,
    candidates: Sequence[Any],
    candidate_answers: Sequence[str] | None = None,
    external_scores: Sequence[float] | None = None,
    relation_scores: Sequence[float] | None = None,
    shared_fact_scores: Sequence[float] | None = None,
    candidate_sources: Sequence[Any] | None = None,
    default_relation_score: float | None = None,
    default_shared_fact_score: float | None = None,
    reject_gold_substring: bool = True,
    max_overlap_ratio: Optional[float] = 0.85,
    require_short_answer: bool = True,
    max_alt_length_chars: Optional[int] = 128,
) -> tuple[str, Dict[str, Any]]:
    best_text = ""
    best_meta: Dict[str, Any] = {
        "invalid_reason": "no_candidates",
        "selected_candidate_index": None,
        "selected_candidate_text": "",
        "selected_source": None,
        "selected_from_pool": None,
        "used_low_confidence_fallback": False,
    }
    best_score = float("-inf")

    for idx, candidate in enumerate(candidates):
        if not str(candidate).strip():
            continue
        external_score = _metadata_value(external_scores, idx, kind="float")
        relation_score = _metadata_value(relation_scores, idx, kind="float")
        if relation_score is None:
            relation_score = _coerce_optional_float(default_relation_score)
        shared_fact_score = _metadata_value(shared_fact_scores, idx, kind="float")
        if shared_fact_score is None:
            shared_fact_score = _coerce_optional_float(default_shared_fact_score)
        candidate_source = _metadata_value(candidate_sources, idx, kind="text")

        score, meta = score_counterfactual_candidate(
            question=question,
            answer=answer,
            candidate=str(candidate),
            candidate_answers=candidate_answers,
            external_score=external_score,
            relation_score=relation_score,
            shared_fact_score=shared_fact_score,
            candidate_source=candidate_source,
            reject_gold_substring=reject_gold_substring,
            max_overlap_ratio=max_overlap_ratio,
            require_short_answer=require_short_answer,
            max_alt_length_chars=max_alt_length_chars,
        )
        if score > best_score:
            best_score = score
            best_text = clean_counterfactual_text(candidate)
            selected_source = candidate_source
            selected_pool = selected_source or "candidate_pool"
            best_meta = {
                "rank_score": float(score),
                "selected_candidate_index": int(idx),
                "selected_candidate_text": best_text,
                "selected_source": selected_source,
                "selected_from_pool": selected_pool,
                "used_low_confidence_fallback": selected_pool == "low_confidence_fallback",
                **meta,
            }

    return best_text, best_meta


def _perturb_int_text(text: str, seed: int) -> Optional[str]:
    match = re.fullmatch(r"([+-]?)(\d+)", text)
    if not match:
        return None
    sign, digits = match.groups()
    value = int(f"{sign}{digits}")
    step = 1 if seed % 2 == 0 else -1
    if value == 0:
        step = 1
    if value > 0 and value + step <= 0:
        step = 1
    return str(value + step)


def _perturb_decimal_text(text: str, seed: int) -> Optional[str]:
    match = re.fullmatch(r"([+-]?\d+)(\.\d+)", text)
    if not match:
        return None
    value = float(text)
    step = 0.1 if seed % 2 == 0 else -0.1
    decimals = len(match.group(2)) - 1
    return f"{value + step:.{decimals}f}"


def _perturb_decade_text(text: str, seed: int) -> Optional[str]:
    match = re.fullmatch(r"(\d{3,4})s", text)
    if not match:
        return None
    value = int(match.group(1))
    step = 10 if seed % 2 == 0 else -10
    return f"{max(0, value + step)}s"


def _ordinal_suffix(value: int) -> str:
    if 10 <= (value % 100) <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _perturb_ordinal_text(text: str, seed: int) -> Optional[str]:
    match = re.fullmatch(r"(\d+)(st|nd|rd|th)", text.lower())
    if not match:
        return None
    value = int(match.group(1))
    step = 1 if seed % 2 == 0 else -1
    if value <= 1 and step < 0:
        step = 1
    candidate = value + step
    return f"{candidate}{_ordinal_suffix(candidate)}"


def _perturb_month_name_date(text: str, seed: int) -> Optional[str]:
    if not _MONTH_NAME_RE.match(text):
        return None
    parts = clean_counterfactual_text(text).split()
    if len(parts) < 2:
        return None
    day_digits = re.sub(r"[^0-9]", "", parts[1])
    if not day_digits:
        return None
    day_value = int(day_digits)
    step = 1 if seed % 2 == 0 else -1
    if day_value <= 1 and step < 0:
        step = 1
    next_day = max(1, min(28, day_value + step))
    parts[1] = str(next_day) + ("," if "," in parts[1] else "")
    return " ".join(parts)


def build_answer_type_fallback_candidates(answer: Any, seed: int) -> list[str]:
    text = clean_counterfactual_text(answer)
    if not text:
        return []

    percent_suffix = "%" if text.endswith("%") else ""
    normalized = text.replace(",", "").replace("%", "").strip()
    candidates: list[str] = []
    for builder in (
        _perturb_ordinal_text,
        _perturb_decade_text,
        _perturb_decimal_text,
        _perturb_int_text,
        _perturb_month_name_date,
    ):
        candidate = builder(normalized if builder is not _perturb_month_name_date else text, seed)
        if candidate is not None:
            candidates.append(f"{candidate}{percent_suffix}" if builder is not _perturb_month_name_date else candidate)

    answer_type = detect_answer_type(text)
    if answer_type in {"year", "date"} and _INT_RE.match(normalized):
        year = int(normalized)
        candidates.append(str(year + (1 if seed % 2 == 0 else -1)))
    deduped = []
    seen = set()
    for candidate in candidates:
        cleaned = clean_counterfactual_text(candidate)
        if cleaned and cleaned not in seen:
            deduped.append(cleaned)
            seen.add(cleaned)
    return deduped


def build_low_confidence_fallback_candidates(answer: Any) -> list[str]:
    text = clean_counterfactual_text(answer)
    if not text:
        return []
    if detect_answer_type(text) != "short_span":
        return []

    normalized_answer = normalize_text(text)
    if not normalized_answer or normalized_answer.startswith("not "):
        return []
    return [f"not {text}"]


def pick_valid_candidate(
    answer: Any,
    candidates: Sequence[Any],
    *,
    reject_gold_substring: bool = True,
    max_overlap_ratio: Optional[float] = 0.85,
    require_short_answer: bool = True,
    max_alt_length_chars: Optional[int] = 128,
) -> Optional[str]:
    best_text, best_meta = pick_best_counterfactual_v3(
        question="",
        answer=str(answer),
        candidates=candidates,
        candidate_answers=[str(candidate) for candidate in candidates if str(candidate).strip()],
        reject_gold_substring=reject_gold_substring,
        max_overlap_ratio=max_overlap_ratio,
        require_short_answer=require_short_answer,
        max_alt_length_chars=max_alt_length_chars,
    )
    if best_meta.get("invalid_reason") is not None:
        return None
    return best_text


def answer_type_aware_fallback(answer: Any, seed: int = 0) -> Optional[str]:
    candidates = build_answer_type_fallback_candidates(answer, seed=seed)
    return candidates[0] if candidates else None


def _list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def duplicate_candidate_count(values: Sequence[Any]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for value in values:
        cleaned = clean_counterfactual_text(value)
        if not cleaned:
            continue
        if cleaned in seen:
            duplicates += 1
            continue
        seen.add(cleaned)
    return duplicates


def _aligned_metadata_coverage(values: Any, expected_length: int) -> bool:
    if expected_length <= 0 or not isinstance(values, list):
        return False
    return len(values) == expected_length


def build_artifact_quality_report(
    rows: Sequence[Dict[str, Any]],
    *,
    question_key: str = "question",
    answer_key: str = "answer",
    alternate_key: str = "alternate",
) -> Dict[str, Any]:
    total_rows = len(rows)
    invalid_reason_counts: Dict[str, int] = {}
    repair_source_counts: Dict[str, int] = {}
    selected_pool_counts: Dict[str, int] = {}
    exact_match_count = 0
    gold_substring_count = 0
    valid_row_count = 0
    sidecar_rows = 0
    relation_metadata_rows = 0
    shared_fact_metadata_rows = 0
    total_external_candidates = 0
    total_candidates = 0
    duplicate_external_candidate_rows = 0
    duplicate_candidate_bank_rows = 0
    duplicate_external_candidate_total = 0
    duplicate_candidate_bank_total = 0
    low_confidence_fallback_rows = 0
    rank_scores: list[float] = []

    for row in rows:
        answer = str(row.get(answer_key, ""))
        alternate = str(row.get(alternate_key, ""))
        answer_norm = normalize_text(answer)
        alternate_norm = normalize_text(alternate)
        if answer_norm and alternate_norm == answer_norm:
            exact_match_count += 1
        if answer_norm and alternate_norm and answer_norm != alternate_norm:
            if answer_norm in alternate_norm or alternate_norm in answer_norm:
                gold_substring_count += 1

        invalid_reason = row.get("cf_invalid_reason")
        if invalid_reason not in (None, "", "null", "None"):
            invalid_reason_counts[str(invalid_reason)] = (
                invalid_reason_counts.get(str(invalid_reason), 0) + 1
            )
        else:
            valid_row_count += 1

        external_alternates = _list_or_empty(row.get("external_alternates"))
        candidate_answers = _list_or_empty(row.get("candidate_answers"))
        total_external_candidates += len(external_alternates)
        total_candidates += len(external_alternates) + len(candidate_answers)
        if external_alternates:
            sidecar_rows += 1
        external_duplicate_count = duplicate_candidate_count(external_alternates)
        if external_duplicate_count > 0:
            duplicate_external_candidate_rows += 1
            duplicate_external_candidate_total += external_duplicate_count
        bank_duplicate_count = duplicate_candidate_count(candidate_answers)
        if bank_duplicate_count > 0:
            duplicate_candidate_bank_rows += 1
            duplicate_candidate_bank_total += bank_duplicate_count

        if (
            _aligned_metadata_coverage(
                row.get("external_alternate_relation_scores"),
                len(external_alternates),
            )
            or _aligned_metadata_coverage(
                row.get("candidate_relation_scores"),
                len(candidate_answers),
            )
            or row.get("cf_same_relation") not in (None, "", "null", "None")
        ):
            relation_metadata_rows += 1
        if (
            _aligned_metadata_coverage(
                row.get("external_alternate_shared_fact_scores"),
                len(external_alternates),
            )
            or _aligned_metadata_coverage(
                row.get("candidate_shared_fact_scores"),
                len(candidate_answers),
            )
        ):
            shared_fact_metadata_rows += 1

        pick_meta = row.get("cf_pick_meta")
        if isinstance(pick_meta, dict):
            selected_source = str(pick_meta.get("selected_source") or "unknown")
            selected_pool = str(pick_meta.get("selected_from_pool") or "unknown")
            repair_source_counts[selected_source] = (
                repair_source_counts.get(selected_source, 0) + 1
            )
            selected_pool_counts[selected_pool] = (
                selected_pool_counts.get(selected_pool, 0) + 1
            )
            if bool(pick_meta.get("used_low_confidence_fallback", False)):
                low_confidence_fallback_rows += 1
            rank_score = _coerce_optional_float(pick_meta.get("rank_score"))
            if rank_score is not None:
                rank_scores.append(rank_score)

    def _ratio(count: int) -> float:
        if total_rows <= 0:
            return 0.0
        return float(count) / float(total_rows)

    report = {
        "rows": total_rows,
        "question_key": question_key,
        "valid_row_count": valid_row_count,
        "valid_row_rate": _ratio(valid_row_count),
        "invalid_reason_counts": invalid_reason_counts,
        "exact_match_count": exact_match_count,
        "gold_substring_count": gold_substring_count,
        "average_external_candidate_count": (
            float(total_external_candidates) / float(total_rows) if total_rows else 0.0
        ),
        "average_total_candidate_count": (
            float(total_candidates) / float(total_rows) if total_rows else 0.0
        ),
        "duplicate_external_candidate_rows": duplicate_external_candidate_rows,
        "duplicate_candidate_bank_rows": duplicate_candidate_bank_rows,
        "duplicate_external_candidate_total": duplicate_external_candidate_total,
        "duplicate_candidate_bank_total": duplicate_candidate_bank_total,
        "sidecar_coverage_rows": sidecar_rows,
        "sidecar_coverage_rate": _ratio(sidecar_rows),
        "relation_metadata_rows": relation_metadata_rows,
        "relation_metadata_coverage_rate": _ratio(relation_metadata_rows),
        "shared_fact_metadata_rows": shared_fact_metadata_rows,
        "shared_fact_metadata_coverage_rate": _ratio(shared_fact_metadata_rows),
        "repair_source_counts": repair_source_counts,
        "selected_pool_counts": selected_pool_counts,
        "low_confidence_fallback_rows": low_confidence_fallback_rows,
    }
    if rank_scores:
        report["selected_rank_score_mean"] = sum(rank_scores) / float(len(rank_scores))
        report["selected_rank_score_min"] = min(rank_scores)
        report["selected_rank_score_max"] = max(rank_scores)
    return report


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
