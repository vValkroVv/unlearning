#!/usr/bin/env python3
"""Create DualCF counterfactual forget artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.utils import preprocess_chat_instance
from tools.dual_cf_artifact_utils import (
    build_artifact_quality_report,
    build_answer_type_fallback_candidates,
    build_low_confidence_fallback_candidates,
    clean_counterfactual_text,
    counterfactual_invalid_reason,
    dedupe_candidate_metadata,
    duplicate_candidate_count,
    load_dataset_split,
    load_keyed_jsonish,
    load_model_bundle,
    pick_best_counterfactual_v3,
    resolve_answer,
    save_jsonl,
    select_device,
)
from tools.vllm_cf_client import VLLMCFGenerator, chunked


def log(message: str) -> None:
    print(f"[make_counterfactuals] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build counterfactual datasets for DualCF.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--answer-index", type=int, default=None)
    parser.add_argument("--alternate-column", default=None)
    parser.add_argument("--alternate-jsonl", default=None)
    parser.add_argument("--mapping-key", default="index")
    parser.add_argument("--mapping-alternate-key", default="alternate")
    parser.add_argument("--external-score-key", default="scores")
    parser.add_argument("--external-relation-score-key", default="relation_scores")
    parser.add_argument("--external-shared-fact-score-key", default="shared_fact_scores")
    parser.add_argument("--external-source-key", default="candidate_sources")
    parser.add_argument("--allow-list-alternates", action="store_true")
    parser.add_argument("--candidate-bank", default=None)
    parser.add_argument("--generator-backend", choices=("hf", "vllm_openai"), default="hf")
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--generator-concurrency", type=int, default=64)
    parser.add_argument("--generator-batch-size", type=int, default=256)
    parser.add_argument("--model-cfg", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--model-subfolder", default=None)
    parser.add_argument("--tokenizer-subfolder", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-alternates", type=int, default=1)
    parser.add_argument("--prompt-family", default="default")
    parser.add_argument("--repair-invalid", action="store_true")
    parser.add_argument("--allow-low-confidence-fallback", action="store_true")
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def _string_list(value: Any, allow_scalar: bool = True) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if allow_scalar and str(value).strip():
        return [str(value).strip()]
    return []


def _score_list(value: Any, expected_length: int) -> list[Any]:
    if not isinstance(value, list):
        return [None] * expected_length
    scores = list(value[:expected_length])
    if len(scores) < expected_length:
        scores.extend([None] * (expected_length - len(scores)))
    return scores


def _source_list(value: Any, expected_length: int, default_value: str | None = None) -> list[Any]:
    if not isinstance(value, list):
        return [default_value] * expected_length
    sources = [str(item).strip() if str(item).strip() else default_value for item in value[:expected_length]]
    if len(sources) < expected_length:
        sources.extend([default_value] * (expected_length - len(sources)))
    return sources


def _candidate_bank_metadata(
    bank_row: Dict[str, Any],
    row_candidates: list[str],
) -> tuple[list[Any], list[Any], list[Any]]:
    expected_length = len(row_candidates)
    return (
        _score_list(bank_row.get("candidate_relation_scores"), expected_length),
        _score_list(bank_row.get("candidate_shared_fact_scores"), expected_length),
        _source_list(
            bank_row.get("candidate_sources"),
            expected_length,
            default_value="candidate_bank",
        ),
    )


def _default_prompt_version(
    *,
    backend: str,
    prompt_family: str,
) -> str:
    backend_name = str(backend or "unknown")
    prompt_name = str(prompt_family or "default")
    return f"{backend_name}:{prompt_name}:v1"


def load_mapping(path: str, key_field: str, alternate_field: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if key_field not in row or alternate_field not in row:
                raise KeyError(
                    f"JSONL row must contain `{key_field}` and `{alternate_field}`."
                )
            mapping[str(row[key_field])] = row
    return mapping


def build_hf_generator(args):
    log(
        "Loading generator model "
        f"cfg={args.model_cfg} model_path={args.model_path or '<cfg-default>'} "
        f"tokenizer_path={args.tokenizer_path or '<model-path>'}"
    )
    model, tokenizer, template_args = load_model_bundle(
        model_cfg_path=args.model_cfg,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        model_subfolder=args.model_subfolder,
        tokenizer_subfolder=args.tokenizer_subfolder,
    )
    device = select_device(args.device)
    if getattr(model, "hf_device_map", None) is None:
        model = model.to(device)
    model.eval()
    log(f"Generator model ready on device={device}")

    prompt_families = {
        "default": [
            (
                "Question: {question}\n"
                "True answer: {answer}\n"
                "Write one short plausible but incorrect alternative answer. "
                "Output only the alternative answer."
            ),
            (
                "Provide a short counterfactual answer to the question below.\n"
                "It must be different from the true answer.\n"
                "Question: {question}\n"
                "True answer: {answer}\n"
                "Output only the counterfactual answer."
            ),
        ],
        "strict_short": [
            (
                "Question: {question}\n"
                "True answer: {answer}\n"
                "Return {num_alternates} short incorrect alternatives with the same answer type. "
                "Output only the answers, one per line."
            )
        ],
        "duet_relation_safe": [
            (
                "Question: {question}\n"
                "True answer: {answer}\n"
                "Return {num_alternates} short incorrect alternatives that preserve the same relation "
                "and answer type. Do not repeat, negate, or paraphrase the true answer. "
                "Output only the answers, one per line."
            )
        ],
        "rwku_shared_fact_safe": [
            (
                "Question: {question}\n"
                "True answer: {answer}\n"
                "Return {num_alternates} short incorrect alternatives that preserve the same answer type "
                "and avoid changing unrelated shared facts. Do not add explanations. "
                "Output only the answers, one per line."
            )
        ],
    }
    prompt_variants = prompt_families.get(args.prompt_family, prompt_families["default"])

    def generate_alternates(question: str, answer: str) -> list[str]:
        answer_norm = " ".join(str(answer).strip().lower().split())
        seen = set()
        generated_candidates: list[str] = []

        for prompt_template in prompt_variants:
            cf_prompt = prompt_template.format(
                question=question,
                answer=answer,
                num_alternates=max(1, int(args.num_alternates)),
            )
            prompt_item = preprocess_chat_instance(
                tokenizer=tokenizer,
                template_config=template_args,
                prompt_msgs=[cf_prompt],
                response_msgs=[""],
                max_length=args.max_length,
                predict_with_generate=True,
            )
            input_ids = prompt_item["input_ids"].unsqueeze(0)
            attention_mask = prompt_item["attention_mask"].unsqueeze(0)
            if getattr(model, "hf_device_map", None) is None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=args.temperature > 0.0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=max(1, int(args.num_alternates)),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for sequence in outputs:
                generated = sequence[input_ids.shape[-1] :]
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                for candidate in decoded.splitlines():
                    cleaned = clean_counterfactual_text(candidate)
                    candidate_norm = " ".join(cleaned.strip().lower().split())
                    if (
                        cleaned
                        and candidate_norm != answer_norm
                        and candidate_norm not in seen
                    ):
                        seen.add(candidate_norm)
                        generated_candidates.append(cleaned)
                    if len(generated_candidates) >= max(1, int(args.num_alternates)):
                        return generated_candidates

        return generated_candidates

    return generate_alternates


def build_vllm_generator(args) -> VLLMCFGenerator:
    if not args.vllm_base_url:
        raise ValueError("--vllm-base-url is required for --generator-backend=vllm_openai")
    if not args.vllm_model:
        raise ValueError("--vllm-model is required for --generator-backend=vllm_openai")
    return VLLMCFGenerator(
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
        model=args.vllm_model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        concurrency=args.generator_concurrency,
        prompt_family=args.prompt_family,
        num_alternates=args.num_alternates,
    )


def maybe_load_candidate_bank(path: Optional[str], mapping_key: str) -> Dict[str, Dict]:
    if path in (None, "", "null", "None"):
        return {}
    bank = load_keyed_jsonish(path, key_field=mapping_key)
    log(f"Loaded candidate bank rows={len(bank)} from {path}")
    return bank


def build_cf_provenance(
    *,
    args,
    source_row: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "generator_backend": backend or args.generator_backend,
        "prompt_family": args.prompt_family,
        "candidate_count": max(1, int(args.num_alternates)),
        "prompt_version": _default_prompt_version(
            backend=backend or args.generator_backend,
            prompt_family=args.prompt_family,
        ),
    }
    if model_name not in (None, "", "null", "None"):
        provenance["generator_model"] = model_name
    if source_row:
        for key in (
            "generator",
            "prompt_version",
            "temperature",
            "top_p",
            "structured_outputs",
        ):
            value = source_row.get(key)
            if value not in (None, "", "null", "None"):
                provenance[key] = value
        candidate_count = source_row.get("candidate_count")
        if candidate_count not in (None, "", "null", "None"):
            provenance["candidate_count"] = int(candidate_count)
    return provenance


def select_best_alternate(
    *,
    args,
    question: str,
    answer: str,
    seed: int,
    primary_candidates: list[str],
    row_candidates: list[str],
    row_relation_scores: list[Any] | None = None,
    row_shared_fact_scores: list[Any] | None = None,
    row_sources: list[Any] | None = None,
    external_candidates: list[str] | None = None,
    external_scores: list[Any] | None = None,
    external_relation_scores: list[Any] | None = None,
    external_shared_fact_scores: list[Any] | None = None,
    external_sources: list[Any] | None = None,
    default_relation_score: float | None = None,
):
    external_candidates = list(external_candidates or [])
    candidate_pool = []
    score_pool = []
    relation_score_pool = []
    shared_fact_score_pool = []
    source_pool = []
    low_confidence_candidates: list[str] = []
    duplicate_candidates_removed = 0

    for candidate in primary_candidates:
        candidate_pool.append(candidate)
        score_pool.append(None)
        relation_score_pool.append(None)
        shared_fact_score_pool.append(None)
        source_pool.append("primary")
    for idx, candidate in enumerate(external_candidates):
        candidate_pool.append(candidate)
        score_pool.append(
            external_scores[idx] if external_scores is not None and idx < len(external_scores) else None
        )
        relation_score_pool.append(
            external_relation_scores[idx]
            if external_relation_scores is not None and idx < len(external_relation_scores)
            else None
        )
        shared_fact_score_pool.append(
            external_shared_fact_scores[idx]
            if external_shared_fact_scores is not None and idx < len(external_shared_fact_scores)
            else None
        )
        source_pool.append(
            external_sources[idx]
            if external_sources is not None and idx < len(external_sources)
            else "external"
        )
    for candidate in row_candidates:
        candidate_pool.append(candidate)
        score_pool.append(None)
    row_relation_scores = _score_list(row_relation_scores, len(row_candidates))
    row_shared_fact_scores = _score_list(row_shared_fact_scores, len(row_candidates))
    row_sources = _source_list(row_sources, len(row_candidates), default_value="candidate_bank")
    relation_score_pool.extend(row_relation_scores)
    shared_fact_score_pool.extend(row_shared_fact_scores)
    source_pool.extend(row_sources)
    if args.repair_invalid:
        for candidate in build_answer_type_fallback_candidates(answer, seed=seed):
            candidate_pool.append(candidate)
            score_pool.append(None)
            relation_score_pool.append(None)
            shared_fact_score_pool.append(None)
            source_pool.append("typed_fallback")
        if args.allow_low_confidence_fallback:
            low_confidence_candidates = build_low_confidence_fallback_candidates(answer)

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
    best_alt, best_meta = pick_best_counterfactual_v3(
        question=question,
        answer=answer,
        candidates=candidate_pool,
        candidate_answers=row_candidates,
        external_scores=score_pool,
        relation_scores=relation_score_pool,
        shared_fact_scores=shared_fact_score_pool,
        candidate_sources=source_pool,
        default_relation_score=default_relation_score,
        reject_gold_substring=args.reject_gold_substring,
        max_overlap_ratio=args.max_overlap_ratio,
        require_short_answer=args.require_short_answer,
        max_alt_length_chars=args.max_alt_length_chars,
    )
    best_meta["selected_candidate"] = best_meta.get("selected_candidate_text", best_alt)
    best_meta["candidate_pool_size"] = len(candidate_pool)
    best_meta["duplicate_candidates_removed"] = int(duplicate_candidates_removed)
    invalid_reason = counterfactual_invalid_reason(
        best_alt,
        answer,
        reject_gold_substring=args.reject_gold_substring,
        max_overlap_ratio=args.max_overlap_ratio,
        require_short_answer=args.require_short_answer,
        max_alt_length_chars=args.max_alt_length_chars,
    )
    if invalid_reason is not None and args.repair_invalid and low_confidence_candidates:
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
        best_alt, best_meta = pick_best_counterfactual_v3(
            question=question,
            answer=answer,
            candidates=candidate_pool,
            candidate_answers=row_candidates,
            external_scores=score_pool,
            relation_scores=relation_score_pool,
            shared_fact_scores=shared_fact_score_pool,
            candidate_sources=source_pool,
            default_relation_score=default_relation_score,
            reject_gold_substring=args.reject_gold_substring,
            max_overlap_ratio=args.max_overlap_ratio,
            require_short_answer=args.require_short_answer,
            max_alt_length_chars=args.max_alt_length_chars,
        )
        best_meta["selected_candidate"] = best_meta.get("selected_candidate_text", best_alt)
        best_meta["candidate_pool_size"] = len(candidate_pool)
        best_meta["duplicate_candidates_removed"] = int(duplicate_candidates_removed)
        invalid_reason = counterfactual_invalid_reason(
            best_alt,
            answer,
            reject_gold_substring=args.reject_gold_substring,
            max_overlap_ratio=args.max_overlap_ratio,
            require_short_answer=args.require_short_answer,
            max_alt_length_chars=args.max_alt_length_chars,
        )
    baseline_candidate = ""
    for candidate in list(primary_candidates) + external_candidates:
        baseline_candidate = clean_counterfactual_text(candidate)
        if baseline_candidate:
            break
    repaired = bool(best_alt) and clean_counterfactual_text(best_alt) != baseline_candidate
    best_meta["selected_candidate"] = best_meta.get("selected_candidate_text", best_alt)
    best_meta["candidate_pool_size"] = len(candidate_pool)
    best_meta["duplicate_candidates_removed"] = int(duplicate_candidates_removed)
    return best_alt, invalid_reason, repaired, best_meta


def main():
    args = parse_args()
    log(
        "Starting with "
        f"dataset_path={args.dataset_path} split={args.split} "
        f"dataset_name={args.dataset_name} data_files={args.data_files} "
        f"output_path={args.output_path}"
    )
    dataset = load_dataset_split(
        path=args.dataset_path,
        split=args.split,
        name=args.dataset_name,
        data_files=args.data_files,
        max_examples=args.max_examples,
    )
    dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
    if dataset_size is not None:
        log(f"Loaded {dataset_size} source rows")

    mapping = None
    candidate_bank = maybe_load_candidate_bank(args.candidate_bank, args.mapping_key)
    generate_alternates = None
    vllm_generator = None
    if args.alternate_column:
        log(f"Using alternate column `{args.alternate_column}` from the dataset")
    elif args.alternate_jsonl:
        log(
            "Loading alternates from JSONL "
            f"path={args.alternate_jsonl} key={args.mapping_key} "
            f"alternate_key={args.mapping_alternate_key}"
        )
        mapping = load_mapping(
            path=args.alternate_jsonl,
            key_field=args.mapping_key,
            alternate_field=args.mapping_alternate_key,
        )
        log(f"Loaded {len(mapping)} alternate mappings")
    elif args.generator_backend == "hf" and args.model_cfg:
        generate_alternates = build_hf_generator(args)
    elif args.generator_backend == "vllm_openai":
        vllm_generator = build_vllm_generator(args)
    else:
        raise ValueError(
            "Choose one alternate source: --alternate-column, --alternate-jsonl, "
            "or a generator backend with the required model settings."
        )

    rows = []
    empty_alternate_count = 0
    same_as_answer_count = 0
    repaired_count = 0
    invalid_count = 0
    row_iter = tqdm(
        dataset,
        total=dataset_size,
        desc="counterfactual_rows",
        dynamic_ncols=True,
    )
    pending_vllm_rows = []
    pending_vllm_meta = []
    for row_no, row in enumerate(row_iter, start=1):
        row_index = row.get("index", "<missing>")
        answer = resolve_answer(
            row=row,
            answer_key=args.answer_key,
            answer_index=args.answer_index,
        )
        question = str(row[args.question_key])
        bank_row = candidate_bank.get(str(row.get(args.mapping_key, "")), {})
        row_candidates = list(bank_row.get("candidate_answers", []))
        (
            row_relation_scores,
            row_shared_fact_scores,
            row_sources,
        ) = _candidate_bank_metadata(bank_row, row_candidates)

        if args.alternate_column or mapping is not None or generate_alternates is not None:
            try:
                mapping_row: dict[str, Any] | None = None
                external_candidates: list[str] = []
                external_scores: list[Any] | None = None
                external_relation_scores: list[Any] | None = None
                external_shared_fact_scores: list[Any] | None = None
                external_sources: list[Any] | None = None
                primary_candidates: list[str] = []
                cf_source = "hf_generator"
                default_relation_score = None

                if args.alternate_column:
                    primary_candidates = _string_list(row[args.alternate_column])
                    cf_source = "alternate_column"
                elif mapping is not None:
                    join_value = str(row[args.mapping_key])
                    if join_value not in mapping:
                        raise KeyError(
                            f"No alternate found for {args.mapping_key}={join_value} in mapping."
                        )
                    mapping_row = dict(mapping[join_value])
                    mapped_value = mapping_row[args.mapping_alternate_key]
                    if isinstance(mapped_value, list) and not args.allow_list_alternates:
                        raise ValueError(
                            "Received a list of alternates from the sidecar but "
                            "--allow-list-alternates was not enabled."
                        )
                    external_candidates = _string_list(mapped_value)
                    external_scores = _score_list(
                        mapping_row.get(args.external_score_key),
                        expected_length=len(external_candidates),
                    )
                    external_relation_scores = _score_list(
                        mapping_row.get(args.external_relation_score_key),
                        expected_length=len(external_candidates),
                    )
                    external_shared_fact_scores = _score_list(
                        mapping_row.get(args.external_shared_fact_score_key),
                        expected_length=len(external_candidates),
                    )
                    external_sources = _source_list(
                        mapping_row.get(args.external_source_key),
                        expected_length=len(external_candidates),
                        default_value="sidecar",
                    )
                    if mapping_row.get("same_relation") not in (None, "", "null", "None"):
                        default_relation_score = 1.0 if bool(mapping_row.get("same_relation")) else 0.0
                    primary_candidates = []
                    cf_source = "alternate_jsonl_multi" if len(external_candidates) > 1 else "alternate_jsonl"
                else:
                    primary_candidates = generate_alternates(question, str(answer))

                best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
                    args=args,
                    question=question,
                    answer=answer,
                    seed=int(row.get(args.mapping_key, row_no) or row_no),
                    primary_candidates=primary_candidates,
                    row_candidates=row_candidates,
                    row_relation_scores=row_relation_scores,
                    row_shared_fact_scores=row_shared_fact_scores,
                    row_sources=row_sources,
                    external_candidates=external_candidates,
                    external_scores=external_scores,
                    external_relation_scores=external_relation_scores,
                    external_shared_fact_scores=external_shared_fact_scores,
                    external_sources=external_sources,
                    default_relation_score=default_relation_score,
                )

                updated = dict(row)
                updated["answer"] = answer
                updated["alternate"] = best_alt
                updated["candidate_answers"] = row_candidates
                updated["candidate_relation_scores"] = row_relation_scores
                updated["candidate_shared_fact_scores"] = row_shared_fact_scores
                updated["candidate_sources"] = row_sources
                updated["external_alternates"] = external_candidates
                updated["external_alternate_scores"] = external_scores
                updated["external_alternate_relation_scores"] = external_relation_scores
                updated["external_alternate_shared_fact_scores"] = external_shared_fact_scores
                updated["external_alternate_sources"] = external_sources
                updated["cf_pick_meta"] = best_meta
                updated["cf_source"] = cf_source
                updated["cf_generator_backend"] = args.generator_backend
                updated["cf_prompt_family"] = args.prompt_family
                updated["cf_num_alternates"] = max(1, int(args.num_alternates))
                updated["cf_invalid_reason"] = invalid_reason
                updated["cf_is_valid"] = invalid_reason is None
                updated["cf_provenance"] = build_cf_provenance(
                    args=args,
                    source_row=mapping_row,
                    backend=args.generator_backend,
                    model_name=(args.model_path or args.model_cfg) if args.model_cfg else None,
                )
                if args.model_cfg:
                    updated["cf_generator_model"] = args.model_path or args.model_cfg
                if repaired:
                    repaired_count += 1
                if invalid_reason is not None:
                    invalid_count += 1
                if not updated["alternate"]:
                    empty_alternate_count += 1
                if clean_counterfactual_text(updated["alternate"]) == clean_counterfactual_text(answer):
                    same_as_answer_count += 1
                rows.append(updated)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed processing row_no={row_no} index={row_index} "
                    f"question_key={args.question_key}"
                ) from exc
        else:
            pending_vllm_rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "candidate_answers": row_candidates,
                }
            )
            pending_vllm_meta.append(
                (
                    row_no,
                    dict(row),
                    answer,
                    question,
                    row_candidates,
                    row_relation_scores,
                    row_shared_fact_scores,
                    row_sources,
                )
            )

    if vllm_generator is not None and pending_vllm_rows:
        log(f"Sending {len(pending_vllm_rows)} rows to vLLM generator")
        for row_chunk, meta_chunk in zip(
            chunked(pending_vllm_rows, args.generator_batch_size),
            chunked(pending_vllm_meta, args.generator_batch_size),
        ):
            outputs = vllm_generator.many_sync(list(row_chunk))
            for response, (
                row_no,
                row,
                answer,
                question,
                row_candidates,
                row_relation_scores,
                row_shared_fact_scores,
                row_sources,
            ) in zip(outputs, meta_chunk):
                row_index = row.get("index", "<missing>")
                try:
                    response_candidates = _string_list(
                        response.get("alternates", response.get("alternate", "")),
                    )
                    if not response_candidates and str(response.get("alternate", "")).strip():
                        response_candidates = [str(response.get("alternate", "")).strip()]
                    response_scores = _score_list(
                        response.get(args.external_score_key),
                        expected_length=len(response_candidates),
                    )
                    response_relation_scores = _score_list(
                        response.get(args.external_relation_score_key),
                        expected_length=len(response_candidates),
                    )
                    response_shared_fact_scores = _score_list(
                        response.get(args.external_shared_fact_score_key),
                        expected_length=len(response_candidates),
                    )
                    response_sources = _source_list(
                        response.get(args.external_source_key),
                        expected_length=len(response_candidates),
                        default_value="vllm_generated",
                    )
                    default_relation_score = (
                        1.0
                        if bool(response.get("same_relation", True))
                        else 0.0
                    )
                    best_alt, invalid_reason, repaired, best_meta = select_best_alternate(
                        args=args,
                        question=question,
                        answer=answer,
                        seed=int(row.get(args.mapping_key, row_no) or row_no),
                        primary_candidates=[],
                        row_candidates=row_candidates,
                        row_relation_scores=row_relation_scores,
                        row_shared_fact_scores=row_shared_fact_scores,
                        row_sources=row_sources,
                        external_candidates=response_candidates,
                        external_scores=response_scores,
                        external_relation_scores=response_relation_scores,
                        external_shared_fact_scores=response_shared_fact_scores,
                        external_sources=response_sources,
                        default_relation_score=default_relation_score,
                    )
                    updated = dict(row)
                    updated["answer"] = answer
                    updated["alternate"] = best_alt
                    updated["candidate_answers"] = row_candidates
                    updated["candidate_relation_scores"] = row_relation_scores
                    updated["candidate_shared_fact_scores"] = row_shared_fact_scores
                    updated["candidate_sources"] = row_sources
                    updated["external_alternates"] = response_candidates
                    updated["external_alternate_scores"] = response_scores
                    updated["external_alternate_relation_scores"] = response_relation_scores
                    updated["external_alternate_shared_fact_scores"] = response_shared_fact_scores
                    updated["external_alternate_sources"] = response_sources
                    updated["cf_pick_meta"] = best_meta
                    updated["cf_source"] = "vllm_primary"
                    updated["cf_generator_backend"] = "vllm_openai"
                    updated["cf_generator_model"] = args.vllm_model
                    updated["cf_prompt_family"] = args.prompt_family
                    updated["cf_num_alternates"] = max(1, int(args.num_alternates))
                    updated["cf_same_relation"] = bool(response.get("same_relation", True))
                    updated["cf_answer_type"] = str(response.get("answer_type", "unknown"))
                    updated["cf_invalid_reason"] = invalid_reason
                    updated["cf_is_valid"] = invalid_reason is None
                    updated["cf_provenance"] = build_cf_provenance(
                        args=args,
                        source_row=response,
                        backend="vllm_openai",
                        model_name=args.vllm_model,
                    )
                    if repaired:
                        repaired_count += 1
                    if invalid_reason is not None:
                        invalid_count += 1
                    if not updated["alternate"]:
                        empty_alternate_count += 1
                    if clean_counterfactual_text(updated["alternate"]) == clean_counterfactual_text(answer):
                        same_as_answer_count += 1
                    rows.append(updated)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed processing vLLM row_no={row_no} index={row_index} "
                        f"question_key={args.question_key}"
                    ) from exc

    log(f"Saving {len(rows)} rows to {args.output_path}")
    save_jsonl(rows, args.output_path)
    if args.report_path not in (None, "", "null", "None"):
        report = build_artifact_quality_report(
            rows,
            question_key=args.question_key,
            answer_key="answer",
            alternate_key="alternate",
        )
        with open(args.report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=True)
        log(f"Saved artifact-quality report to {args.report_path}")
    log(
        "Done. "
        f"rows={len(rows)} empty_alternates={empty_alternate_count} "
        f"same_as_answer_after_normalization={same_as_answer_count} "
        f"repaired={repaired_count} invalid={invalid_count}"
    )


if __name__ == "__main__":
    main()
