#!/usr/bin/env python3
"""Create DualCF counterfactual forget artifacts.

Supports three practical modes:
1. Copy an existing alternate column from a dataset.
2. Join alternates from a JSONL sidecar file.
3. Generate one alternate answer per question with a local/HF model config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.utils import preprocess_chat_instance
from tools.dual_cf_artifact_utils import (
    clean_counterfactual_text,
    counterfactual_invalid_reason,
    load_dataset_split,
    load_keyed_jsonish,
    load_model_bundle,
    pick_valid_candidate,
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
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repair-invalid", action="store_true")
    parser.add_argument("--reject-gold-substring", action="store_true")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-overlap-ratio", type=float, default=0.85)
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    return parser.parse_args()


def load_mapping(path: str, key_field: str, alternate_field: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if key_field not in row or alternate_field not in row:
                raise KeyError(
                    f"JSONL row must contain `{key_field}` and `{alternate_field}`."
                )
            mapping[str(row[key_field])] = str(row[alternate_field]).strip()
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

    def generate_alternate(question: str, answer: str) -> str:
        prompt_variants = [
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
        ]

        answer_norm = " ".join(str(answer).strip().lower().split())
        last_text = ""
        for prompt_template in prompt_variants:
            cf_prompt = prompt_template.format(question=question, answer=answer)
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
            target_device = input_ids.device
            if getattr(model, "hf_device_map", None) is None:
                target_device = torch.device(device)
                input_ids = input_ids.to(target_device)
                attention_mask = attention_mask.to(target_device)
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=args.temperature > 0.0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[0, input_ids.shape[-1] :]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            last_text = text
            if text and " ".join(str(text).strip().lower().split()) != answer_norm:
                return text
        return last_text

    return generate_alternate


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
    )


def maybe_load_candidate_bank(path: Optional[str], mapping_key: str) -> Dict[str, Dict]:
    if path in (None, "", "null", "None"):
        return {}
    bank = load_keyed_jsonish(path, key_field=mapping_key)
    log(f"Loaded candidate bank rows={len(bank)} from {path}")
    return bank


def repair_or_validate_alternate(args, answer: str, alternate: str, row_candidates: List[str]):
    cleaned = clean_counterfactual_text(alternate)
    invalid_reason = counterfactual_invalid_reason(
        cleaned,
        answer,
        reject_gold_substring=args.reject_gold_substring,
        max_overlap_ratio=args.max_overlap_ratio,
        require_short_answer=args.require_short_answer,
        max_alt_length_chars=args.max_alt_length_chars,
    )
    if invalid_reason and args.repair_invalid:
        repaired = pick_valid_candidate(
            answer,
            row_candidates,
            reject_gold_substring=args.reject_gold_substring,
            max_overlap_ratio=args.max_overlap_ratio,
            require_short_answer=args.require_short_answer,
            max_alt_length_chars=args.max_alt_length_chars,
        )
        if repaired is not None:
            return repaired, None, True
    return cleaned, invalid_reason, False


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
    generate_alternate = None
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
        log(
            "Generating alternates with model "
            f"max_new_tokens={args.max_new_tokens} temperature={args.temperature} top_p={args.top_p}"
        )
        generate_alternate = build_hf_generator(args)
    elif args.generator_backend == "vllm_openai":
        log(
            "Generating alternates with vLLM server "
            f"base_url={args.vllm_base_url} model={args.vllm_model} "
            f"concurrency={args.generator_concurrency} batch_size={args.generator_batch_size}"
        )
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
        bank_row = candidate_bank.get(str(row.get(args.mapping_key, "")), {})
        row_candidates = list(bank_row.get("candidate_answers", []))
        if args.alternate_column or mapping is not None or generate_alternate is not None:
            try:
                if args.alternate_column:
                    alternate = row[args.alternate_column]
                elif mapping is not None:
                    join_value = str(row[args.mapping_key])
                    if join_value not in mapping:
                        raise KeyError(
                            f"No alternate found for {args.mapping_key}={join_value} in mapping."
                        )
                    alternate = mapping[join_value]
                else:
                    alternate = generate_alternate(
                        str(row[args.question_key]),
                        str(answer),
                    )

                final_alternate, invalid_reason, repaired = repair_or_validate_alternate(
                    args=args,
                    answer=answer,
                    alternate=str(alternate),
                    row_candidates=row_candidates,
                )
                updated = dict(row)
                updated["answer"] = answer
                updated["alternate"] = final_alternate
                updated["candidate_answers"] = row_candidates
                updated["cf_source"] = (
                    "alternate_column"
                    if args.alternate_column
                    else "alternate_jsonl"
                    if mapping is not None
                    else "hf_generator"
                )
                updated["cf_generator_backend"] = args.generator_backend
                updated["cf_invalid_reason"] = invalid_reason
                if args.model_cfg:
                    updated["cf_generator_model"] = args.model_path or args.model_cfg
                if repaired:
                    repaired_count += 1
                if invalid_reason is not None:
                    invalid_count += 1
                if not updated["alternate"]:
                    empty_alternate_count += 1
                if updated["alternate"] == answer:
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
                    "question": str(row[args.question_key]),
                    "answer": answer,
                    "candidate_answers": row_candidates,
                }
            )
            pending_vllm_meta.append((row_no, dict(row), answer, row_candidates))

    if vllm_generator is not None and pending_vllm_rows:
        log(f"Sending {len(pending_vllm_rows)} rows to vLLM generator")
        for row_chunk, meta_chunk in zip(
            chunked(pending_vllm_rows, args.generator_batch_size),
            chunked(pending_vllm_meta, args.generator_batch_size),
        ):
            outputs = vllm_generator.many_sync(list(row_chunk))
            for response, (row_no, row, answer, row_candidates) in zip(outputs, meta_chunk):
                row_index = row.get("index", "<missing>")
                try:
                    final_alternate, invalid_reason, repaired = repair_or_validate_alternate(
                        args=args,
                        answer=answer,
                        alternate=response.get("alternate", ""),
                        row_candidates=row_candidates,
                    )
                    updated = dict(row)
                    updated["answer"] = answer
                    updated["alternate"] = final_alternate
                    updated["candidate_answers"] = row_candidates
                    updated["cf_source"] = "vllm_primary"
                    updated["cf_generator_backend"] = "vllm_openai"
                    updated["cf_generator_model"] = args.vllm_model
                    updated["cf_same_relation"] = bool(response.get("same_relation", True))
                    updated["cf_answer_type"] = str(response.get("answer_type", "unknown"))
                    updated["cf_invalid_reason"] = invalid_reason
                    if repaired:
                        repaired_count += 1
                    if invalid_reason is not None:
                        invalid_count += 1
                    if not updated["alternate"]:
                        empty_alternate_count += 1
                    if updated["alternate"] == answer:
                        same_as_answer_count += 1
                    rows.append(updated)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed processing vLLM row_no={row_no} index={row_index} "
                        f"question_key={args.question_key}"
                    ) from exc

    log(f"Saving {len(rows)} rows to {args.output_path}")
    save_jsonl(rows, args.output_path)
    log(
        "Done. "
        f"rows={len(rows)} empty_alternates={empty_alternate_count} "
        f"same_as_answer_after_normalization={same_as_answer_count} "
        f"repaired={repaired_count} invalid={invalid_count}"
    )


if __name__ == "__main__":
    main()
