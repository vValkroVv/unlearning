#!/usr/bin/env python3
"""Build belief alternates for forget artifacts using a local/base model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.utils import preprocess_chat_instance
from tools.dual_cf_artifact_utils import (
    clean_counterfactual_text,
    counterfactual_invalid_reason,
    load_model_bundle,
    normalize_text,
    read_jsonl,
    save_jsonl,
    select_device,
)


def log(message: str) -> None:
    print(f"[build_forget_belief_bank] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--alternate-key", default="alternate")
    parser.add_argument("--belief-key", default="belief_alternate")
    parser.add_argument("--belief-candidates-key", default="belief_candidates")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--model-cfg", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--model-subfolder", default=None)
    parser.add_argument("--tokenizer-subfolder", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-return-sequences", type=int, default=3)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt-version", default="belief_v1")
    parser.add_argument("--require-short-answer", action="store_true")
    parser.add_argument("--max-alt-length-chars", type=int, default=128)
    return parser.parse_args()


def _dedupe_texts(values: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        cleaned = clean_counterfactual_text(value)
        norm = normalize_text(cleaned)
        if cleaned and norm not in seen:
            seen.add(norm)
            out.append(cleaned)
    return out


def build_generator(args):
    model, tokenizer, template_args = load_model_bundle(
        model_cfg_path=args.model_cfg,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        model_subfolder=args.model_subfolder,
        tokenizer_subfolder=args.tokenizer_subfolder,
    )
    device = select_device(args.device)
    move_to_device = getattr(model, "hf_device_map", None) is None
    if move_to_device:
        model = model.to(device)
    model.eval()
    log(f"Loaded model on device={device}")

    def generate(question: str) -> list[str]:
        prompt = (
            "Question: {question}\n"
            "Provide short likely answers.\n"
            "Output only answers, one per line."
        ).format(question=question)

        encoded = preprocess_chat_instance(
            tokenizer=tokenizer,
            template_config=template_args,
            prompt_msgs=[prompt],
            response_msgs=[""],
            max_length=args.max_length,
            predict_with_generate=True,
        )
        input_ids = encoded["input_ids"].unsqueeze(0)
        attention_mask = encoded["attention_mask"].unsqueeze(0)
        if move_to_device:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        do_sample = float(args.temperature) > 0.0
        num_return_sequences = max(1, int(args.num_return_sequences))
        num_beams = max(int(args.num_beams), num_return_sequences)
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "max_new_tokens": max(1, int(args.max_new_tokens)),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(args.temperature)
            gen_kwargs["top_p"] = float(args.top_p)
            gen_kwargs["num_beams"] = 1
        else:
            gen_kwargs["num_beams"] = num_beams

        with torch.inference_mode():
            outputs = model.generate(**gen_kwargs)

        candidates: list[str] = []
        for sequence in outputs:
            generated = sequence[input_ids.shape[-1] :]
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
            candidates.extend(decoded.splitlines())
        return _dedupe_texts(candidates)

    return generate


def _is_valid_belief(
    belief: str,
    *,
    answer: str,
    alternate: str,
    require_short_answer: bool,
    max_alt_length_chars: int,
) -> bool:
    cleaned = clean_counterfactual_text(belief)
    if not cleaned:
        return False
    if normalize_text(cleaned) in {normalize_text(answer), normalize_text(alternate)}:
        return False
    reason = counterfactual_invalid_reason(
        cleaned,
        answer,
        reject_gold_substring=False,
        max_overlap_ratio=None,
        require_short_answer=require_short_answer,
        max_alt_length_chars=max_alt_length_chars,
    )
    return reason is None


def main():
    args = parse_args()
    rows = read_jsonl(args.input_path)
    if args.max_examples and int(args.max_examples) > 0:
        rows = rows[: int(args.max_examples)]

    generate = build_generator(args)
    out_rows: list[dict[str, Any]] = []
    filled = 0
    skipped = 0

    row_iter = tqdm(rows, total=len(rows), desc="belief_bank", dynamic_ncols=True)
    for row_no, row in enumerate(row_iter, start=1):
        try:
            updated = dict(row)
            answer = str(updated.get(args.answer_key, ""))
            alternate = str(updated.get(args.alternate_key, ""))
            question = str(updated[args.question_key])

            if args.skip_existing and str(updated.get(args.belief_key, "")).strip():
                skipped += 1
                out_rows.append(updated)
                continue

            candidates = generate(question)
            valid_candidates = [
                cand
                for cand in candidates
                if _is_valid_belief(
                    cand,
                    answer=answer,
                    alternate=alternate,
                    require_short_answer=bool(args.require_short_answer),
                    max_alt_length_chars=int(args.max_alt_length_chars),
                )
            ]

            updated[args.belief_candidates_key] = valid_candidates
            updated[args.belief_key] = valid_candidates[0] if valid_candidates else ""
            updated["belief_source"] = "base_model_generate"
            updated["belief_model_path"] = args.model_path or args.model_cfg
            updated["belief_prompt_version"] = args.prompt_version
            updated["belief_num_candidates"] = len(valid_candidates)
            if updated[args.belief_key]:
                filled += 1

            out_rows.append(updated)
        except Exception as exc:
            raise RuntimeError(
                f"Failed processing row_no={row_no} index={row.get('index', '<missing>')}"
            ) from exc

    save_jsonl(out_rows, args.output_path)
    log(
        "Done. "
        f"rows_in={len(rows)} rows_out={len(out_rows)} "
        f"belief_filled={filled} belief_skipped={skipped}"
    )


if __name__ == "__main__":
    main()
