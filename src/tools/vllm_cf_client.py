from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, Iterable, Optional, Sequence

from openai import AsyncOpenAI


def _strip_json_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


def _normalize_alternate(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    return first_line[:128]


def _normalize_alternates(value: Any) -> list[str]:
    if isinstance(value, list):
        out = [_normalize_alternate(item) for item in value]
        return [item for item in out if item]
    alternate = _normalize_alternate(value)
    return [alternate] if alternate else []


def _normalize_scores(value: Any, expected_length: int) -> list[Any]:
    if not isinstance(value, list):
        return []

    normalized: list[Any] = []
    for raw_score in value[:expected_length]:
        try:
            normalized.append(float(raw_score))
        except Exception:
            normalized.append(None)
    if len(normalized) < expected_length:
        normalized.extend([None] * (expected_length - len(normalized)))
    return normalized


def _normalize_optional_text_list(value: Any, expected_length: int) -> list[Any]:
    if not isinstance(value, list):
        return []

    normalized: list[Any] = []
    for raw_value in value[:expected_length]:
        text = str(raw_value or "").strip()
        normalized.append(text or None)
    if len(normalized) < expected_length:
        normalized.extend([None] * (expected_length - len(normalized)))
    return normalized


@dataclass
class VLLMCFGenerator:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 32
    concurrency: int = 64
    timeout: float = 300.0
    prompt_family: str = "default"
    num_alternates: int = 1

    def __post_init__(self) -> None:
        self.use_structured_outputs = (
            os.environ.get("VLLM_USE_STRUCTURED_OUTPUTS", "0").strip().lower()
            in {"1", "true", "yes"}
        )
        candidate_limit = max(1, int(self.num_alternates))
        if candidate_limit > 1 and not self.use_structured_outputs:
            raise ValueError(
                "Multi-alternate vLLM generation requires VLLM_USE_STRUCTURED_OUTPUTS=1."
            )
        self.schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "alternates": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    "minItems": 1,
                    "maxItems": candidate_limit,
                },
                "scores": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "relation_scores": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "shared_fact_scores": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "candidate_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "same_relation": {"type": "boolean"},
                "answer_type": {"type": "string"},
            },
            "required": ["alternates", "same_relation", "answer_type"],
        }
        # Qwen3 enables thinking traces by default unless the chat template is
        # told otherwise. Disable them here so counterfactual generation stays a
        # short answer span or short JSON payload.
        self.extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
        if self.use_structured_outputs:
            self.extra_body["structured_outputs"] = {
                "json": self.schema,
            }

    def _make_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def _system_rules(self) -> str:
        base_rules = [
            "You generate short plausible but incorrect alternative answers for a factual unlearning dataset.",
            "Rules:",
            "1. Keep the same answer type as the gold answer.",
            "2. Output short answer spans, not explanations or full sentences.",
            "3. Never mention, quote, negate, or compare against the gold answer unless explicitly forced by the prompt.",
            "4. Do not add prefixes like Alternative answer or Wrong answer.",
        ]
        prompt_family = str(self.prompt_family or "default")
        if prompt_family == "strict_short":
            base_rules.append(
                "5. Prefer compact spans that can be written on one line."
            )
        elif prompt_family == "duet_relation_safe":
            base_rules.append(
                "5. Preserve the same semantic relation as the gold answer and avoid paraphrasing the gold answer."
            )
        elif prompt_family == "rwku_shared_fact_safe":
            base_rules.append(
                "5. Change only the target answer and avoid altering unrelated shared facts."
            )
        else:
            base_rules.append(
                "5. Keep the answer plausible and relation-consistent when possible."
            )
        return "\n".join(base_rules)

    def build_messages(
        self,
        *,
        question: str,
        answer: str,
        candidate_answers: Optional[Sequence[str]] = None,
    ) -> list[Dict[str, str]]:
        system = self._system_rules()
        if self.use_structured_outputs:
            system += (
                "\n6. Return valid JSON only with keys `alternates`, optional `scores`, "
                "`relation_scores`, `shared_fact_scores`, optional `candidate_sources`, "
                "`same_relation`, and `answer_type`."
            )
        else:
            system += (
                "\n6. Return only one alternative answer span.\n"
                "7. Do not output JSON, bullets, labels, or explanation."
            )

        user = f"Question: {question}\nGold answer: {answer}\n"
        prompt_family = str(self.prompt_family or "default")
        if candidate_answers:
            user += "Candidate alternatives:\n"
            for idx, candidate in enumerate(candidate_answers, start=1):
                user += f"{idx}. {candidate}\n"
            if self.use_structured_outputs:
                user += (
                    f"Select up to {max(1, int(self.num_alternates))} best short alternatives from the candidate list "
                    "or minimal rewrites of that list. If you provide `scores`, `relation_scores`, "
                    "`shared_fact_scores`, or `candidate_sources`, align them with `alternates`."
                )
            else:
                user += (
                    "Select the best candidate or minimally rewrite one candidate for fluency. "
                    "Return only the final alternative answer span."
                )
        else:
            if self.use_structured_outputs:
                user += (
                    f"Generate up to {max(1, int(self.num_alternates))} short incorrect alternatives with the same answer type. "
                )
            else:
                user += "Generate one plausible alternative answer of the same answer type. "

            if prompt_family == "duet_relation_safe":
                user += "Preserve the same relation and do not repeat or paraphrase the gold answer. "
            elif prompt_family == "rwku_shared_fact_safe":
                user += "Change only the target answer and preserve unrelated shared facts. "

            user += (
                "Return only the requested answer span or JSON payload."
                if self.use_structured_outputs
                else "Return only the answer span."
            )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _normalize_response(
        self,
        *,
        alternates: Sequence[Any],
        scores: Sequence[Any] | None = None,
        relation_scores: Sequence[Any] | None = None,
        shared_fact_scores: Sequence[Any] | None = None,
        candidate_sources: Sequence[Any] | None = None,
        same_relation: bool = True,
        answer_type: str = "unknown",
    ) -> Dict[str, Any]:
        normalized_alternates = _normalize_alternates(list(alternates))
        normalized_scores = _normalize_scores(scores, len(normalized_alternates))
        normalized_relation_scores = _normalize_scores(
            relation_scores,
            len(normalized_alternates),
        )
        normalized_shared_fact_scores = _normalize_scores(
            shared_fact_scores,
            len(normalized_alternates),
        )
        normalized_candidate_sources = _normalize_optional_text_list(
            candidate_sources,
            len(normalized_alternates),
        )
        alternate = normalized_alternates[0] if normalized_alternates else ""
        return {
            "alternate": alternate,
            "alternates": normalized_alternates,
            "scores": normalized_scores,
            "relation_scores": normalized_relation_scores,
            "shared_fact_scores": normalized_shared_fact_scores,
            "candidate_sources": normalized_candidate_sources,
            "same_relation": bool(same_relation),
            "answer_type": str(answer_type),
        }

    async def one(
        self,
        client: AsyncOpenAI,
        *,
        question: str,
        answer: str,
        candidate_answers: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        response = await client.chat.completions.create(
            model=self.model,
            messages=self.build_messages(
                question=question,
                answer=answer,
                candidate_answers=candidate_answers,
            ),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        content = response.choices[0].message.content
        if not self.use_structured_outputs:
            alternate = _normalize_alternate(_strip_json_fence(content)) if content else ""
            return self._normalize_response(
                alternates=[alternate] if alternate else [],
                same_relation=True,
                answer_type="plain_text",
            )
        return self._parse_payload(content)

    def _parse_payload(self, content: Any) -> Dict[str, Any]:
        stripped = _strip_json_fence(content)
        if not stripped:
            return self._normalize_response(
                alternates=[],
                same_relation=False,
                answer_type="empty_response",
            )

        try:
            payload = json.loads(stripped)
        except JSONDecodeError:
            fallback_alternate = _normalize_alternate(stripped)
            if fallback_alternate and not fallback_alternate.startswith("{"):
                return self._normalize_response(
                    alternates=[fallback_alternate],
                    same_relation=True,
                    answer_type="free_text_fallback",
                )
            return self._normalize_response(
                alternates=[],
                same_relation=False,
                answer_type="invalid_json",
            )

        if not isinstance(payload, dict):
            return self._normalize_response(
                alternates=[],
                same_relation=False,
                answer_type="invalid_payload",
            )

        alternates = payload.get("alternates")
        if alternates is None:
            alternates = payload.get("alternate", "")
        normalized_alternates = _normalize_alternates(alternates)
        return self._normalize_response(
            alternates=normalized_alternates,
            scores=payload.get("scores"),
            same_relation=bool(payload.get("same_relation", bool(normalized_alternates))),
            answer_type=str(payload.get("answer_type", "invalid_schema")),
            relation_scores=payload.get("relation_scores"),
            shared_fact_scores=payload.get("shared_fact_scores"),
            candidate_sources=payload.get("candidate_sources"),
        )

    async def many(self, rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, int(self.concurrency)))
        client = self._make_client()

        async def _bound(row: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.one(
                    client,
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    candidate_answers=row.get("candidate_answers"),
                )

        try:
            tasks = [_bound(row) for row in rows]
            return await asyncio.gather(*tasks)
        finally:
            await client.close()

    def many_sync(self, rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        return asyncio.run(self.many(rows))


def chunked(values: Sequence[Dict[str, Any]], chunk_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]
