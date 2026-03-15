from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, Iterable, List, Optional, Sequence

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class CounterfactualResponse(BaseModel):
    alternate: str = Field(min_length=1, max_length=128)
    same_relation: bool = True
    answer_type: str = "unknown"


def _strip_json_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


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

    def __post_init__(self) -> None:
        self.use_structured_outputs = (
            os.environ.get("VLLM_USE_STRUCTURED_OUTPUTS", "0").strip().lower()
            in {"1", "true", "yes"}
        )
        # Qwen3 enables thinking traces by default unless the chat template is
        # told otherwise. Disable them here so counterfactual generation stays a
        # short JSON-only answer span.
        self.schema = CounterfactualResponse.model_json_schema()
        self.extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
        if self.use_structured_outputs:
            # vLLM documents structured outputs for the OpenAI-compatible server
            # through extra_body. Keep the schema here rather than relying on
            # response_format mapping inside the compatibility layer.
            self.extra_body["structured_outputs"] = {
                "json": self.schema,
            }

    def _make_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def build_messages(
        self,
        *,
        question: str,
        answer: str,
        candidate_answers: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, str]]:
        system = (
            "You generate a short plausible but incorrect alternative answer for "
            "a factual unlearning dataset.\n"
            "Rules:\n"
            "1. Keep the same semantic relation as the gold answer.\n"
            "2. Output a short answer span, not a sentence or explanation.\n"
            "3. Never mention, quote, negate, or compare against the gold answer.\n"
            "4. Do not add prefixes like Alternative answer or Wrong answer.\n"
        )
        if self.use_structured_outputs:
            system += "5. Return valid JSON only."
        else:
            system += (
                "5. Return only the alternative answer span.\n"
                "6. Do not output JSON, bullets, labels, or explanation."
            )
        user = f"Question: {question}\nGold answer: {answer}\n"
        if candidate_answers:
            user += "Candidate alternatives:\n"
            for idx, candidate in enumerate(candidate_answers, start=1):
                user += f"{idx}. {candidate}\n"
            user += (
                "Select the best candidate or minimally rewrite one candidate for fluency. "
                "Return only the final alternative answer span."
            )
        else:
            user += (
                "Generate one plausible alternative answer of the same answer type. "
                "Return only the answer span."
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

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
            alternate = _strip_json_fence(content).splitlines()[0].strip() if content else ""
            return {
                "alternate": alternate,
                "same_relation": True,
                "answer_type": "plain_text",
            }
        return self._parse_payload(content)

    def _parse_payload(self, content: Any) -> Dict[str, Any]:
        stripped = _strip_json_fence(content)
        if not stripped:
            return {
                "alternate": "",
                "same_relation": False,
                "answer_type": "empty_response",
            }

        try:
            payload = json.loads(stripped)
        except JSONDecodeError:
            # Some backends may fail structured decoding but still emit a short
            # plain-text span. Preserve that instead of aborting the whole batch.
            fallback_alternate = str(stripped).splitlines()[0].strip()[:128]
            if fallback_alternate and not fallback_alternate.startswith("{"):
                return {
                    "alternate": fallback_alternate,
                    "same_relation": True,
                    "answer_type": "free_text_fallback",
                }
            return {
                "alternate": "",
                "same_relation": False,
                "answer_type": "invalid_json",
            }

        if not isinstance(payload, dict):
            return {
                "alternate": "",
                "same_relation": False,
                "answer_type": "invalid_payload",
            }

        try:
            return CounterfactualResponse.model_validate(payload).model_dump()
        except Exception:
            fallback_alternate = str(payload.get("alternate", "")).strip()[:128]
            return {
                "alternate": fallback_alternate,
                "same_relation": bool(payload.get("same_relation", False)),
                "answer_type": str(payload.get("answer_type", "invalid_schema")),
            }

    async def many(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    def many_sync(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return asyncio.run(self.many(rows))


def chunked(values: Sequence[Dict[str, Any]], chunk_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]
