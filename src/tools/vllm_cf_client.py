from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
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
        # Qwen3 enables thinking traces by default unless the chat template is
        # told otherwise. Disable them here so counterfactual generation stays a
        # short JSON-only answer span.
        self.extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "counterfactual_response",
                "schema": CounterfactualResponse.model_json_schema(),
            },
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
            "5. Return valid JSON only."
        )
        user = f"Question: {question}\nGold answer: {answer}\n"
        if candidate_answers:
            user += "Candidate alternatives:\n"
            for idx, candidate in enumerate(candidate_answers, start=1):
                user += f"{idx}. {candidate}\n"
            user += (
                "Select the best candidate or minimally rewrite one candidate for fluency. "
                "Do not invent a long explanation."
            )
        else:
            user += "Generate one plausible alternative answer of the same answer type."
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
            response_format=self.response_format,
            extra_body=self.extra_body,
        )
        content = response.choices[0].message.content
        payload = json.loads(_strip_json_fence(content))
        return CounterfactualResponse.model_validate(payload).model_dump()

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
