"""OpenAI LLM provider — alternative to Gemini.

Supports structured JSON generation via gpt-4o and embeddings via
text-embedding-3-small. Embedding dimensions (1536) differ from
Gemini (768) — switching providers requires re-embedding all problems.
"""

import json
import logging
import time
from typing import Any

from app.llm.base import LLMProvider
from app.llm.client import LLMCallRecord, _record_and_persist

logger = logging.getLogger(__name__)


class OpenAIClient(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def generate_json(self, prompt: str, prompt_version: str | None = None) -> dict[str, Any]:
        record = LLMCallRecord(
            model=self._model,
            operation="generate_json",
            prompt_version=prompt_version,
        )
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            record.latency_ms = (time.perf_counter() - start) * 1000
            if response.usage:
                record.input_tokens = response.usage.prompt_tokens
                record.output_tokens = response.usage.completion_tokens
            _record_and_persist(record)
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as exc:
            record.latency_ms = (time.perf_counter() - start) * 1000
            record.error = str(exc)
            _record_and_persist(record)
            raise

    def embed_text(self, text: str) -> list[float]:
        record = LLMCallRecord(
            model="text-embedding-3-small",
            operation="embed_text",
        )
        start = time.perf_counter()
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            record.latency_ms = (time.perf_counter() - start) * 1000
            record.input_tokens = len(text) // 4
            _record_and_persist(record)
            return response.data[0].embedding
        except Exception as exc:
            record.latency_ms = (time.perf_counter() - start) * 1000
            record.error = str(exc)
            _record_and_persist(record)
            raise
