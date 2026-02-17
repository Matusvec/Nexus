import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import google.generativeai as genai

from app.config import settings

logger = logging.getLogger(__name__)
_client = None


class LLMCallRecord:
    """Tracks metadata for a single LLM call."""

    __slots__ = (
        "model", "operation", "prompt_version", "input_tokens",
        "output_tokens", "latency_ms", "cost_usd", "timestamp", "error",
    )

    def __init__(
        self,
        model: str,
        operation: str,
        prompt_version: str | None = None,
    ) -> None:
        self.model = model
        self.operation = operation
        self.prompt_version = prompt_version
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.latency_ms: float = 0.0
        self.cost_usd: float = 0.0
        self.timestamp: datetime = datetime.now(timezone.utc)
        self.error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "operation": self.operation,
            "prompt_version": self.prompt_version,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


# In-memory cost accumulator (also persisted to DB via _persist_record)
_call_log: list[LLMCallRecord] = []


async def _persist_record(record: LLMCallRecord) -> None:
    """Persist an LLM call record to the database (fire-and-forget)."""
    try:
        from app.database import AsyncSessionLocal
        from app.models.jobs import LLMCallLog

        async with AsyncSessionLocal() as session:
            row = LLMCallLog(
                model=record.model,
                operation=record.operation,
                prompt_version=record.prompt_version,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                latency_ms=record.latency_ms,
                cost_usd=record.cost_usd,
                error=record.error,
            )
            session.add(row)
            await session.commit()
    except Exception:  # noqa: BLE001
        logger.debug("Failed to persist LLM call record to DB", exc_info=True)


def _record_and_persist(record: LLMCallRecord) -> None:
    """Append to in-memory log and schedule DB persistence."""
    _call_log.append(record)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_persist_record(record))
    except RuntimeError:
        # No running event loop (e.g. during tests) — skip DB persistence
        pass


def get_call_log() -> list[dict[str, Any]]:
    """Return the accumulated LLM call records."""
    return [r.to_dict() for r in _call_log]


def get_cost_summary() -> dict[str, Any]:
    """Return total and per-model cost summary."""
    total_cost = sum(r.cost_usd for r in _call_log)
    total_input = sum(r.input_tokens for r in _call_log)
    total_output = sum(r.output_tokens for r in _call_log)
    by_model: dict[str, float] = {}
    for r in _call_log:
        by_model[r.model] = by_model.get(r.model, 0) + r.cost_usd
    return {
        "total_calls": len(_call_log),
        "total_cost_usd": round(total_cost, 6),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "by_model": by_model,
    }


# Approximate pricing (USD per 1K tokens) for Gemini models
_PRICING = {
    "gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
    "text-embedding-004": {"input": 0.000025, "output": 0.0},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _PRICING.get(model, {"input": 0.0001, "output": 0.0004})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000


class GeminiClient:
    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    def generate_json(
        self, prompt: str, prompt_version: str | None = None
    ) -> dict[str, Any]:
        record = LLMCallRecord(
            model=settings.gemini_model,
            operation="generate_json",
            prompt_version=prompt_version,
        )
        start = time.perf_counter()
        try:
            response = self.model.generate_content(prompt)
            record.latency_ms = (time.perf_counter() - start) * 1000

            # Extract token usage if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                record.input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                record.output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            else:
                # Rough estimate fallback
                record.input_tokens = len(prompt) // 4
                record.output_tokens = len(response.text) // 4 if response.text else 0

            record.cost_usd = _estimate_cost(
                record.model, record.input_tokens, record.output_tokens
            )
            _record_and_persist(record)
            logger.debug(
                "LLM call: model=%s tokens_in=%d tokens_out=%d cost=$%.6f latency=%.0fms",
                record.model, record.input_tokens, record.output_tokens,
                record.cost_usd, record.latency_ms,
            )
            return _parse_json_response(response.text)
        except Exception as exc:
            record.latency_ms = (time.perf_counter() - start) * 1000
            record.error = str(exc)
            _record_and_persist(record)
            raise

    def embed_text(self, text: str) -> list[float]:
        record = LLMCallRecord(
            model=settings.gemini_embedding_model,
            operation="embed_text",
        )
        start = time.perf_counter()
        try:
            response = genai.embed_content(
                model=settings.gemini_embedding_model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            )
            record.latency_ms = (time.perf_counter() - start) * 1000
            record.input_tokens = len(text) // 4
            record.cost_usd = _estimate_cost(
                record.model, record.input_tokens, 0
            )
            _record_and_persist(record)

            embedding = response.get("embedding")
            if not embedding:
                raise ValueError("Embedding response missing embedding vector.")
            return embedding
        except Exception as exc:
            record.latency_ms = (time.perf_counter() - start) * 1000
            record.error = str(exc)
            _record_and_persist(record)
            raise


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain JSON object.")
    return json.loads(cleaned[start : end + 1])


def get_client() -> "GeminiClient":
    global _client
    if _client is None:
        _client = GeminiClient()
        logger.info("Gemini client initialized")
    return _client
