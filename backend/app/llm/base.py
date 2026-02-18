"""Abstract interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate_json(self, prompt: str, prompt_version: str | None = None) -> dict[str, Any]:
        """Generate structured JSON from a prompt."""
        ...

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate a text embedding vector."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
