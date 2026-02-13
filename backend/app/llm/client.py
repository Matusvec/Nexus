import json
from typing import Any

import google.generativeai as genai

from app.config import settings


class GeminiClient:
    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)

    def generate_json(self, prompt: str) -> dict[str, Any]:
        response = self.model.generate_content(prompt)
        return _parse_json_response(response.text)


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
