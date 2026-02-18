"""Validate golden_set.json structural integrity (no LLM calls needed)."""

import json
from pathlib import Path

import pytest

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"


def test_golden_set_structure():
    golden = json.loads(GOLDEN_SET_PATH.read_text())
    assert len(golden) >= 20, f"Golden set too small: {len(golden)} entries (need ≥20)"
    for i, entry in enumerate(golden):
        assert "chunk_text" in entry, f"Entry {i}: missing chunk_text"
        assert "expected_problems" in entry, f"Entry {i}: missing expected_problems"
        for j, prob in enumerate(entry["expected_problems"]):
            assert prob["quote_text"] in entry["chunk_text"], \
                f"Entry {i}, problem {j}: quote_text not found in chunk_text"
            assert prob["severity"] in ("critical", "high", "medium", "low"), \
                f"Entry {i}, problem {j}: invalid severity '{prob['severity']}'"
