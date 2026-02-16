"""Golden-set evaluation harness for extraction regression testing.

Usage (from backend/):
    python -m app.eval.harness --golden-set app/eval/golden_set.json

The golden set is a JSON file with this structure:
[
  {
    "chunk_text": "...",
    "expected_problems": [
      {
        "problem_statement": "...",
        "severity": "high",
        "quote_text": "..."
      }
    ]
  }
]

The harness runs extraction on each chunk, compares results to the golden set,
and reports precision, recall, and F1 at the problem-statement level.
"""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from thefuzz import fuzz

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 80  # fuzzy match threshold for problem statements


@dataclass
class EvalResult:
    """Result of evaluating one golden-set entry."""
    chunk_index: int
    expected_count: int
    extracted_count: int
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


def match_problems(
    expected: list[dict],
    extracted: list[dict],
    threshold: int = MATCH_THRESHOLD,
) -> EvalResult:
    """Match extracted problems to expected ones using fuzzy matching."""
    result = EvalResult(
        chunk_index=0,
        expected_count=len(expected),
        extracted_count=len(extracted),
    )

    matched_expected: set[int] = set()
    matched_extracted: set[int] = set()

    for i, exp in enumerate(expected):
        best_score = 0
        best_j = -1
        for j, ext in enumerate(extracted):
            if j in matched_extracted:
                continue
            score = fuzz.ratio(
                exp["problem_statement"].lower(),
                ext["problem_statement"].lower(),
            )
            if score > best_score:
                best_score = score
                best_j = j

        if best_score >= threshold and best_j >= 0:
            result.true_positives += 1
            matched_expected.add(i)
            matched_extracted.add(best_j)
            result.matched_pairs.append(
                (exp["problem_statement"], extracted[best_j]["problem_statement"])
            )

    result.false_negatives = len(expected) - len(matched_expected)
    result.false_positives = len(extracted) - len(matched_extracted)
    return result


async def run_eval(golden_path: str) -> list[EvalResult]:
    """Run extraction on each golden-set entry and return results."""
    from app.llm.client import get_client
    from app.schemas.problems import LLMProblemsResponse
    from app.services.extraction_service import _build_prompt

    golden_set = json.loads(Path(golden_path).read_text())
    client = get_client()
    results = []

    for idx, entry in enumerate(golden_set):
        chunk_text = entry["chunk_text"]
        expected = entry["expected_problems"]

        prompt = _build_prompt(chunk_text)
        try:
            raw = await asyncio.to_thread(client.generate_json, prompt)
            parsed = LLMProblemsResponse.model_validate(raw)
            extracted = [p.model_dump() for p in parsed.problems]
        except Exception as exc:
            logger.error("Extraction failed for chunk %d: %s", idx, exc)
            extracted = []

        result = match_problems(expected, extracted)
        result.chunk_index = idx
        results.append(result)

        logger.info(
            "Chunk %d: P=%.2f R=%.2f F1=%.2f (TP=%d FP=%d FN=%d)",
            idx, result.precision, result.recall, result.f1,
            result.true_positives, result.false_positives, result.false_negatives,
        )

    # Aggregate
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)
    agg_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    agg_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    agg_f1 = 2 * agg_p * agg_r / (agg_p + agg_r) if (agg_p + agg_r) > 0 else 0

    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY ({len(results)} chunks)")
    print(f"  Precision: {agg_p:.3f}")
    print(f"  Recall:    {agg_r:.3f}")
    print(f"  F1:        {agg_f1:.3f}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"{'='*60}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction eval harness")
    parser.add_argument(
        "--golden-set", required=True, help="Path to golden set JSON file"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_eval(args.golden_set))


if __name__ == "__main__":
    main()
