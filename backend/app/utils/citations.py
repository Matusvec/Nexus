"""Citation verification utilities for proposal rationale text.

Verifies that [Quote: "..."] citations in LLM-generated rationale
actually correspond to real quotes from cluster member problem mentions.
"""

import logging
import re

from thefuzz import fuzz

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r'\[Quote:\s*"([^"]+)"\]', re.IGNORECASE)


def verify_rationale_citations(
    rationale: str,
    member_quotes: list[str],
    threshold: float = 0.85,
) -> tuple[str, list[dict]]:
    """
    Verify [Quote: "..."] citations in rationale against actual member quotes.

    Returns:
        - cleaned_rationale: rationale with unverifiable citations removed
        - verified_citations: list of {quote_text, matched_source, score}
    """
    citations = CITATION_PATTERN.findall(rationale)
    verified = []
    cleaned = rationale

    for cited_text in citations:
        best_score = 0
        best_source = None
        for source_quote in member_quotes:
            score = fuzz.partial_ratio(cited_text.lower(), source_quote.lower()) / 100
            if score > best_score:
                best_score = score
                best_source = source_quote

        if best_score >= threshold:
            verified.append({
                "quote_text": cited_text,
                "matched_source": best_source,
                "score": best_score,
            })
        else:
            # Strip the unverifiable citation from rationale
            cleaned = cleaned.replace(f'[Quote: "{cited_text}"]', cited_text)
            logger.warning(
                "Stripped unverifiable citation: %.50s... (best score: %.2f)",
                cited_text, best_score,
            )

    return cleaned, verified
