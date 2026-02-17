import re


def _split_sentences(text: str) -> list[dict]:
    """Split text into sentences, tracking character offsets."""
    # Match sentence-ending punctuation followed by whitespace or end-of-string.
    # Handles common abbreviations poorly, but good enough for transcripts/notes.
    pattern = r'(?<=[.!?])\s+'
    sentences = []
    prev_end = 0

    for match in re.finditer(pattern, text):
        end = match.start()  # end of the sentence (after punctuation)
        sentence_text = text[prev_end:match.end()].strip()
        if sentence_text:
            sentences.append({
                "text": sentence_text,
                "start": prev_end,
                "end": match.end(),
            })
        prev_end = match.end()

    # Capture the last sentence (no trailing split)
    remaining = text[prev_end:].strip()
    if remaining:
        sentences.append({
            "text": remaining,
            "start": prev_end,
            "end": len(text),
        })

    return sentences


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters (English-heavy text).
    This will be less accurate for CJK text, code, or URL-heavy content."""
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[dict]:
    """Split text into chunks at sentence boundaries, respecting token limits.

    - Never splits mid-sentence
    - Overlaps by repeating the last few sentences of the previous chunk
    - Tracks character offsets for provenance
    - Returns estimated token count per chunk
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    chunk_index = 0
    i = 0  # current sentence index

    while i < len(sentences):
        chunk_sentences = []
        chunk_token_count = 0

        # Accumulate sentences until we approach the token limit
        j = i
        while j < len(sentences):
            sent_tokens = _estimate_tokens(sentences[j]["text"])

            # If adding this sentence exceeds the limit and we already have
            # at least one sentence, stop here.
            if chunk_token_count + sent_tokens > max_tokens and chunk_sentences:
                break

            chunk_sentences.append(sentences[j])
            chunk_token_count += sent_tokens
            j += 1

        # Build the chunk
        start_offset = chunk_sentences[0]["start"]
        end_offset = chunk_sentences[-1]["end"]
        # m7 fix: use original text slice for accurate chunk content
        # instead of joining sentences (which may alter whitespace)
        chunk_text_str = text[start_offset:end_offset]

        chunks.append({
            "index": chunk_index,
            "text": chunk_text_str,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "token_count": chunk_token_count,
        })
        chunk_index += 1

        # If we consumed all sentences, we're done
        if j >= len(sentences):
            break

        # Calculate overlap: walk backwards from the end of this chunk
        # to find how many sentences fit within overlap_tokens
        overlap_count = 0
        overlap_sum = 0
        for k in range(len(chunk_sentences) - 1, -1, -1):
            sent_tokens = _estimate_tokens(chunk_sentences[k]["text"])
            if overlap_sum + sent_tokens > overlap_tokens:
                break
            overlap_sum += sent_tokens
            overlap_count += 1

        # Next chunk starts overlap_count sentences back from where this one ended
        # Safety: always advance at least one sentence to avoid infinite loop
        i = max(j - overlap_count, i + 1)

    return chunks
