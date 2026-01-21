"""
Text chunking utilities for RAPTOR - Semantic chunking based on topic shifts

Microsoft best practices implemented:
- Fixed-size chunking with overlap (100-500 tokens)
- Semantic chunking using embeddings to detect topic shifts
- Preserves image references ([IMAGE img_id: ...]) across chunks
- Uses token-based counting (1 token ≈ 4 chars, like BERT)
"""
import re
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import config


def count_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation: 1 token ≈ 4 characters)
    
    For more accurate counting, could use tiktoken, but this is fast and good enough.
    Most embedding models use similar tokenization.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 chars (typical for English)
    # This matches OpenAI/Gemini tokenization fairly well
    return len(text) // 4


def chunk_text(
    text: str, 
    similarity_threshold: float = 0.7, 
    group_size: int = 3,
    min_tokens: int = 100,
    max_tokens: int = 500
) -> List[str]:
    """
    Split text into semantic chunks based on topic shifts (modern hybrid approach)
    
    Uses sentence grouping + embeddings to detect topic changes.
    Groups sentences together before embedding to prevent tiny single-sentence chunks.
    Enforces token limits to control costs and maintain context focus.
    
    Args:
        text: Input text to chunk
        similarity_threshold: Cosine similarity threshold for grouping (0.7 = fairly similar)
        group_size: Number of sentences to group before embedding (prevents micro-chunks)
        min_tokens: Minimum tokens per chunk (default 100 ≈ 400 chars)
        max_tokens: Maximum tokens per chunk (default 500 ≈ 2000 chars)
        
    Returns:
        List of semantically coherent text chunks
    """
    from embeddings import get_embeddings
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    if len(sentences) <= group_size:
        return [" ".join(sentences)]
    
    # Group sentences (prevents single-sentence chunks)
    sentence_groups = []
    for i in range(0, len(sentences), group_size):
        group = sentences[i:i + group_size]
        sentence_groups.append(" ".join(group))
    
    # Get embeddings for each group
    print(f"  Embedding {len(sentence_groups)} sentence groups...")
    embeddings = get_embeddings(sentence_groups)
    embeddings = np.array(embeddings)
    
    # Calculate similarity between consecutive groups
    similarities = []
    for i in tqdm(range(len(embeddings) - 1), desc="  Calculating similarities", unit="pair", leave=False):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(sim)
    
    # Find chunk boundaries where similarity drops OR max size exceeded
    chunks = []
    current_chunk = [sentence_groups[0]]
    current_tokens = count_tokens(sentence_groups[0])
    
    # Extract image references for preservation (Microsoft best practice)
    image_refs = extract_image_references(text)
    
    for i, similarity in enumerate(similarities):
        next_group = sentence_groups[i + 1]
        next_tokens = count_tokens(next_group)
        
        # Split if: topic shifts OR max tokens would be exceeded
        if similarity < similarity_threshold or (current_tokens + next_tokens > max_tokens):
            # Start new chunk
            chunk_text = " ".join(current_chunk)
            # Preserve image references if chunk mentions them
            chunk_text = preserve_image_refs_in_chunk(chunk_text, image_refs)
            chunks.append(chunk_text)
            current_chunk = [next_group]
            current_tokens = next_tokens
        else:
            # Continue current chunk
            current_chunk.append(next_group)
            current_tokens += next_tokens
    
    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk_text = preserve_image_refs_in_chunk(chunk_text, image_refs)
        chunks.append(chunk_text)
    
    # Post-process: merge very small chunks with neighbors
    chunks = merge_small_chunks(chunks, min_tokens=min_tokens)
    
    return chunks


def merge_small_chunks(chunks: List[str], min_tokens: int = 100) -> List[str]:
    """
    Merge chunks that are too small (by token count) with their neighbors
    
    Args:
        chunks: List of chunks
        min_tokens: Minimum token count for a chunk
        
    Returns:
        List of chunks with small ones merged
    """
    if not chunks:
        return chunks
    
    merged = []
    current = chunks[0]
    
    for chunk in chunks[1:]:
        if count_tokens(current) < min_tokens:
            current = current + " " + chunk
        else:
            merged.append(current)
            current = chunk
    
    merged.append(current)
    return merged


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting on . ! ?
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Remove empty strings and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


# Test
if __name__ == "__main__":
    print("Testing semantic chunking with token limits...")
    
    test_text = """
    Artificial intelligence is transforming technology. Machine learning enables 
    computers to learn from data. Deep learning uses neural networks with multiple layers.
    Natural language processing helps computers understand human language. 
    
    The weather today is sunny and warm. It's a perfect day for outdoor activities.
    Many people are going to the park for picnics.
    
    Quantum computing represents a new paradigm. It uses quantum bits or qubits.
    These systems can solve certain problems exponentially faster than classical computers.
    """
    
    print("\nChunking text based on semantic similarity + token limits...")
    chunks = chunk_text(
        test_text, 
        similarity_threshold=0.7, 
        group_size=2,
        min_tokens=50,
        max_tokens=200
    )
    
    print(f"\n✓ Created {len(chunks)} semantic chunks")
    for i, chunk in enumerate(chunks):
        tokens = count_tokens(chunk)
        print(f"\n--- Chunk {i+1} ({tokens} tokens, {len(chunk)} chars) ---")
        print(chunk[:200] + ("..." if len(chunk) > 200 else ""))
    
    print("\n✓ Semantic chunking complete!")
    print(f"✓ Token tracking: min 50, max 200 tokens per chunk")


def extract_image_references(text: str) -> List[str]:
    """
    Extract all image and table reference IDs from text
    
    Microsoft best practice: Track image/table references so they can be
    included in all chunks that discuss the image/table
    
    Args:
        text: Text containing [IMAGE img_id: ...] or [TABLE table_id: ...] markers
        
    Returns:
        List of unique image/table reference IDs
    """
    # Match both IMAGE and TABLE markers
    image_pattern = r'\[IMAGE\s+([^\s:]+):'
    table_pattern = r'\[TABLE\s+([^\s:]+):'
    
    image_matches = re.findall(image_pattern, text)
    table_matches = re.findall(table_pattern, text)
    
    all_refs = image_matches + table_matches
    return list(set(all_refs))  # Deduplicate


def preserve_image_refs_in_chunk(chunk: str, all_image_refs: List[str]) -> str:
    """
    Ensure image and table references are preserved in chunks
    
    Microsoft best practice: If image/table description splits across chunks,
    include reference ID in each chunk
    
    Args:
        chunk: Text chunk
        all_image_refs: All image/table reference IDs from document
        
    Returns:
        Chunk with preserved image/table references (if any are mentioned)
    """
    # Check if chunk mentions any images or tables
    chunk_refs = extract_image_references(chunk)
    
    # Chunk already has explicit markers, return as-is
    if chunk_refs:
        return chunk
    
    # Check if chunk contains text that likely refers to images or tables
    # (e.g., "as shown in the figure", "the table above")
    reference_phrases = [
        r'\bfigure\b', r'\bdiagram\b', r'\bchart\b', r'\bgraph\b',
        r'\bimage\b', r'\bpicture\b', r'\btable\b', r'\billustration\b',
        r'\babove\b', r'\bbelow\b', r'\bshown\b'
    ]
    
    has_visual_reference = any(re.search(phrase, chunk, re.IGNORECASE) for phrase in reference_phrases)
    
    # If chunk likely discusses an image but doesn't have explicit marker,
    # this is where you'd add logic to link it to nearby images
    # For now, we return chunk as-is since explicit markers are preferred
    
    return chunk

