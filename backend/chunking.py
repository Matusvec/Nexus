"""
Text chunking utilities for RAPTOR - Semantic chunking based on topic shifts

Industry best practices implemented:
- Semantic chunking using embeddings to detect topic shifts
- Token-based sizing (100-500 tokens per chunk)
- Chunk overlap for context preservation (Microsoft)
- Contextual embeddings (Anthropic's Contextual Retrieval)
- Separate handling of tables/images as standalone chunks
"""
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Import shared utilities
from utils import count_tokens, split_into_sentences


def chunk_text(
    text: str, 
    similarity_threshold: float = 0.7, 
    group_size: int = 3,
    min_tokens: int = 100,
    max_tokens: int = 500,
    overlap_tokens: int = 50
) -> List[str]:
    """
    Split text into semantic chunks based on topic shifts (modern hybrid approach)
    
    Uses sentence grouping + embeddings to detect topic changes.
    Includes overlap between chunks for better context preservation.
    
    Args:
        text: Input text to chunk
        similarity_threshold: Cosine similarity threshold for grouping (0.7 = fairly similar)
        group_size: Number of sentences to group before embedding (prevents micro-chunks)
        min_tokens: Minimum tokens per chunk (default 100 ≈ 400 chars)
        max_tokens: Maximum tokens per chunk (default 500 ≈ 2000 chars)
        overlap_tokens: Token overlap between consecutive chunks (default 50)
        
    Returns:
        List of semantically coherent text chunks with overlap
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
    raw_chunks = []
    current_chunk = [sentence_groups[0]]
    current_tokens = count_tokens(sentence_groups[0])
    
    for i, similarity in enumerate(similarities):
        next_group = sentence_groups[i + 1]
        next_tokens = count_tokens(next_group)
        
        # Split if: topic shifts OR max tokens would be exceeded
        if similarity < similarity_threshold or (current_tokens + next_tokens > max_tokens):
            # Start new chunk
            raw_chunks.append(" ".join(current_chunk))
            current_chunk = [next_group]
            current_tokens = next_tokens
        else:
            # Continue current chunk
            current_chunk.append(next_group)
            current_tokens += next_tokens
    
    # Add final chunk
    if current_chunk:
        raw_chunks.append(" ".join(current_chunk))
    
    # Post-process: merge very small chunks with neighbors
    raw_chunks = merge_small_chunks(raw_chunks, min_tokens=min_tokens)
    
    # Add overlap between chunks
    chunks = add_chunk_overlap(raw_chunks, overlap_tokens=overlap_tokens)
    
    return chunks


def add_chunk_overlap(chunks: List[str], overlap_tokens: int = 50) -> List[str]:
    """
    Add overlap between consecutive chunks for better context preservation
    
    Microsoft best practice: Overlapping chunks help preserve context
    that might be split at chunk boundaries.
    
    Args:
        chunks: List of non-overlapping chunks
        overlap_tokens: Number of tokens to overlap (default 50)
        
    Returns:
        List of chunks with overlap added
    """
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks
    
    overlapped = []
    overlap_chars = overlap_tokens * 4  # ~4 chars per token
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk: add beginning of next chunk as suffix
            if len(chunks) > 1:
                next_start = chunks[1][:overlap_chars]
                overlapped.append(chunk + " ... " + next_start)
            else:
                overlapped.append(chunk)
        elif i == len(chunks) - 1:
            # Last chunk: add end of previous chunk as prefix
            prev_end = chunks[i-1][-overlap_chars:]
            overlapped.append(prev_end + " ... " + chunk)
        else:
            # Middle chunks: add both prefix and suffix overlap
            prev_end = chunks[i-1][-overlap_chars:]
            next_start = chunks[i+1][:overlap_chars]
            overlapped.append(prev_end + " ... " + chunk + " ... " + next_start)
    
    return overlapped


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


def contextualize_chunks(
    chunks: List[str], 
    doc_summary: str, 
    doc_name: str,
    use_llm_context: bool = False
) -> List[Dict[str, str]]:
    """
    Add document-level context to each chunk (Anthropic's Contextual Retrieval)
    
    This prepends contextual information to each chunk before embedding,
    which improves retrieval accuracy by ~49% according to Anthropic's research.
    
    Args:
        chunks: List of text chunks
        doc_summary: Summary of the full document
        doc_name: Document filename
        use_llm_context: If True, use LLM to generate specific context per chunk
                        If False, use simple template (faster, cheaper)
        
    Returns:
        List of dicts with 'original', 'contextualized', and 'context' keys
    """
    from gemini_client import generate_chunk_context
    
    contextualized = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="  Adding context to chunks", unit="chunk")):
        if use_llm_context:
            # Use LLM to generate specific context (slower but better)
            context = generate_chunk_context(chunk, doc_summary, doc_name)
        else:
            # Use simple template (faster, still effective)
            context = f"From '{doc_name}': {doc_summary[:200]}..."
        
        contextualized.append({
            "original": chunk,
            "contextualized": f"{context}\n\n{chunk}",
            "context": context,
            "chunk_index": i
        })
    
    return contextualized


def process_structured_content(
    text_content: str,
    tables: List[Dict],
    images: List[Dict],
    doc_summary: str,
    doc_name: str,
    **chunk_kwargs
) -> List[Dict[str, str]]:
    """
    Process document content with tables/images as separate chunks
    
    Industry best practice: Tables and images should be their own chunks
    with context, not mixed into text chunks.
    
    Args:
        text_content: Plain text from document (without table/image markers)
        tables: List of table dicts with 'id', 'content', 'context'
        images: List of image dicts with 'id', 'description', 'context'
        doc_summary: Document summary for contextualization
        doc_name: Document filename
        **chunk_kwargs: Arguments passed to chunk_text()
        
    Returns:
        List of contextualized chunks (text + tables + images)
    """
    all_chunks = []
    
    # 1. Chunk the text content
    if text_content.strip():
        text_chunks = chunk_text(text_content, **chunk_kwargs)
        text_contextualized = contextualize_chunks(text_chunks, doc_summary, doc_name)
        
        for tc in text_contextualized:
            tc["content_type"] = "text"
        all_chunks.extend(text_contextualized)
    
    # 2. Add tables as separate chunks with context
    for table in tables:
        table_context = f"From '{doc_name}': {doc_summary[:150]}. This table appears in the document."
        table_chunk = {
            "original": table["content"],
            "contextualized": f"{table_context}\n\n{table['content']}",
            "context": table_context,
            "content_type": "table",
            "ref_id": table.get("id", ""),
            "chunk_index": len(all_chunks)
        }
        all_chunks.append(table_chunk)
    
    # 3. Add images as separate chunks with context  
    for image in images:
        image_context = f"From '{doc_name}': {doc_summary[:150]}. This image appears in the document."
        image_chunk = {
            "original": image["description"],
            "contextualized": f"{image_context}\n\n{image['description']}",
            "context": image_context,
            "content_type": "image",
            "ref_id": image.get("id", ""),
            "chunk_index": len(all_chunks)
        }
        all_chunks.append(image_chunk)
    
    return all_chunks


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
    
    print(f"\n[OK] Created {len(chunks)} semantic chunks")
    for i, chunk in enumerate(chunks):
        tokens = count_tokens(chunk)
        print(f"\n--- Chunk {i+1} ({tokens} tokens, {len(chunk)} chars) ---")
        print(chunk[:200] + ("..." if len(chunk) > 200 else ""))
    
    print("\n[OK] Semantic chunking complete!")
    print(f"[OK] Token tracking: min 50, max 200 tokens per chunk")
