"""
Embedding service for Nexus

Uses the centralized Gemini client for generating text embeddings.
"""
from typing import List
from gemini_client import get_embedding as _get_embedding, get_embeddings as _get_embeddings


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using Gemini
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    return _get_embeddings(texts)


def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text
    
    Args:
        text: Text string to embed
        
    Returns:
        Embedding vector
    """
    return _get_embedding(text)


# Test
if __name__ == "__main__":
    print("Testing Gemini embeddings...")
    test_text = "This is a test document about artificial intelligence."
    embedding = get_embedding(test_text)
    print(f"[OK] Generated embedding with dimension: {len(embedding)}")
    print(f"[OK] First 5 values: {embedding[:5]}")
