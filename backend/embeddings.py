"""
Embedding service using Google Gemini
"""
from google import genai
from google.genai import types
from typing import List
import config

# Configure Gemini client
client = genai.Client(api_key=config.GEMINI_API_KEY)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using Gemini
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = []
    
    for text in texts:
        response = client.models.embed_content(
            model=config.GEMINI_EMBEDDING_MODEL,
            contents=text
        )
        embeddings.append(response.embeddings[0].values)
    
    return embeddings


def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a single text
    
    Args:
        text: Text string to embed
        
    Returns:
        Embedding vector
    """
    return get_embeddings([text])[0]


# Test
if __name__ == "__main__":
    print("Testing Gemini embeddings...")
    test_text = "This is a test document about artificial intelligence."
    embedding = get_embedding(test_text)
    print(f"✓ Generated embedding with dimension: {len(embedding)}")
    print(f"✓ First 5 values: {embedding[:5]}")
