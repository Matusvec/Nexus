"""
Centralized Gemini API client for Nexus

Single source of truth for all Gemini API interactions.
Import this module instead of initializing clients in each file.
"""
from google import genai
from google.genai import types
import config

# Single Gemini client instance - used by all modules
client = genai.Client(api_key=config.GEMINI_API_KEY)

# Re-export types for convenience
ContentType = types.Content
PartType = types.Part


def generate_content(prompt: str, model: str = None) -> str:
    """
    Generate text content using Gemini
    
    Args:
        prompt: Text prompt
        model: Model to use (defaults to config.GEMINI_GENERATION_MODEL)
        
    Returns:
        Generated text response
    """
    model = model or config.GEMINI_GENERATION_MODEL
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text.strip()


def generate_with_image(image_data: bytes, prompt: str, mime_type: str = "image/png") -> str:
    """
    Generate content from an image using Gemini Vision
    
    Args:
        image_data: Raw image bytes
        prompt: Text prompt describing what to do with the image
        mime_type: Image MIME type (default: image/png)
        
    Returns:
        Generated text response
    """
    response = client.models.generate_content(
        model=config.GEMINI_GENERATION_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    types.Part(text=prompt)
                ]
            )
        ]
    )
    return response.text.strip()


def get_embedding(text: str) -> list:
    """
    Get embedding for a single text
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector as list of floats
    """
    response = client.models.embed_content(
        model=config.GEMINI_EMBEDDING_MODEL,
        contents=text
    )
    return response.embeddings[0].values


def get_embeddings(texts: list) -> list:
    """
    Get embeddings for multiple texts
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings = []
    for text in texts:
        embeddings.append(get_embedding(text))
    return embeddings


def generate_document_summary(text: str, filename: str = "") -> str:
    """
    Generate a concise summary of a document for contextual retrieval
    
    Anthropic's Contextual Retrieval: Each chunk needs document-level context
    to improve retrieval accuracy by ~49%.
    
    Args:
        text: Full document text
        filename: Document filename for context
        
    Returns:
        2-3 sentence summary of the document
    """
    # Take first and last portions of document for summary
    text_sample = text[:3000] + "\n...\n" + text[-1000:] if len(text) > 4000 else text
    
    prompt = f"""Summarize this document in 2-3 sentences. Focus on:
- What type of document is this?
- What are the main topics/subjects covered?
- What is the purpose of this document?

Document name: {filename}

Document content:
{text_sample}

Provide ONLY the summary, no other text."""

    return generate_content(prompt)


def generate_chunk_context(chunk: str, doc_summary: str, doc_name: str) -> str:
    """
    Generate situational context for a specific chunk (Anthropic's method)
    
    Args:
        chunk: The chunk text
        doc_summary: Summary of the full document
        doc_name: Document filename
        
    Returns:
        Contextual prefix for the chunk
    """
    prompt = f"""Given this document summary and a chunk from it, write a SHORT (1-2 sentence) context 
that situates this chunk within the document. This context will be prepended to the chunk 
for better search retrieval.

Document: {doc_name}
Document Summary: {doc_summary}

Chunk:
{chunk[:500]}

Provide ONLY the contextual prefix, nothing else. Example format:
"This chunk from [doc name] discusses [specific topic], following the section on [previous topic]."
"""
    
    return generate_content(prompt)
