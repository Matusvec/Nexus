"""
Shared utility functions for Nexus backend

This module contains common utilities used across multiple modules
to avoid code duplication.
"""
import re
from typing import List, Dict


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
    return len(text) // 4


def extract_content_references(text: str) -> Dict[str, any]:
    """
    Extract image and table references from text with [IMAGE ...] and [TABLE ...] markers
    
    Args:
        text: Text chunk with markers
        
    Returns:
        Dict with content_types, image_refs, table_refs, has_images, has_tables
    """
    # Extract image references
    image_pattern = r'\[IMAGE\s+([^\s:]+):'
    image_refs = re.findall(image_pattern, text)
    
    # Extract table references
    table_pattern = r'\[TABLE\s+([^\s:]+):'
    table_refs = re.findall(table_pattern, text)
    
    # Determine content types
    content_types = ["text"]  # Always has text
    if image_refs:
        content_types.append("image")
    if table_refs:
        content_types.append("table")
    
    return {
        "content_types": content_types,
        "image_refs": image_refs,
        "table_refs": table_refs,
        "has_images": len(image_refs) > 0,
        "has_tables": len(table_refs) > 0
    }


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
