"""
Text chunking utilities for RAPTOR
"""
import re
from typing import List
import config


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks by sentences
    
    Args:
        text: Input text to chunk
        chunk_size: Target characters per chunk (default from config)
        overlap: Overlap between chunks (default from config)
        
    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or config.RAPTOR_CHUNK_SIZE
    overlap = overlap or config.RAPTOR_CHUNK_OVERLAP
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # If single sentence exceeds chunk_size, add it alone
        if sentence_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            chunks.append(sentence)
            continue
        
        # If adding sentence exceeds chunk_size, start new chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


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
    print("Testing text chunking...")
    
    test_text = """
    Artificial intelligence is transforming technology. Machine learning enables 
    computers to learn from data. Deep learning uses neural networks with multiple layers.
    Natural language processing helps computers understand human language. Computer vision 
    allows machines to interpret images. These technologies are revolutionizing industries.
    AI applications include healthcare, finance, and transportation. Research continues 
    to advance the field rapidly.
    """
    
    chunks = chunk_text(test_text, chunk_size=150, overlap=30)
    
    print(f"✓ Split text into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f"  {chunk[:80]}...")
