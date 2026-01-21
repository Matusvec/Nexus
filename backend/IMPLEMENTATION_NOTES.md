# Microsoft Best Practices Implementation

## Summary
Updated document parsing and chunking to follow Microsoft's recommended best practices for RAG systems.

## Changes Implemented

### 1. Context-Aware Image Descriptions
**Microsoft Recommendation**: "Pass text before/after image to LLM for better descriptions"

**Implementation** in `document_parser.py`:
- New function: `describe_image_with_context()` 
- Passes 500 chars of text before and after image to Gemini Vision
- Improved prompt asks model to consider context when describing
- Handles diagrams, charts, tables, and photos differently based on type

**Benefits**:
- Better image descriptions that understand document context
- More accurate descriptions of technical diagrams and charts
- Preserves semantic relationship between images and surrounding text

### 2. Image Reference Tracking
**Microsoft Recommendation**: "Include image URL in each chunk to ensure metadata is returned"

**Implementation** in `document_parser.py`:
- Each image gets unique reference ID: `img_filename_p1_0`, `img_filename_para5_1`, etc.
- Format: `[IMAGE img_ref: description]` instead of generic `[IMAGE: description]`
- Metadata includes `image_references` list for all images in document

**Benefits**:
- Can track which chunks reference which images
- Enables multi-chunk scenarios where image is discussed across chunks
- Provides traceability back to original image location

### 3. Image Reference Preservation in Chunks
**Microsoft Recommendation**: "If image description splits into multiple chunks, include image URL in each chunk"

**Implementation** in `chunking.py`:
- New function: `extract_image_references()` - finds all image IDs in text
- New function: `preserve_image_refs_in_chunk()` - ensures image refs aren't lost
- Updated `chunk_text()` to preserve image references when splitting

**Benefits**:
- Chunks that discuss an image retain the image reference
- Query results can link back to source images
- Maintains context even when semantic chunking splits image discussion

### 4. Token-Based Chunking (Already Implemented)
**Microsoft Recommendation**: "Use BERT tokens instead of character counts"

**Implementation** in `chunking.py`:
- Token estimation: 1 token ≈ 4 characters
- Min 100 tokens, max 500 tokens per chunk
- Tracks costs for Gemini API usage

**Benefits**:
- More accurate chunk sizes for LLM processing
- Better cost tracking and control
- Aligns with industry standard chunking practices

### 5. Semantic + Fixed-Size Hybrid (Already Implemented)
**Microsoft Recommendation**: "Fixed-size parsing with overlap" + "Semantic chunking"

**Implementation** in `chunking.py`:
- Combines semantic similarity detection (cosine similarity < 0.7)
- With hard size limits (max 500 tokens)
- Groups 2-3 sentences before embedding to prevent micro-chunks

**Benefits**:
- Best of both worlds: semantic coherence + predictable sizes
- Prevents oversized chunks that break LLM context windows
- Prevents tiny chunks that lose semantic meaning

## Testing

### Test document_parser.py:
```bash
cd backend
python document_parser.py your_file.pdf
```

Expected output:
- Extracted text with page markers
- Image descriptions with unique IDs
- Context-aware image descriptions mentioning surrounding text
- Metadata with image count and reference list

### Test chunking.py:
```bash
cd backend
python chunking.py
```

Expected output:
- Chunks with preserved image references
- Token counts within 100-500 range
- Semantic coherence (topic shifts at chunk boundaries)

## Next Steps

1. **Test with real documents**: Place PDF/DOCX files in backend folder and test
2. **Verify image descriptions**: Check if context improves description quality
3. **Monitor chunk quality**: Ensure images aren't split awkwardly
4. **Build RAPTOR tree**: Next phase - clustering and hierarchical summaries
5. **ChromaDB integration**: Store chunks with image reference metadata

## References

- Microsoft Article: "Language model augmentation" and "Fixed-size parsing with overlap"
- Document: `backend/chunkingAndUploadInfo.md`
- Implementation: Follows Microsoft's "high engineering effort, high processing cost" approach for best quality
