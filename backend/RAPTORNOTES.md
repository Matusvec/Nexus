# RAPTOR Implementation Notes

## Original User Notes
> Next we have to start implementing the RAPTOR hierarchy and then querying from there. Search the web for advice on RAPTOR hierarchy. I know it's cutting edge and there are problems with adding and removing documents. However, we will solve that or at least mitigate it by:
> - For removals: go up the tree and re-summarize only the affected nodes, always checking if we aren't significantly rewriting the tree structure (in cases where there aren't many nodes at all)
> - We will re-make the entire tree periodically at night or something in the background

---

## RAPTOR Paper Summary (ICLR 2024)
**Title:** RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval  
**Authors:** Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning (Stanford)  
**Paper:** arXiv:2401.18059v1

### Core Problem
Traditional RAG retrieves only short contiguous chunks, limiting holistic understanding of document context. This fails for:
- Thematic questions requiring integration across multiple parts
- Multi-hop reasoning questions
- Questions about entire narratives (e.g., "How did Cinderella reach her happy ending?")

### RAPTOR Solution
Build a **hierarchical tree structure** that captures both high-level themes AND low-level details:

```
                    [Root: Full Doc Summary]
                           /          \
              [Summary L2]              [Summary L2]
              /    |    \                /    |    \
        [Sum L1] [Sum L1] [Sum L1]  [Sum L1] [Sum L1]
          /|\       |       /|\        |       /|\
       [chunks] [chunks] [chunks]  [chunks] [chunks]  <- Layer 0 (Leaf nodes)
```

---

## RAPTOR Algorithm (Step-by-Step)

### Phase 1: Tree Construction (Bottom-Up)

#### Step 1: Initial Chunking
- Split corpus into **100-token chunks**
- If sentence exceeds 100 tokens, move entire sentence to next chunk
- Preserves semantic coherence within chunks

#### Step 2: Embed Chunks
- Use SBERT (`multi-qa-mpnet-base-cos-v1`) for embeddings
- These become **leaf nodes** (Layer 0)

#### Step 3: Clustering (GMM + UMAP)
Key innovation: **Soft clustering** where nodes can belong to MULTIPLE clusters

1. **Dimensionality Reduction with UMAP**
   - High-dimensional embeddings → lower dimensions
   - `n_neighbors` parameter controls local vs global structure
   - Use DIFFERENT `n_neighbors` values:
     - Large value first → find global clusters
     - Small value second → find local clusters within global

2. **Gaussian Mixture Models (GMM)**
   - Soft clustering (probabilistic membership)
   - Use **Bayesian Information Criterion (BIC)** to determine optimal cluster count
   - BIC = ln(N)k − 2ln(L̂) where N=data points, k=parameters, L̂=likelihood

3. **Recursive Clustering**
   - If cluster exceeds token threshold → recursively cluster again
   - Ensures summaries stay within LLM context limits

#### Step 4: Summarization
- Use LLM (GPT-3.5-turbo in paper) to summarize each cluster
- ~4% of summaries had minor hallucinations (didn't propagate to parents)

#### Step 5: Re-embed and Repeat
- Embed the summaries
- These become the next layer's nodes
- Repeat clustering + summarization until no more clustering possible

### Phase 2: Querying (Two Methods)

#### Method A: Tree Traversal
```
1. Start at root layer
2. Compute cosine similarity between query and all nodes at current layer
3. Select top-k nodes → set S₁
4. Move to children of S₁
5. Select top-k from children → set S₂
6. Repeat until leaf nodes
7. Concatenate all selected sets: S₁ ∪ S₂ ∪ ... ∪ Sₐ
```
**Limitation:** Fixed ratio of high-level to granular information

#### Method B: Collapsed Tree (RECOMMENDED)
```
1. Flatten entire tree into single layer (set C)
2. Compute cosine similarity between query and ALL nodes in C
3. Select nodes until reaching token limit (e.g., 2000 tokens ≈ top-20 nodes)
```
**Advantage:** More flexible - retrieves at the right granularity for each question

**Paper's finding:** Collapsed tree with 2000 tokens consistently outperformed tree traversal

---

## Key Technical Details

### Clustering Parameters (from paper)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Chunk size | 100 tokens | ~400 chars |
| Embedding model | SBERT multi-qa-mpnet-base-cos-v1 | For embeddings |
| Summarization model | GPT-3.5-turbo | For cluster summaries |
| Query context | 2000 tokens | ~top-20 nodes |
| Cluster selection | BIC (Bayesian Info Criterion) | Auto-determines k |

### UMAP Settings
- Vary `n_neighbors` for hierarchical clustering:
  - Higher values → global structure
  - Lower values → local structure within global clusters

### Soft Clustering Benefits
- Text segments often relate to multiple topics
- Single chunk can appear in multiple cluster summaries
- Better captures interconnected themes

---

## Performance Results (from paper)

| Dataset | RAPTOR + GPT-4 | Best Baseline | Improvement |
|---------|----------------|---------------|-------------|
| QuALITY | **82.6%** | 62.3% (CoLISA) | +20.3% |
| QuALITY-HARD | **76.2%** | 54.7% | +21.5% |
| QASPER | **55.7%** F1 | 53.9% (CoLT5 XL) | +1.8% |
| NarrativeQA | **19.1** METEOR | 11.1 (Retriever+Reader) | +8.0 |

---

## Implementation Plan for Nexus

### Current Stack
- **Embeddings:** Gemini `gemini-embedding-001`
- **Generation:** Gemini `gemini-2.0-flash-exp`
- **Storage:** ChromaDB (persistent)
- **Chunking:** Semantic chunking with overlap + contextual retrieval

### Phase 1: Build Tree Structure

```python
# raptor.py structure
def build_raptor_tree(document_id: str) -> Dict:
    """
    Build RAPTOR tree from existing Layer 0 chunks
    
    Returns:
        Dict with tree metadata (layer count, node counts, etc.)
    """
    # 1. Get all Layer 0 chunks for this document
    # 2. Cluster using GMM + UMAP
    # 3. Summarize each cluster
    # 4. Store summaries as Layer 1 chunks
    # 5. Repeat until single root or no more clustering
```

### Phase 2: Storage Schema Updates

Current chunk metadata:
```python
{
    "document_id": str,
    "layer": int,  # Already have this!
    "chunk_index": int,
    "content_type": str,
    # ... existing fields
}
```

Need to add:
```python
{
    "parent_ids": List[str],    # For tree traversal
    "child_ids": List[str],     # For tree traversal
    "cluster_id": str,          # Which cluster this belongs to
    "is_summary": bool,         # True for Layer 1+
}
```

### Phase 3: Query Updates

```python
# query.py - collapsed tree method
def query_raptor(question: str, max_tokens: int = 2000) -> List[Dict]:
    """
    Query using collapsed tree method
    
    1. Get ALL nodes (all layers) for relevant documents
    2. Compute similarity to question
    3. Return top nodes until max_tokens reached
    """
```

### Phase 4: Incremental Updates (Your Mitigation Strategy)

```python
def add_document_to_tree(document_id: str):
    """Add new document chunks and update affected summaries"""
    # 1. Store new chunks at Layer 0
    # 2. Find which existing clusters they might belong to
    # 3. Re-cluster affected regions only
    # 4. Re-summarize only changed clusters
    # 5. Propagate changes up tree (mark for re-summary)

def remove_document_from_tree(document_id: str):
    """Remove document and update tree"""
    # 1. Delete all chunks for this document
    # 2. Find parent summaries that referenced these chunks
    # 3. Re-cluster the remaining siblings
    # 4. Re-summarize affected parents
    # 5. If tree structure significantly changed → flag for full rebuild

def schedule_full_rebuild():
    """Periodic full tree rebuild (nightly)"""
    # 1. Get all Layer 0 chunks
    # 2. Rebuild entire tree from scratch
    # 3. Swap old tree for new tree atomically
```

---

## Dependencies to Add

```txt
# requirements.txt additions
umap-learn>=0.5.0      # For dimensionality reduction
scikit-learn>=1.0.0    # For GMM clustering (already have for cosine_similarity)
```

---

## Implementation Priority

1. **[HIGH]** Implement `raptor.py` with `build_raptor_tree()`
2. **[HIGH]** Update `storage.py` with tree metadata fields
3. **[HIGH]** Implement collapsed tree query in `query.py`
4. **[MEDIUM]** Add `build-tree` command to `main.py`
5. **[MEDIUM]** Implement incremental update functions
6. **[LOW]** Add scheduled full rebuild task

---

## Open Questions / Decisions

1. **Embedding model for summaries:** Use same Gemini model or switch to SBERT?
   - Paper used SBERT for all nodes
   - Our contextual retrieval uses Gemini
   - **Decision:** Use Gemini for consistency with existing chunks

2. **Cluster count:** Use BIC auto-detection or fixed number?
   - Paper used BIC
   - **Decision:** Start with BIC, add manual override option

3. **Max tree depth:** How many layers before stopping?
   - Paper didn't specify hard limit
   - **Decision:** Stop when < 3 nodes remain OR max 5 layers

4. **Soft clustering threshold:** Probability cutoff for multiple cluster membership
   - Paper didn't specify
   - **Decision:** Include in cluster if probability > 0.1

