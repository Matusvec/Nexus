# T-Retriever Implementation Notes

## Overview

This document describes the implementation of **T-Retriever** (Tree-based Hierarchical Retrieval for Textual Graphs) which replaces the original RAPTOR implementation.

Based on: "T-Retriever: Tree-based Hierarchical Retrieval Augmented Generation for Textual Graphs" (Wei et al., 2026)

## Key Improvements Over RAPTOR

| Feature | RAPTOR | T-Retriever |
|---------|--------|-------------|
| Structure | Pure tree hierarchy | Tree + Entity Graph |
| Clustering | GMM/Leiden on embeddings | Entity-aware clustering |
| Context | Embedding similarity only | Embeddings + entity relationships |
| Retrieval | Collapsed tree search | Hybrid (tree + graph expansion) |
| Multi-hop | Limited | Graph traversal enables multi-hop |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     T-RETRIEVER ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   Layer 2   │    │              │    │   Entity      │  │
│  │  Summaries  │◄───│   Hybrid     │◄───│   Graph       │  │
│  └──────┬──────┘    │  Retrieval   │    │   (edges by   │  │
│         │           │              │    │   shared      │  │
│  ┌──────▼──────┐    │  • Tree      │    │   entities)   │  │
│  │   Layer 1   │◄───│    Search    │    │               │  │
│  │  Summaries  │    │  • Graph     │    │  chunk1 ──────│  │
│  └──────┬──────┘    │    Expansion │    │     │         │  │
│         │           │  • Entity    │    │  chunk2 ──────│  │
│  ┌──────▼──────┐    │    Matching  │    │     │         │  │
│  │   Layer 0   │◄───│              │    │  chunk3       │  │
│  │ Base Chunks │    └──────────────┘    └───────────────┘  │
│  │ + Entities  │                                            │
│  └─────────────┘                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Entity Extraction (`t_retriever.py`)

Extracts entities from each chunk using:
- **LLM extraction**: Uses Gemini to identify key entities (people, concepts, methods, etc.)
- **Simple fallback**: Pattern-based extraction for capitalized phrases and technical terms

Each entity has:
- `name`: The entity text
- `type`: person, organization, concept, method, technology, location, other
- `importance`: 1-10 score

### 2. Entity Graph

Nodes are chunks, edges connect chunks that:
- Share entities (weighted by entity importance)
- Have high embedding similarity

The graph enables:
- **Graph expansion**: Find related chunks not found by embedding similarity
- **Entity-based retrieval**: Direct lookup by entity name
- **Multi-hop reasoning**: Traverse relationships for complex queries

### 3. Entity-Aware Clustering

Unlike RAPTOR's pure embedding clustering, T-Retriever uses:
```
combined_similarity = 0.7 * embedding_similarity + 0.3 * entity_overlap
```

This keeps semantically related content together even if embeddings differ slightly.

### 4. Hybrid Retrieval

Query processing combines three methods:

1. **Tree Retrieval** (60% weight by default):
   - Collapsed tree search across all layers
   - Entity boost for chunks matching query terms

2. **Graph Expansion** (28% weight):
   - Starts from top tree results
   - Expands 2 hops through entity graph
   - Finds related content missed by embedding search

3. **Entity Matching** (12% weight):
   - Direct lookup of chunks containing query entities
   - Fast and precise for known-entity queries

Results are fused using reciprocal rank fusion.

## Configuration

Key parameters in `t_retriever.py`:

```python
# Entity extraction
MAX_ENTITIES_PER_CHUNK = 15
ENTITY_SIMILARITY_THRESHOLD = 0.85  # For deduplication

# Graph building
GRAPH_EDGE_SIMILARITY_THRESHOLD = 0.3
GRAPH_KNN_NEIGHBORS = 8
ENTITY_EDGE_WEIGHT = 1.5  # Boost for entity connections

# Retrieval
GRAPH_EXPANSION_HOPS = 2
HYBRID_ALPHA = 0.6  # Tree vs graph weight
```

## Usage

### Building a T-Retriever Tree

```bash
# Upload document first
python main.py upload my_document.pdf

# Build tree with entity extraction
python main.py build-tree -d my_document

# Rebuild if needed
python main.py build-tree -d my_document --rebuild
```

### Querying

```bash
# Standard query (auto-selects strategy)
python main.py query "What is the relationship between X and Y?"

# Query specific document
python main.py query "Explain the methodology" -d my_document
```

### Interactive Mode

```bash
python t_query.py interactive

# Commands:
# doc:<id>    - Filter to document
# graph:on    - Enable graph expansion
# graph:off   - Disable graph expansion
```

## Query Classification

Queries are auto-classified:

| Type | Indicators | Strategy |
|------|------------|----------|
| Simple | "what is", "define", "who" | Tree only, lower layers |
| Complex | "compare", "relationship", "how does X affect Y" | Hybrid (tree + graph) |
| Exploratory | "tell me about", "summarize" | Tree only, higher layers |

Multi-hop indicators ("relationship between", "leads to", "depends on") trigger graph expansion.

## File Structure

```
backend/
├── t_retriever.py      # Core T-Retriever implementation
│   ├── Entity extraction (LLM + simple)
│   ├── EntityGraph class
│   ├── Entity-aware clustering
│   ├── Tree building
│   └── Graph save/load
│
├── t_query.py          # Query module for T-Retriever
│   ├── Query classification
│   ├── Tree retrieval
│   ├── Graph expansion retrieval
│   ├── Entity-based retrieval
│   ├── Hybrid fusion
│   └── Answer generation
│
├── main.py             # CLI (updated to use T-Retriever)
├── raptor.py           # Original RAPTOR (kept for reference)
└── query.py            # Original query module (kept for reference)
```

## Performance Considerations

### Entity Extraction
- LLM extraction: ~1-2 seconds per chunk
- Use `extract_entities_simple()` for faster processing

### Graph Building
- O(n²) for edge computation
- Consider batching for large documents (1000+ chunks)

### Storage
- Graph stored as JSON in ChromaDB metadata
- Loaded on-demand per document

## Comparison with Research Paper

| Paper Feature | Implementation Status |
|--------------|----------------------|
| Tree hierarchy | ✅ Implemented |
| Entity extraction | ✅ LLM + fallback |
| Entity graph | ✅ k-NN + entity edges |
| Graph expansion | ✅ Multi-hop traversal |
| Hybrid retrieval | ✅ Tree + Graph + Entity |
| Textual graph input | 🔄 Adapted for documents |

The paper targets knowledge graphs with explicit relations. Our implementation adapts this for document retrieval by:
1. Extracting entities from text chunks
2. Inferring relations from co-occurrence
3. Using embedding similarity as relation strength

## Future Improvements

1. **Relation extraction**: Extract explicit relations ("X causes Y") not just entities
2. **Incremental updates**: Add/remove documents without full rebuild
3. **Cross-document graph**: Link entities across multiple documents
4. **Custom entity types**: Domain-specific entity schemas
5. **Graph visualization**: Visual exploration of entity connections

## References

- T-Retriever paper: arXiv:2601.04945v1 (2026)
- RAPTOR paper: arXiv:2401.18059 (2024)
- LATTICE paper: arXiv:2510.13217 (2025)
- BookRAG paper: arXiv:2512.03413 (2025)
