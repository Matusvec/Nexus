# T-Retrieval: Technical Summary

> Paper-to-code mapping for the Nexus T-Retriever implementation.
> Based on "T-Retriever: Tree-based Hierarchical Retrieval Augmented Generation for Textual Graphs" (Wei et al., 2026).

---

## 1. Paper Overview

T-Retriever introduces a **tree-based hierarchical retrieval** approach that combines:

1. **Hierarchical tree structure**: Documents are chunked, clustered, and summarised recursively to produce a multi-layer tree. Leaf nodes are raw chunks; inner nodes are cluster summaries.
2. **Entity-aware clustering**: Bottom-layer clusters are formed using both embedding similarity *and* shared-entity overlap, so thematically *and* factually related chunks stay together.
3. **Hybrid retrieval**: At query time, a collapsed-tree search (across all layers simultaneously) is fused with graph-expansion retrieval that walks entity relationships, improving multi-hop reasoning.

### Key concepts

| Paper concept | Code location | Description |
|---|---|---|
| Textual graph | `t_retriever.EntityGraph` | Chunks become nodes; edges are weighted by embedding sim + entity overlap |
| Hierarchical tree | `build_tretriever_tree()` | Recursive cluster → summarise loop building layers 0 → N |
| Bottom-layer clustering | `cluster_with_entities()` | UMAP reduction → Leiden / GMM / HDBSCAN, combined similarity 70 % embed + 30 % entity Jaccard |
| Entity extraction | `extract_entities()` / `extract_entities_simple()` | Fast rule-based (default) or LLM-based entity extraction per chunk |
| Collapsed tree retrieval | `t_query.collapsed_tree_retrieval()` | Query embedding searched against **all** layers simultaneously |
| Graph expansion | `t_query.graph_expansion_retrieval()` | BFS from seed chunks along entity graph edges |
| Hybrid fusion | `t_query.hybrid_retrieval()` | α-weighted merge of tree + graph + entity match scores |
| Adaptive strategy | `t_query.adaptive_retrieval()` | Query classification → strategy selection (simple / complex / exploratory) |

---

## 2. Tree Construction Pipeline

```
Document
  │
  ▼
parse_document()          → raw text + images + tables
  │
  ▼
chunk_text()              → semantic chunks (Layer 0 leaf nodes)
  │
  ▼
contextualize_chunks()    → prepend document context (Anthropic method)
  │
  ▼
store_contextualized_chunks()   → ChromaDB (embeddings from contextualised text)
  │
  ▼
build_tretriever_tree()   → for each layer:
  │                            1. extract_entities() on every node
  │                            2. build EntityGraph (nodes + edges)
  │                            3. cluster_with_entities()
  │                            4. summarize_cluster_with_entities()
  │                            5. store next layer
  ▼
save_document_graph()     → persist graph structure in ChromaDB
```

### Layer-adaptive resolution

At higher layers the Leiden clustering resolution decreases (`resolution = 1.0 × 0.7^layer`), producing broader clusters.  This mirrors the paper's recommendation that upper levels capture broader thematic regions.

---

## 3. Query-Time Retrieval

```
Query
  │
  ▼
classify_query()          → simple | complex | exploratory
  │
  ├─ simple ──────► collapsed_tree_retrieval()  (layers 0–1)
  │
  ├─ exploratory ─► collapsed_tree_retrieval()  (layers 1–3, summaries)
  │
  └─ complex ─────► hybrid_retrieval()
                       ├─ collapsed_tree_retrieval()   (weight α)
                       ├─ graph_expansion_retrieval()  (weight 1-α × 0.7)
                       └─ entity_based_retrieval()     (weight 1-α × 0.3)
                       → fusion_score ranking
  │
  ▼
deduplicate_results()  → remove overlapping chunks covered by parent summaries
  │
  ▼
build_context_from_results()  → context window for LLM generation
  │
  ▼
generate_content()  → final answer
```

### Entity-query boost

During collapsed-tree retrieval, chunks whose `entity_names` metadata matches query entities receive a distance discount (`ENTITY_QUERY_BOOST = 0.2`).  This is the paper's "entity-aware scoring" translated into cosine-distance space.

---

## 4. Design Decisions & Deviations from the Paper

| Decision | Rationale |
|---|---|
| **Fast entity extraction by default** | The paper assumes an NER pipeline; we use rule-based extraction for speed, with LLM extraction as opt-in (`ENTITY_EXTRACTION_MODE=llm`). |
| **UMAP before clustering** | The RAPTOR paper (cited by T-Retriever) recommends dimensionality reduction for tighter clusters; falls back gracefully when `umap-learn` is absent. |
| **Three clustering strategies** | Leiden (community detection, default), GMM (RAPTOR soft clustering), HDBSCAN (density). Configurable via `CLUSTERING_STRATEGY`. |
| **ChromaDB for all layers** | The paper is storage-agnostic; ChromaDB provides persistent vector storage with metadata filtering, allowing `layer >= 0` queries. |
| **Graph serialised in ChromaDB** | Graph structure is JSON-serialised into a special chunk (`layer = -1`) for persistence without an external graph DB. |
| **Document-scoped graphs** | Each document has its own `EntityGraph`; cross-document entity linking is a planned future extension. |

---

## 5. Scalability Notes

| Operation | Complexity | Notes |
|---|---|---|
| Entity extraction (fast) | O(n) per chunk | Regex-based, sub-millisecond per chunk |
| Entity extraction (LLM) | O(n) × LLM latency | ~1 s per chunk; parallelisable |
| Embedding generation | O(n) × API latency | Batched via Gemini API |
| Graph edge construction | O(n²) | Quadratic in nodes per layer; mitigated by per-document scope |
| Leiden clustering | O(E + n log n) | Linear in edges; fast for < 10 k nodes |
| UMAP reduction | O(n^1.14) approx | Sub-quadratic; fast for < 50 k |
| Collapsed tree retrieval | O(log n) via vector index | ChromaDB ANN lookup |
| Graph expansion | O(k^h) | k = top-K neighbors, h = hops (default 3² = 9 lookups max) |

For corpora up to ~100 k chunks, the current architecture runs comfortably.
Beyond that, consider sharding graphs per document group and ANN-based edge construction.

---

## 6. Incremental Updates (deviation from naive approach)

The T-Retriever paper assumes a static corpus: build the tree once, query many times.
Our implementation extends this for an **online system** where documents are added and
removed continuously.

### What the paper assumes

> Build the full tree once.  Rebuild on any change.

### What we do instead

1. **Membership index** (`TreeIndex`): Every chunk knows its cluster, parent summary,
   and ancestor chain.  This is built during `build_tretriever_tree()` and persisted
   alongside the tree.

2. **Localized repair on deletion**: When a chunk is removed, only its parent cluster
   and ancestor summaries are re-computed.  The cost is O(tree_depth × LLM_latency)
   instead of O(n² + n × LLM_latency) for a full rebuild.

3. **Cluster merging**: If a cluster drops below `MIN_CLUSTER_SIZE_AFTER_DELETE`,
   its remaining children are moved to the nearest sibling cluster (by centroid
   cosine similarity).  This preserves the hierarchical structure without requiring
   a full re-clustering.

4. **Background compaction**: Dirty nodes accumulate; when the threshold is reached,
   tombstoned entries are purged and `tree_version` is bumped.  Queries are not
   blocked during compaction.

### Why this is faithful to the paper

The paper's key contribution is the **entity-aware hierarchical structure** and
**hybrid retrieval**.  Our incremental updates preserve both:

- **Entity graph** is unchanged by localized repair (graph edges are between chunks,
  not between summaries).
- **Hierarchical summaries** are regenerated from the same child nodes — the only
  difference is that one child is missing.
- **Collapsed-tree retrieval** searches all layers simultaneously in ChromaDB;
  it is unaffected by index metadata changes.
- **Graph expansion** walks entity edges that are unrelated to cluster membership.

The hierarchical retrieval guarantees (multi-resolution search, entity-aware scoring,
hybrid fusion) are fully maintained.

---

## 7. Future Work

- Cross-document entity graphs (merge entity indices across docs)
- Online incremental clustering (add new chunks to existing clusters)
- Retrieval feedback loop (re-rank based on user interaction)
- Streaming generation with source highlighting
- Scheduled compaction as a background thread or cron job
