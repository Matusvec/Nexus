# T-Retriever Architecture

> System overview for the Nexus hierarchical RAG system.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEXUS PLATFORM                           │
│                                                                 │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Frontend  │  │ Agentic AI   │  │ AR / VR  │  │  External  │ │
│  │ (Next.js) │  │ Personas     │  │ Clients  │  │  Agents    │ │
│  └─────┬─────┘  └──────┬───────┘  └─────┬────┘  └─────┬──────┘ │
│        │               │                │              │        │
│        ▼               ▼                ▼              ▼        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 RETRIEVAL API LAYER                      │    │
│  │            backend/retrieval_api.py                      │    │
│  │                                                         │    │
│  │  add_documents()  remove_documents()  query()           │    │
│  │  explain_retrieval()  get_tree_info()                   │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                       │
│           ┌─────────────┼──────────────┐                        │
│           ▼             ▼              ▼                        │
│  ┌──────────────┐ ┌──────────┐ ┌────────────────┐              │
│  │ Document     │ │ T-Query  │ │  T-Retriever   │              │
│  │ Manager      │ │ (hybrid  │ │  (tree build,  │              │
│  │ (lifecycle)  │ │ retrieval│ │   clustering)  │              │
│  └──────┬───────┘ └────┬─────┘ └───────┬────────┘              │
│         │              │               │                        │
│         ▼              ▼               ▼                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    STORAGE LAYER                         │    │
│  │              ChromaDB (vector + metadata)                │    │
│  │                                                         │    │
│  │  Layer 0: raw chunks    Layer 1+: summaries             │    │
│  │  Layer -1: graph metadata                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   SUPPORT MODULES                        │    │
│  │  gemini_client.py  embeddings.py  chunking.py           │    │
│  │  document_parser.py  config.py  utils.py                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Dependency Graph

```
retrieval_api.py
  ├── document_manager.py
  │     ├── document_parser.py
  │     ├── chunking.py
  │     │     └── embeddings.py → gemini_client.py
  │     ├── storage.py → chromadb
  │     └── t_retriever.py
  │           ├── gemini_client.py
  │           ├── embeddings.py
  │           ├── storage.py
  │           └── (umap, igraph, leidenalg, hdbscan – optional)
  └── t_query.py
        ├── t_retriever.py (EntityGraph, constants)
        ├── storage.py
        └── gemini_client.py
```

---

## 3. Tree Structure (per document)

```
            ┌──────────────────┐
            │  Layer 3 (root)  │  1 – 2 summary nodes
            │  Broad overview  │
            └────────┬─────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
   ┌──────┴──────┐     ┌───────┴──────┐
   │  Layer 2    │     │  Layer 2     │  3 – 8 summaries
   │  Themes     │     │  Themes      │
   └──────┬──────┘     └───────┬──────┘
          │                    │
    ┌─────┴─────┐        ┌────┴─────┐
    │           │        │          │
┌───┴───┐ ┌────┴──┐ ┌───┴───┐ ┌───┴───┐
│Layer 1│ │Layer 1│ │Layer 1│ │Layer 1│  10 – 30 summaries
│Topics │ │Topics │ │Topics │ │Topics │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │         │         │         │
  ┌─┴─┐    ┌─┴─┐    ┌──┴─┐    ┌─┴──┐
  │   │    │   │    │    │    │    │
 L0  L0   L0  L0   L0  L0   L0  L0    Raw chunks (Layer 0)
```

Each Layer-0 chunk stores:
- Document text (original)
- Contextualised embedding (Anthropic method)
- Extracted entities (JSON in metadata)
- Content references (images, tables)

Each Layer-1+ summary stores:
- LLM-generated summary text
- Aggregated entities
- `child_ids` (comma-separated IDs of the chunks it summarises)

---

## 4. Entity Graph (per document)

```
EntityGraph
  ├── nodes: {chunk_id → {entities, embedding, layer, metadata}}
  ├── edges: {chunk_id → {neighbor_id: weight}}
  └── entity_index: {entity_name → {chunk_id, chunk_id, ...}}

Edge weight = cosine_similarity(embedding) + entity_overlap × ENTITY_EDGE_WEIGHT
```

The graph is built once during tree construction and serialised into ChromaDB as a special chunk (`layer = -1`).  At query time it is loaded and used for graph-expansion retrieval.

---

## 5. Retrieval Flow

```
                          ┌─────────┐
                          │  query  │
                          └────┬────┘
                               │
                     classify_query()
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
         SIMPLE           COMPLEX          EXPLORATORY
              │                │                 │
              ▼                ▼                 ▼
     collapsed_tree    hybrid_retrieval    collapsed_tree
     (layers 0-1)       ┌────┼────┐       (layers 1-3)
              │          │    │    │              │
              │    tree  graph entity             │
              │          │    │    │              │
              │          └────┼────┘              │
              │           fusion                  │
              └────────────────┼──────────────────┘
                               ▼
                     deduplicate_results()
                               │
                               ▼
                  build_context_from_results()
                               │
                               ▼
                     generate_content()  → answer
```

---

## 6. Document Lifecycle

| Operation | API | What happens |
|---|---|---|
| **Add** | `add_documents(paths)` | parse → chunk → contextualise → store L0 → build tree → save graph |
| **Remove** | `remove_documents(ids)` | delete all layers + graph metadata + clear cache |
| **Update** | `update_documents(ids, paths)` | remove old → add new (atomic) |
| **Rebuild tree** | `rebuild_document_tree(id)` | delete L1+ → rebuild from L0 |
| **Incremental add** | `add_documents(paths, build_tree=False)` then `rebuild_document_tree(id)` | Add chunks first, rebuild later |

---

## 7. Explainability

The `explain_retrieval()` API returns:

```json
{
  "query_classification": {
    "type": "complex",
    "confidence": 0.8,
    "scores": {"simple": 0, "complex": 3, "exploratory": 0, "multihop": 2},
    "strategy": {"retrieval": "hybrid", "top_k": 15, "use_graph": true}
  },
  "retrieval_path": [
    {"step": "tree_retrieval", "results": 10, "layers_searched": [0, 1, 2]},
    {"step": "graph_expansion", "seeds": 5, "expanded": 8, "hops": 2},
    {"step": "entity_retrieval", "entities": ["UMAP", "clustering"], "matches": 4},
    {"step": "fusion", "total_candidates": 22, "final": 15}
  ],
  "results": [
    {
      "id": "doc_L0_chunk3",
      "layer": 0,
      "source": "tree+graph",
      "fusion_score": 0.87,
      "entity_matches": ["UMAP"],
      "text_preview": "..."
    }
  ]
}
```

---

## 8. Known Limitations

1. **Entity graph is per-document**: No cross-document entity linking yet.
2. **No streaming generation**: Answers are generated in one shot.
3. **Graph edge construction is O(n²)**: Acceptable up to ~10 k nodes per document; larger documents should be chunked more aggressively.
4. **No quantitative evaluation**: Retrieval quality is not automatically benchmarked.
5. **Cluster merge is local**: When a cluster shrinks below threshold, it merges with the nearest sibling only; a full re-clustering of the layer might produce better partitions (handled by background compaction).

---

## 9. Incremental Updates + Background Compaction

### Problem

The original implementation rebuilt every summary layer (L1+) from scratch whenever
a document was modified.  This is O(n) in the total number of chunks — unacceptable
for an online system serving concurrent queries.

### Solution: Localized Repair via Membership Index

```
TreeIndex  (backend/tree_index.py)
  ├── membership:   chunk_id → { cluster_id, parent_id, layer, children }
  ├── dirty_nodes:  set of IDs modified since last compaction
  ├── _tombstones:  soft-deleted IDs (purged on compaction)
  └── tree_version: monotonically increasing integer
```

**On chunk deletion** (`incremental_remove_chunks`):

```
  delete chunk from ChromaDB
       │
  mark tombstone in TreeIndex
       │
  remove chunk from parent.children
       │
  if parent.children.size < MIN_CLUSTER_SIZE_AFTER_DELETE:
  │   find nearest sibling cluster (cosine similarity)
  │   move remaining children → sibling
  │   tombstone old parent summary
  │   re-summarize sibling  ←  ONLY this node
  else:
  │   re-summarize parent  ←  ONLY this node
       │
  propagate upward along ancestor_chain()
       │  (re-summarize each ancestor — O(tree_depth) nodes)
       │
  if dirty_nodes ≥ COMPACTION_DIRTY_THRESHOLD:
       │   compact()  →  purge tombstones, bump tree_version
       ▼
  save_document_index()
```

This repairs **only the affected subtree** — unrelated clusters and their
summaries are never touched.

### Background Compaction

| Parameter | Default | Description |
|---|---|---|
| `COMPACTION_DIRTY_THRESHOLD` | 10 | Number of dirty nodes before auto-compaction |
| `COMPACTION_ENABLED` | True | Toggle compaction on/off |
| `MIN_CLUSTER_SIZE_AFTER_DELETE` | 2 | Merge when cluster drops below this |

Compaction:
1. Purges all tombstoned entries from the membership dict.
2. Clears the dirty-node set.
3. Bumps `tree_version`.

Because compaction only mutates the in-memory index and the ChromaDB metadata
chunk (layer = -2), it is **lock-free** with respect to concurrent queries:
queries read the vector index directly and are not affected by index metadata
changes.

### Atomic Versioning

`tree_version` is stored in the tree index metadata (layer = -2) and
incremented on every compaction.  Downstream consumers can poll `get_tree_info()`
to detect when a new version is available.

### Retrieval API

| Endpoint | Behaviour |
|---|---|
| `remove_documents(ids)` | Full removal — deletes all layers, graph, and index |
| `remove_chunks(doc_id, chunk_ids)` | **Incremental** — localized repair, no rebuild |
| `query()` / `explain_retrieval()` | Unchanged — reads current state of ChromaDB |

---

## 10. Future Extensions

- **Cross-document entity graph**: Link entities across documents for corpus-wide multi-hop reasoning.
- **Online incremental clustering**: Add new chunks to existing clusters without rebuilding the layer.
- **FastAPI server**: Expose retrieval API over HTTP (spec in `frontend/API_SPECIFICATION.md`).
- **AR/VR integration**: Spatial knowledge graph navigation.
- **Feedback loop**: Re-rank results based on user interaction signals.
- **Scheduled compaction job**: Run compaction on a timer or as a background thread.
