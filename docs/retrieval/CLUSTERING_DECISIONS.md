# Clustering Decisions

> Analysis of bottom-layer clustering strategies for the Nexus T-Retriever.

---

## 1. Why Clustering Matters

Bottom-layer clustering is the most critical step in hierarchical RAG: it decides **which chunks get grouped together** to form the next-level summary.  Poor clustering produces incoherent summaries that poison all higher layers.

The T-Retriever paper recommends clustering that respects both **semantic similarity** and **entity overlap**, so that chunks sharing key concepts always land in the same cluster.

---

## 2. Strategies Evaluated

### 2.1 Leiden Community Detection (default: `CLUSTERING_STRATEGY = "leiden"`)

| Aspect | Detail |
|---|---|
| Algorithm | Leiden (successor to Louvain) on a weighted similarity graph |
| Input | Combined similarity matrix: 70 % cosine(embedding) + 30 % Jaccard(entity overlap) |
| Resolution | Layer-adaptive: `resolution = 1.0 × 0.7^layer` → broader clusters at higher layers |
| Dependencies | `python-igraph`, `leidenalg` |
| Strengths | Produces high-modularity partitions; naturally determines cluster count; respects graph structure |
| Weaknesses | Hard assignment (no soft membership); requires building a graph first |
| Fit for Nexus | **Excellent** – entity graph is already built; community structure matches document sections |

### 2.2 Gaussian Mixture Model (soft clustering, RAPTOR-style: `"gmm"`)

| Aspect | Detail |
|---|---|
| Algorithm | sklearn `GaussianMixture` with BIC-based automatic component selection |
| Input | UMAP-reduced embeddings (entity similarity used indirectly through UMAP neighbourhood) |
| Strengths | Soft assignments (probability per cluster); well-studied; deterministic with `random_state=42` |
| Weaknesses | Assumes Gaussian distribution; BIC search adds overhead; no entity awareness in cluster assignments |
| Fit for Nexus | **Good fallback** when Leiden is unavailable; useful for research comparison |

### 2.3 HDBSCAN (density-based: `"hdbscan"`)

| Aspect | Detail |
|---|---|
| Algorithm | `hdbscan.HDBSCAN` on UMAP-reduced embeddings |
| Noise handling | Noise points reassigned to nearest cluster by entity similarity |
| Strengths | No assumed shape; automatically finds cluster count; robust to outliers |
| Weaknesses | Can produce many noise points with small datasets; sensitive to `min_cluster_size` |
| Fit for Nexus | **Good for large corpora** (1 k+ chunks) where density structure emerges clearly |

---

## 3. Combined Similarity Matrix

All strategies benefit from the **entity-aware similarity matrix**:

```
combined_sims = 0.7 × cosine_similarity(embeddings) + 0.3 × jaccard_entity_overlap
```

- **70 % embedding**: captures semantic relatedness (topic, meaning)
- **30 % entity overlap**: ensures factually connected chunks cluster together even when phrased differently

This weighting was chosen empirically: entity overlap is a strong signal but noisy for short chunks.

---

## 4. UMAP Dimensionality Reduction

Before clustering, high-dimensional embeddings (768-d from Gemini) are reduced via UMAP:

| Parameter | Value | Rationale |
|---|---|---|
| `n_components` | 10 | Enough to preserve structure; low enough for GMM |
| `n_neighbors` | 15 | Controls local vs global structure preservation |
| `min_dist` | 0.1 | Allows tight clusters |
| `metric` | cosine | Matches embedding space |

UMAP preserves local neighbourhood structure better than PCA, producing tighter clusters (per RAPTOR paper).  If `umap-learn` is not installed, clustering runs directly on raw embeddings.

---

## 5. Why Leiden is the Default

1. **Entity graph is already built**: T-Retriever constructs an `EntityGraph`; Leiden operates natively on graphs.
2. **Automatic cluster count**: No need to specify k; the resolution parameter controls granularity.
3. **Layer-adaptive resolution**: Decreasing resolution at higher layers naturally produces broader summaries.
4. **Quality**: Leiden maximises modularity (provably better than Louvain).
5. **Performance**: Near-linear in edges; handles 10 k+ nodes comfortably.

---

## 6. Configuration

All clustering parameters are configurable in `t_retriever.py`:

```python
CLUSTERING_STRATEGY = "leiden"       # "leiden" | "gmm" | "hdbscan"
MIN_CLUSTER_SIZE = 2                 # Minimum nodes per cluster
MIN_NODES_FOR_CLUSTERING = 3         # Below this → single cluster
GRAPH_EDGE_SIMILARITY_THRESHOLD = 0.3  # Minimum edge weight
UMAP_N_COMPONENTS = 10               # UMAP target dimensions
```

To override at runtime, pass `strategy=` to `cluster_with_entities()`.

---

## 7. Scaling Guidance

| Corpus size | Recommended strategy | Notes |
|---|---|---|
| < 50 chunks | GMM | Small datasets benefit from soft clustering |
| 50 – 1 000 chunks | Leiden (default) | Community detection handles this range well |
| 1 000 – 50 000 chunks | Leiden or HDBSCAN | HDBSCAN may find tighter density clusters |
| > 50 000 chunks | HDBSCAN with per-document scope | Avoid building full similarity matrix; use approximate methods |
