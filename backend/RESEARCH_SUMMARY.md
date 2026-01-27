# Research Papers Analysis for Nexus RAG System

This document summarizes the key findings from 5 research papers and identifies actionable improvements for the Nexus project.

---

## 📚 Paper Summaries

### Paper 1: RAPTOR - Recursive Abstractive Processing for Tree-Organized Retrieval
**Source:** arXiv:2401.18059 | ICLR 2024  
**Authors:** Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning (Stanford)

#### Key Contributions
- **Hierarchical tree structure** capturing both high-level themes AND low-level details
- **Soft clustering with GMM** - nodes can belong to MULTIPLE clusters (key innovation!)
- **UMAP dimensionality reduction** before clustering
- **Two retrieval methods**: Tree Traversal and Collapsed Tree (collapsed performs better)
- **20% absolute accuracy improvement** on QuALITY benchmark

#### Technical Details
| Component | Implementation |
|-----------|----------------|
| Clustering | GMM + UMAP with BIC for optimal k |
| Summarization | GPT-3.5-turbo |
| Embeddings | SBERT `multi-qa-mpnet-base-cos-v1` |
| Chunk size | 100 tokens with sentence boundaries |
| Retrieval | Collapsed tree, ~2000 tokens context |

#### Status in Nexus
✅ **Already implemented** - Core RAPTOR with GMM + UMAP clustering

---

### Paper 2: LATTICE - LLM-guided Hierarchical Retrieval
**Source:** arXiv:2510.13217  
**Authors:** Nilesh Gupta, Wei-Cheng Chang, Ngot Bui, Cho-Jui Hsieh, Inderjit S. Dhillon (UT Austin, UCLA, Google)

#### Key Contributions
- **LLM as active search agent** - navigates the tree during retrieval, not just retrieves
- **Calibrated path relevance scores** - converts local LLM judgments into globally coherent signals
- **Logarithmic search complexity** - scales to 420K+ documents
- **9% improvement in Recall@100** on BRIGHT benchmark (zero-shot)

#### Technical Details
```
Path Relevance Score:
p̂_rel(v) = (1-α) · s^calibrated(v) + α · p̂_rel(parent(v))

Where:
- α = 0.5 (momentum factor)
- s^calibrated = LLM score normalized against siblings
```

**Score Calibration Method:**
1. Augment comparison slate with high-relevance siblings
2. Include previously visited leaves for reference
3. Normalize scores across branches

**Tree Construction Options:**
- Bottom-up: Agglomerative clustering (like RAPTOR)
- Top-down: Divisive clustering with multi-level summaries

#### Actionable for Nexus
🆕 **Can implement:**
1. LLM-guided tree traversal during query time
2. Score calibration for cross-branch comparison
3. Path relevance aggregation with momentum

---

### Paper 3: BookRAG - Hierarchical Structure-aware Index
**Source:** arXiv:2512.03413  
**Authors:** Shu Wang, Yingli Zhou, Yixiang Fang (CUHK Shenzhen)

#### Key Contributions
- **BookIndex** = Hierarchical Tree + Knowledge Graph combined
- **Entity-node mapping** - KG entities linked to tree nodes
- **Agent-based query workflow** using Information Foraging Theory
- **Dynamic query classification** - adapts retrieval strategy to query type
- **Layout-aware parsing** - preserves document structure

#### Technical Details

**Index Structure:**
```
                    [Document Root]
                         │
         ┌───────────────┼───────────────┐
    [Chapter 1]     [Chapter 2]     [Chapter 3]
         │               │               │
    [Sections...]   [Sections...]   [Sections...]
         │
    [Leaf chunks with entity annotations]
         │
         ▼
    [Knowledge Graph: entities + relationships]
```

**Query Classification:**
| Query Type | Strategy |
|------------|----------|
| Simple lookup | Direct tree traversal |
| Multi-hop | KG traversal + tree retrieval |
| Comparative | Multiple branch retrieval |

#### Actionable for Nexus
🆕 **Can implement:**
1. Knowledge graph layer alongside tree
2. Entity extraction and disambiguation
3. Query classification for adaptive retrieval
4. Enhanced document structure parsing

---

### Paper 4: T-Retriever - Tree-based Hierarchical RAG for Textual Graphs
**Source:** arXiv:2601.04945  
**Authors:** Chunyu Wei, Huaiyu Qin, Siyuan He, Yunhai Wang, Yueguo Chen (Renmin University of China)

#### Key Contributions
- **Semantic-Structural Entropy (S²-Entropy)** - optimizes structure AND semantics jointly
- **Adaptive Compression Encoding** - top-down Shannon-Fano inspired partitioning
- **Catalytic effect** - groups semantically similar but structurally distant nodes
- **GNN-augmented generation** - Graph Neural Networks enhance prompting
- **6.63% improvement** on BookGraphs dataset

#### Technical Details

**S²-Entropy Formula:**
```
H_S²(G; α) = H_T(G; α) + λ · H_sem(V_α)

Where:
- H_T = structural entropy (tree-based)
- H_sem = semantic density entropy (KDE-based)
- λ = balance parameter
- α = tree partition
```

**Tree Construction (Top-Down):**
1. **Partition**: Split nodes to minimize S²-Entropy (binary splits)
2. **Prune**: Remove nodes that minimize entropy increase
3. **Regulate**: Insert intermediate nodes for tree structure

**GNN-Augmented Retrieval:**
```
Graph → GNN Encoder → Pooling → MLP → Soft Prompt → LLM
```

#### Actionable for Nexus
🆕 **Can implement:**
1. S²-Entropy metric for tree optimization
2. Top-down partitioning (alternative to bottom-up GMM)
3. Entropy-based pruning for compact trees

---

### Paper 5: Enhancing RAPTOR with Semantic Chunking and Adaptive Graph Clustering
**Source:** Frontiers in Computer Science 2026 | DOI: 10.3389/fcomp.2025.1710121  
**Authors:** Yan Liu, Xiaodong Xie, Xin Wan, Yi Pan, Cheng Wang (Huaqiao University, Changsha University)

#### Key Contributions
- **Semantic Segmentation** replaces fixed-token chunking
- **Adaptive Graph Clustering (AGC)** using Leiden algorithm instead of GMM
- **Layer-aware dual-adaptive parameters** - granularity adapts to tree depth
- **76% reduction in summary nodes** - much more compact hierarchy
- **65.5% accuracy on QuALITY** - outperforms original RAPTOR (62.2%)

#### Technical Details

**Leiden Algorithm Advantages over GMM:**
| Aspect | GMM | Leiden |
|--------|-----|--------|
| Cluster shape | Elliptical Gaussians | Arbitrary graph communities |
| Scalability | O(n²) | O(n log n) |
| Parameter tuning | n_clusters via BIC | Resolution parameter |
| Overlap handling | Soft probabilities | Community detection |

**Layer-Aware Adaptive Parameters:**
```python
# Resolution decreases at higher layers (broader clusters)
resolution_l = base_resolution * decay_factor^layer

# Threshold adapts to semantic density
threshold_l = adaptive_threshold(layer, semantic_density)
```

**Graph Construction:**
```
1. Compute pairwise cosine similarity between chunk embeddings
2. Build k-NN graph (or threshold-based)
3. Apply Leiden community detection
4. Use detected communities as clusters
```

#### Actionable for Nexus
🆕 **High priority to implement:**
1. **Replace GMM with Leiden algorithm** for clustering
2. **Layer-aware adaptive parameters** (resolution decay per layer)
3. **Graph-topological clustering** instead of density-based

---

## 🎯 Implementation Priority Matrix

| Enhancement | Source Paper | Impact | Complexity | Priority |
|-------------|--------------|--------|------------|----------|
| Leiden clustering | Paper 5 | High (76% node reduction) | Medium | ⭐⭐⭐ HIGH |
| Layer-aware params | Paper 5 | Medium | Low | ⭐⭐⭐ HIGH |
| LLM-guided traversal | Paper 2 | High | Medium | ⭐⭐ MEDIUM |
| Score calibration | Paper 2 | Medium | Medium | ⭐⭐ MEDIUM |
| Query classification | Paper 3 | Medium | Low | ⭐⭐ MEDIUM |
| Knowledge graph | Paper 3 | High | High | ⭐ LOW (future) |
| S²-Entropy optimization | Paper 4 | Medium | High | ⭐ LOW (future) |
| GNN-augmented gen | Paper 4 | Medium | High | ⭐ LOW (future) |

---

## 🔧 Recommended Implementation Roadmap

### Phase 1: Clustering Upgrade (This Sprint)
1. ✅ Keep GMM as fallback
2. 🆕 Add Leiden algorithm option
3. 🆕 Implement layer-aware resolution decay
4. 🆕 Add graph-based similarity computation

### Phase 2: Retrieval Enhancement (Next Sprint)
1. 🆕 Add query classification (simple vs complex)
2. 🆕 Implement LLM-guided tree traversal option
3. 🆕 Add score calibration for cross-branch queries

### Phase 3: Advanced Features (Future)
1. 🆕 Knowledge graph integration
2. 🆕 Entity extraction and linking
3. 🆕 S²-Entropy tree optimization
4. 🆕 Multi-hop reasoning support

---

## 📊 Expected Improvements

Based on the papers, implementing these enhancements should yield:

| Metric | Current (RAPTOR) | Expected (Enhanced) | Source |
|--------|------------------|---------------------|--------|
| QuALITY accuracy | ~62% | ~65.5% | Paper 5 |
| Summary node count | 100% baseline | 24% (76% reduction) | Paper 5 |
| Recall@100 | baseline | +9% | Paper 2 |
| Scalability | O(n²) GMM | O(n log n) Leiden | Paper 5 |

---

## 📖 Quick Reference: Algorithm Comparison

### Clustering Methods
```
GMM (Current):
- Pros: Soft clustering, probabilistic
- Cons: O(n²), assumes Gaussian distributions

Leiden (Recommended):
- Pros: O(n log n), arbitrary community shapes, graph-native
- Cons: Harder to get soft assignments (need post-processing)

S²-Entropy (Advanced):
- Pros: Joint semantic-structural optimization
- Cons: Complex implementation, requires careful tuning
```

### Retrieval Methods
```
Collapsed Tree (Current):
- Flatten all layers, retrieve by similarity
- Simple but effective

LLM-Guided Traversal (Enhancement):
- LLM decides which branches to explore
- Better for complex queries, higher latency

Hybrid (Future):
- Classify query → choose retrieval method
- Best of both worlds
```
