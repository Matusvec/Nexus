# Nexus RAG System - Academic Sources & References

This document lists all research papers and academic sources that informed the implementation of the Nexus RAG system.

---

## 🎓 Core Papers

### 1. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

**Citation:**
```
Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C. D. (2024).
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
International Conference on Learning Representations (ICLR 2024).
arXiv:2401.18059
```

**Link:** https://arxiv.org/abs/2401.18059

**Contributions used in Nexus:**
- ✅ Hierarchical tree structure with multi-layer summaries
- ✅ Soft clustering with GMM allowing nodes in multiple clusters
- ✅ UMAP dimensionality reduction before clustering
- ✅ BIC-based optimal cluster count detection
- ✅ Collapsed tree retrieval (query all layers simultaneously)
- ✅ Layer-weighted scoring

**Files implementing this:**
- `backend/raptor.py` - Tree building, clustering, summarization
- `backend/query.py` - Collapsed tree retrieval

---

### 2. LATTICE: LLM-guided Hierarchical Retrieval and Reasoning

**Citation:**
```
Gupta, N., Chang, W. C., Bui, N., Hsieh, C. J., & Dhillon, I. S. (2025).
LATTICE: LLM-guided Hierarchical Retrieval and Reasoning for Complex Queries.
arXiv:2510.13217
```

**Link:** https://arxiv.org/abs/2510.13217

**Contributions used in Nexus:**
- ✅ Path relevance score aggregation concept
- 🔄 LLM-guided tree traversal (planned for future)
- 🔄 Score calibration for cross-branch comparison (planned)

**Key insight:** Using the LLM as an active search agent during retrieval improves complex query handling.

---

### 3. BookRAG: Hierarchical Structure-aware RAG for Long Documents

**Citation:**
```
Wang, S., Zhou, Y., & Fang, Y. (2025).
BookRAG: Hierarchical Structure-aware Index and Retrieval for Long Documents.
arXiv:2512.03413
```

**Link:** https://arxiv.org/abs/2512.03413

**Contributions used in Nexus:**
- ✅ Query classification (simple vs complex vs exploratory)
- ✅ Adaptive retrieval strategy based on query type
- ✅ Information Foraging Theory-based retrieval
- 🔄 Knowledge graph integration (planned for future)
- 🔄 Entity extraction and linking (planned)

**Files implementing this:**
- `backend/query.py` - `classify_query()`, `adaptive_retrieval()`

---

### 4. T-Retriever: Tree-based Hierarchical RAG with Semantic-Structural Optimization

**Citation:**
```
Wei, C., Qin, H., He, S., Wang, Y., & Chen, Y. (2026).
T-Retriever: Tree-based Hierarchical Retrieval for Textual Graphs with Semantic-Structural Entropy.
arXiv:2601.04945
```

**Link:** https://arxiv.org/abs/2601.04945

**Contributions considered for Nexus:**
- 🔄 S²-Entropy metric (structural + semantic) (future consideration)
- 🔄 Top-down partitioning approach (future consideration)
- 🔄 GNN-augmented generation (future consideration)

**Key insight:** Jointly optimizing structural and semantic properties during tree construction improves retrieval quality.

---

### 5. Enhanced RAPTOR with Semantic Chunking and Adaptive Graph Clustering

**Citation:**
```
Liu, Y., Xie, X., Wan, X., Pan, Y., & Wang, C. (2026).
Enhancing RAPTOR with Semantic Chunking and Adaptive Graph Clustering.
Frontiers in Computer Science, 7, 1710121.
DOI: 10.3389/fcomp.2025.1710121
```

**Link:** https://www.frontiersin.org/articles/10.3389/fcomp.2025.1710121

**Contributions used in Nexus:**
- ✅ Leiden algorithm for clustering (alternative to GMM)
- ✅ Layer-aware adaptive resolution parameters
- ✅ Graph-based similarity construction for clustering
- ✅ Configurable clustering method selection

**Files implementing this:**
- `backend/raptor.py` - `cluster_with_leiden()`, `get_layer_resolution()`, `cluster_adaptive()`

---

## 📚 Additional References

### Contextual Retrieval (Anthropic)

**Source:** Anthropic Research Blog (2024)
**Link:** https://www.anthropic.com/news/contextual-retrieval

**Contributions used in Nexus:**
- ✅ Document-level context prepended to chunks
- ✅ LLM-generated contextual descriptions for embeddings
- ✅ Improved embedding quality through context

**Files implementing this:**
- `backend/chunking.py` - `contextualize_chunks()`

---

### Semantic Chunking

**Based on:** Various research on boundary detection and semantic coherence

**Contributions used in Nexus:**
- ✅ Embedding-based similarity for chunk boundaries
- ✅ Sentence-aware splitting
- ✅ Overlap tokens for context preservation

**Files implementing this:**
- `backend/chunking.py` - `chunk_text()`

---

## 🛠️ Implementation Summary

| Feature | Source Paper | Status |
|---------|--------------|--------|
| Hierarchical tree structure | RAPTOR | ✅ Implemented |
| GMM soft clustering | RAPTOR | ✅ Implemented |
| UMAP dimensionality reduction | RAPTOR | ✅ Implemented |
| BIC cluster optimization | RAPTOR | ✅ Implemented |
| Collapsed tree retrieval | RAPTOR | ✅ Implemented |
| Leiden clustering | Frontiers 2026 | ✅ Implemented |
| Layer-aware resolution | Frontiers 2026 | ✅ Implemented |
| Query classification | BookRAG | ✅ Implemented |
| Adaptive retrieval | BookRAG | ✅ Implemented |
| Contextual embeddings | Anthropic | ✅ Implemented |
| LLM-guided traversal | LATTICE | 🔄 Planned |
| Knowledge graph | BookRAG | 🔄 Planned |
| S²-Entropy optimization | T-Retriever | 🔄 Considered |

---

## 📖 Recommended Reading Order

For understanding the Nexus implementation:

1. **Start with RAPTOR** - Core architecture
2. **Then Frontiers 2026** - Clustering improvements
3. **Then BookRAG** - Query handling
4. **Then LATTICE** - Advanced traversal concepts
5. **Finally T-Retriever** - Future optimization ideas

---

## 📝 BibTeX References

```bibtex
@inproceedings{sarthi2024raptor,
  title={RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval},
  author={Sarthi, Parth and Abdullah, Salman and Tuli, Aditi and Khanna, Shubh and Goldie, Anna and Manning, Christopher D},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@article{gupta2025lattice,
  title={LATTICE: LLM-guided Hierarchical Retrieval and Reasoning for Complex Queries},
  author={Gupta, Nilesh and Chang, Wei-Cheng and Bui, Ngot and Hsieh, Cho-Jui and Dhillon, Inderjit S},
  journal={arXiv preprint arXiv:2510.13217},
  year={2025}
}

@article{wang2025bookrag,
  title={BookRAG: Hierarchical Structure-aware Index and Retrieval for Long Documents},
  author={Wang, Shu and Zhou, Yingli and Fang, Yixiang},
  journal={arXiv preprint arXiv:2512.03413},
  year={2025}
}

@article{wei2026tretriever,
  title={T-Retriever: Tree-based Hierarchical Retrieval for Textual Graphs},
  author={Wei, Chunyu and Qin, Huaiyu and He, Siyuan and Wang, Yunhai and Chen, Yueguo},
  journal={arXiv preprint arXiv:2601.04945},
  year={2026}
}

@article{liu2026enhanced,
  title={Enhancing RAPTOR with Semantic Chunking and Adaptive Graph Clustering},
  author={Liu, Yan and Xie, Xiaodong and Wan, Xin and Pan, Yi and Wang, Cheng},
  journal={Frontiers in Computer Science},
  volume={7},
  pages={1710121},
  year={2026},
  doi={10.3389/fcomp.2025.1710121}
}
```

---

*Last updated: January 27, 2026*
