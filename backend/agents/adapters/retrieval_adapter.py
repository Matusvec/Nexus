"""
Retrieval Adapter — bridges the agentic system to the T-Retrieval backend.

When the T-retrieval branch modules are present, this adapter delegates
directly.  When they are absent (pre-merge), it falls back to a mock
implementation so the agentic system can still be developed and demoed.

Feature flag:  NEXUS_MOCK_RETRIEVAL=1  forces mock mode even if real
modules are available (useful for testing).
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Optional, Any

log = logging.getLogger("nexus.adapters.retrieval")

# ── Feature flag ──────────────────────────────────────────────────

_FORCE_MOCK = os.getenv("NEXUS_MOCK_RETRIEVAL", "0") == "1"


def _real_modules_available() -> bool:
    """Check whether the T-retrieval branch modules are importable."""
    try:
        import t_query  # noqa: F401
        import storage  # noqa: F401
        return True
    except Exception:
        return False


MOCK_MODE = _FORCE_MOCK or not _real_modules_available()

if MOCK_MODE:
    log.warning(
        "⚠️  RETRIEVAL ADAPTER: Running in MOCK mode. "
        "T-retrieval modules not found or NEXUS_MOCK_RETRIEVAL=1. "
        "RAG results will be synthetic."
    )
else:
    log.info("✅ RETRIEVAL ADAPTER: T-retrieval modules detected — using real retrieval.")


# ── Public interface ──────────────────────────────────────────────
# All agents call these functions.  They NEVER import t_query directly.


def query(
    query_text: str,
    *,
    top_k: int = 5,
    document_id: Optional[str] = None,
    layer: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Search the RAG hierarchy.

    Returns a list of result dicts with keys:
        id, text, layer, score, document_id, metadata
    """
    if MOCK_MODE:
        return _mock_query(query_text, top_k=top_k, document_id=document_id, layer=layer)
    return _real_query(query_text, top_k=top_k, document_id=document_id, layer=layer)


def explain(
    query_text: str,
    result_ids: List[str],
) -> Dict[str, Any]:
    """
    Explain *why* certain chunks were retrieved (retrieval path through
    the tree/graph hierarchy).

    Returns:
        {
            "query_entities": [...],
            "retrieval_path": [...],
            "layer_distribution": {0: n, 1: m, ...},
            "explanation": "human-readable summary"
        }
    """
    if MOCK_MODE:
        return _mock_explain(query_text, result_ids)
    return _real_explain(query_text, result_ids)


def list_documents() -> List[Dict[str, Any]]:
    """List all documents in the knowledge base."""
    if MOCK_MODE:
        return _mock_list_documents()
    return _real_list_documents()


def get_document_summary(document_id: str) -> Dict[str, Any]:
    """Get summary + stats for a specific document."""
    if MOCK_MODE:
        return _mock_document_summary(document_id)
    return _real_document_summary(document_id)


def get_stats() -> Dict[str, Any]:
    """Get overall knowledge-base statistics."""
    if MOCK_MODE:
        return {"total_chunks": 0, "documents": [], "layers": {}, "mock": True}
    return _real_stats()


# ── Real implementations (delegate to T-retrieval) ────────────────


def _real_query(query_text, *, top_k, document_id, layer):
    from t_query import collapsed_tree_retrieval
    results = collapsed_tree_retrieval(
        query=query_text,
        document_id=document_id,
        top_k=top_k,
    )
    # If layer-specific search requested, filter
    if layer is not None:
        results = [r for r in results if r.get("layer") == layer]
    return results


def _real_explain(query_text, result_ids):
    from t_query import extract_query_entities
    from storage import get_or_create_collection

    entities = extract_query_entities(query_text)
    collection = get_or_create_collection()

    # Gather layer distribution
    layer_dist: Dict[int, int] = {}
    retrieval_path = []
    for rid in result_ids:
        try:
            result = collection.get(ids=[rid], include=["metadatas"])
            if result and result["metadatas"]:
                meta = result["metadatas"][0]
                layer = meta.get("layer", 0)
                layer_dist[layer] = layer_dist.get(layer, 0) + 1
                retrieval_path.append({
                    "chunk_id": rid,
                    "layer": layer,
                    "document_id": meta.get("document_id", ""),
                    "parent_ids": meta.get("parent_ids", []),
                })
        except Exception:
            pass

    return {
        "query_entities": entities,
        "retrieval_path": retrieval_path,
        "layer_distribution": layer_dist,
        "explanation": (
            f"Query matched entities: {entities}. "
            f"Retrieved {len(result_ids)} chunks across layers: {dict(sorted(layer_dist.items()))}."
        ),
    }


def _real_list_documents():
    from storage import get_collection_stats
    stats = get_collection_stats()
    docs = []
    for doc_id in stats.get("documents", []):
        docs.append({"document_id": doc_id})
    return docs


def _real_document_summary(document_id):
    from storage import get_or_create_collection
    collection = get_or_create_collection()
    results = collection.get(
        where={"document_id": document_id},
        include=["metadatas", "documents"],
    )
    layers: Dict[int, int] = {}
    for meta in results.get("metadatas", []):
        layer = meta.get("layer", 0)
        layers[layer] = layers.get(layer, 0) + 1

    return {
        "document_id": document_id,
        "chunk_count": len(results.get("ids", [])),
        "layers": layers,
    }


def _real_stats():
    from storage import get_collection_stats
    return get_collection_stats()


# ── Mock implementations ──────────────────────────────────────────


def _mock_query(query_text, *, top_k, document_id, layer):
    """Return synthetic results for demo / pre-merge development."""
    results = []
    for i in range(min(top_k, 3)):
        results.append({
            "id": f"mock_chunk_{i}",
            "text": f"[MOCK] Synthetic result {i+1} for query: '{query_text}'",
            "document": f"[MOCK] Synthetic result {i+1} for query: '{query_text}'",
            "layer": layer if layer is not None else i % 3,
            "score": round(0.95 - i * 0.1, 3),
            "document_id": document_id or "mock_doc",
        })
    return results


def _mock_explain(query_text, result_ids):
    return {
        "query_entities": [query_text.split()[0]] if query_text else [],
        "retrieval_path": [{"chunk_id": rid, "layer": 0} for rid in result_ids],
        "layer_distribution": {0: len(result_ids)},
        "explanation": f"[MOCK] Explain path for {len(result_ids)} chunks (mock mode).",
    }


def _mock_list_documents():
    return [
        {"document_id": "mock_doc_1"},
        {"document_id": "mock_doc_2"},
    ]


def _mock_document_summary(document_id):
    return {
        "document_id": document_id,
        "chunk_count": 10,
        "layers": {0: 7, 1: 2, 2: 1},
        "mock": True,
    }
