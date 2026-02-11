"""
Retrieval API – stable interface for T-Retriever operations.

Designed for downstream consumers (agentic AI, frontend, AR/VR clients).
Decoupled from CLI and UI logic.

Public functions:
    add_documents(paths, **kwargs) -> List[Dict]
    remove_documents(document_ids)  -> List[Dict]
    query(question, **kwargs)       -> Dict
    explain_retrieval(question, **kwargs) -> Dict
    get_tree_info(document_id)      -> Dict
    update_documents(document_ids, paths) -> List[Dict]
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Internal imports (lazy where possible to keep import fast)
# ---------------------------------------------------------------------------

def _import_document_manager():
    from document_manager import (
        add_document,
        remove_document,
        list_documents,
        rebuild_document_tree,
    )
    return add_document, remove_document, list_documents, rebuild_document_tree


def _import_query():
    from t_query import (
        adaptive_retrieval,
        hybrid_retrieval,
        collapsed_tree_retrieval,
        answer_question,
        build_context_from_results,
        deduplicate_results,
        classify_query,
        extract_query_entities,
    )
    return (
        adaptive_retrieval,
        hybrid_retrieval,
        collapsed_tree_retrieval,
        answer_question,
        build_context_from_results,
        deduplicate_results,
        classify_query,
        extract_query_entities,
    )


def _import_retriever():
    from t_retriever import get_tree_stats, load_document_graph
    return get_tree_stats, load_document_graph


# ============================================================================
# ADD DOCUMENTS
# ============================================================================

def add_documents(
    paths: List[str],
    *,
    build_tree: bool = True,
    collection_name: str = "nexus_chunks",
    **kwargs,
) -> List[Dict]:
    """
    Ingest one or more documents into the T-Retriever system.

    Each path goes through: parse → chunk → contextualise → store → build tree.

    Args:
        paths:           List of file paths (PDF / DOCX).
        build_tree:      Build the hierarchical tree after storing.
        collection_name: ChromaDB collection name.
        **kwargs:        Forwarded to ``document_manager.add_document``
                         (similarity_threshold, min_tokens, max_tokens, etc.).

    Returns:
        List of result dicts, one per document.
    """
    _add, _, _, _ = _import_document_manager()
    results: List[Dict] = []
    for p in paths:
        result = _add(
            str(p),
            build_tree=build_tree,
            collection_name=collection_name,
            **kwargs,
        )
        results.append(result)
    return results


# ============================================================================
# REMOVE DOCUMENTS
# ============================================================================

def remove_documents(
    document_ids: List[str],
    *,
    collection_name: str = "nexus_chunks",
) -> List[Dict]:
    """
    Fully remove documents (all layers + graph + cache).

    Args:
        document_ids:    List of document identifiers.
        collection_name: ChromaDB collection name.

    Returns:
        List of result dicts with deletion counts.
    """
    _, _remove, _, _ = _import_document_manager()
    results: List[Dict] = []
    for doc_id in document_ids:
        result = _remove(doc_id, collection_name=collection_name)
        results.append(result)
    return results


# ============================================================================
# QUERY
# ============================================================================

def query(
    question: str,
    *,
    document_id: Optional[str] = None,
    top_k: int = 10,
    collection_name: str = "nexus_chunks",
    show_sources: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Answer a question using T-Retriever hybrid retrieval.

    Returns a dict with:
        answer:               Generated text answer.
        sources:              List of source references.
        query_classification:  Query type & strategy used.
    """
    (
        _adaptive,
        _hybrid,
        _tree,
        _answer,
        _build_ctx,
        _dedup,
        _classify,
        _extract_ents,
    ) = _import_query()

    result = _answer(
        question=question,
        document_id=document_id,
        top_k=top_k,
        collection_name=collection_name,
        show_sources=show_sources,
        verbose=verbose,
    )
    return result


# ============================================================================
# EXPLAIN RETRIEVAL
# ============================================================================

def explain_retrieval(
    question: str,
    *,
    document_id: Optional[str] = None,
    top_k: int = 10,
    collection_name: str = "nexus_chunks",
) -> Dict:
    """
    Return a detailed explanation of how retrieval was performed.

    This is the key **explainability** endpoint.  It exposes:
    - query classification (type, confidence, entity extraction)
    - retrieval path (which steps ran, how many results each produced)
    - per-result provenance (layer, source method, fusion score, entity matches)

    Returns:
        Dict with query_classification, retrieval_path, and annotated results.
    """
    (
        _adaptive,
        _hybrid,
        _tree,
        _answer,
        _build_ctx,
        _dedup,
        _classify,
        _extract_ents,
    ) = _import_query()
    _get_tree_stats, _load_graph = _import_retriever()

    # Step 1 – classify
    classification = _classify(question)
    strategy = classification["strategy"]
    query_entities = classification["query_entities"]

    retrieval_path: List[Dict] = []

    # Step 2 – tree retrieval
    tree_results = _tree(
        query=question,
        document_id=document_id,
        top_k=top_k,
        collection_name=collection_name,
        query_entities=query_entities,
    )
    layers_searched = sorted({r["layer"] for r in tree_results})
    retrieval_path.append({
        "step": "tree_retrieval",
        "results": len(tree_results),
        "layers_searched": layers_searched,
    })

    # Step 3 – graph expansion (if complex / multi-hop)
    graph_results: List[Dict] = []
    entity_results: List[Dict] = []

    if strategy.get("use_graph") and document_id:
        from t_query import graph_expansion_retrieval, entity_based_retrieval

        seed_ids = [r["id"] for r in tree_results[:5]]
        graph_results = graph_expansion_retrieval(
            seed_chunk_ids=seed_ids,
            document_id=document_id,
            collection_name=collection_name,
        )
        retrieval_path.append({
            "step": "graph_expansion",
            "seeds": len(seed_ids),
            "expanded": len(graph_results),
            "hops": strategy.get("graph_hops", 2),
        })

        entity_results = entity_based_retrieval(
            query_entities=query_entities,
            document_id=document_id,
            top_k=5,
            collection_name=collection_name,
        )
        retrieval_path.append({
            "step": "entity_retrieval",
            "entities": query_entities,
            "matches": len(entity_results),
        })

    # Step 4 – fusion
    all_ids = set()
    for r in tree_results + graph_results + entity_results:
        all_ids.add(r["id"])
    retrieval_path.append({
        "step": "fusion",
        "total_candidates": len(all_ids),
        "final": min(len(all_ids), top_k),
    })

    # Build annotated results
    annotated: List[Dict] = []
    for r in tree_results[:top_k]:
        entry = {
            "id": r["id"],
            "layer": r.get("layer", 0),
            "source": r.get("source", "tree"),
            "fusion_score": r.get("fusion_score", r.get("adjusted_distance", None)),
            "entity_matches": [],
            "text_preview": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
        }
        # Check entity matches
        chunk_entities = r.get("metadata", {}).get("entity_names", "").lower()
        entry["entity_matches"] = [
            e for e in query_entities if e.lower() in chunk_entities
        ]
        annotated.append(entry)

    return {
        "query_classification": classification,
        "retrieval_path": retrieval_path,
        "results": annotated,
    }


# ============================================================================
# TREE INFO
# ============================================================================

def get_tree_info(
    document_id: str,
    *,
    collection_name: str = "nexus_chunks",
) -> Dict:
    """
    Inspect the hierarchical tree for a specific document.

    Returns tree depth, node counts per layer, entity count, graph edge count.
    """
    _get_tree_stats, _ = _import_retriever()
    return _get_tree_stats(document_id, collection_name)


# ============================================================================
# UPDATE DOCUMENTS (remove + re-add)
# ============================================================================

def update_documents(
    document_ids: List[str],
    paths: List[str],
    *,
    collection_name: str = "nexus_chunks",
    **kwargs,
) -> List[Dict]:
    """
    Update documents by removing old versions and re-ingesting new files.

    ``document_ids[i]`` is replaced by ``paths[i]``.  The operation is
    sequential (remove then add) per document.

    Args:
        document_ids:    Document IDs to replace.
        paths:           New file paths (same length as document_ids).
        collection_name: ChromaDB collection name.
        **kwargs:        Forwarded to ``add_document``.

    Returns:
        List of add-result dicts.
    """
    if len(document_ids) != len(paths):
        raise ValueError("document_ids and paths must have the same length")

    _add, _remove, _, _ = _import_document_manager()

    results: List[Dict] = []
    for doc_id, path in zip(document_ids, paths):
        _remove(doc_id, collection_name=collection_name)
        result = _add(str(path), collection_name=collection_name, **kwargs)
        results.append(result)
    return results
