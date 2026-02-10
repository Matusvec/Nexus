"""
Document Manager - Clean API for adding and removing documents
in the T-Retriever hierarchical RAG system.

Provides a unified lifecycle for documents:
  add_document   : parse → chunk → contextualize → store → build tree
  remove_document: delete all layers + graph metadata + cache cleanup
  list_documents : enumerate documents with tree status
  rebuild_document_tree: re-cluster and re-summarize existing chunks
"""
from pathlib import Path
from typing import Dict, List, Optional

from storage import (
    get_or_create_collection,
    get_collection_stats,
    store_contextualized_chunks,
    delete_document_chunks,
)
from t_retriever import (
    build_tretriever_tree,
    delete_tree_layers,
    get_tree_stats,
    save_document_graph,
    get_document_graph,
    _document_graphs,
)


# ============================================================================
# ADD DOCUMENT
# ============================================================================

def add_document(
    file_path: str,
    *,
    build_tree: bool = True,
    similarity_threshold: float = 0.7,
    min_tokens: int = 100,
    max_tokens: int = 500,
    group_size: int = 2,
    overlap_tokens: int = 50,
    use_llm_context: bool = False,
    max_tree_layers: int = 3,
    collection_name: str = "nexus_chunks",
) -> Dict:
    """
    Full pipeline: parse → chunk → contextualize → store → build tree.

    Args:
        file_path:            Path to PDF / DOCX file.
        build_tree:           Whether to build the T-Retriever tree after storing.
        similarity_threshold: Cosine sim threshold for semantic chunking.
        min_tokens:           Minimum tokens per chunk.
        max_tokens:           Maximum tokens per chunk.
        group_size:           Sentence group size for chunking.
        overlap_tokens:       Token overlap between chunks.
        use_llm_context:      Use LLM for per-chunk context (slower).
        max_tree_layers:      Maximum tree depth when building.
        collection_name:      ChromaDB collection name.

    Returns:
        Dict with document_id, chunk_count, tree_stats (if built).
    """
    from document_parser import parse_document
    from chunking import chunk_text, contextualize_chunks
    from gemini_client import generate_document_summary

    file_path = str(file_path)

    # 1. Parse
    print(f"[ADD] Parsing {Path(file_path).name}...")
    parsed = parse_document(file_path)

    # 2. Generate summary
    print("[ADD] Generating document summary...")
    doc_summary = generate_document_summary(
        parsed["text"], parsed["metadata"]["filename"]
    )

    # 3. Chunk
    print("[ADD] Chunking text...")
    raw_chunks = chunk_text(
        parsed["text"],
        similarity_threshold=similarity_threshold,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        group_size=group_size,
        overlap_tokens=overlap_tokens,
    )

    # 4. Contextualize
    print("[ADD] Adding contextual embeddings...")
    contextualized = contextualize_chunks(
        raw_chunks,
        doc_summary,
        parsed["metadata"]["filename"],
        use_llm_context=use_llm_context,
    )

    # 5. Store base layer
    document_id = Path(file_path).stem
    print(f"[ADD] Storing {len(contextualized)} chunks for '{document_id}'...")
    chunk_ids = store_contextualized_chunks(
        chunks=contextualized,
        document_id=document_id,
        doc_summary=doc_summary,
        collection_name=collection_name,
        layer=0,
    )

    result: Dict = {
        "document_id": document_id,
        "filename": parsed["metadata"]["filename"],
        "chunk_count": len(chunk_ids),
        "images_found": parsed["images_found"],
        "tables_found": parsed["tables_found"],
        "tree_stats": None,
    }

    # 6. Build tree
    if build_tree:
        print("[ADD] Building T-Retriever tree...")
        tree_stats = build_tretriever_tree(
            document_id,
            collection_name=collection_name,
            max_depth=max_tree_layers,
        )
        result["tree_stats"] = tree_stats

    print(f"[ADD] Document '{document_id}' added successfully.")
    return result


# ============================================================================
# REMOVE DOCUMENT
# ============================================================================

def remove_document(
    document_id: str,
    collection_name: str = "nexus_chunks",
) -> Dict:
    """
    Completely remove a document: all layers, graph metadata, and caches.

    Args:
        document_id:     Document identifier (stem of original filename).
        collection_name: ChromaDB collection name.

    Returns:
        Dict with counts of deleted items.
    """
    collection = get_or_create_collection(collection_name)

    # 1. Count what we're about to delete
    all_results = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )
    if not all_results["ids"]:
        print(f"[REMOVE] No data found for document: {document_id}")
        return {"document_id": document_id, "deleted_chunks": 0, "deleted_graph": False}

    total = len(all_results["ids"])

    # 2. Delete graph metadata chunk (layer == -1 / type == graph_metadata)
    graph_chunk_id = f"{document_id}_graph_metadata"
    graph_deleted = False
    try:
        existing = collection.get(ids=[graph_chunk_id])
        if existing["ids"]:
            collection.delete(ids=[graph_chunk_id])
            graph_deleted = True
    except Exception:
        pass

    # 3. Delete all document chunks (all layers including base)
    delete_document_chunks(document_id, collection_name=collection_name)

    # 4. Clear in-memory graph cache
    if document_id in _document_graphs:
        del _document_graphs[document_id]

    print(f"[REMOVE] Document '{document_id}' fully removed ({total} items).")
    return {
        "document_id": document_id,
        "deleted_chunks": total,
        "deleted_graph": graph_deleted,
    }


# ============================================================================
# LIST DOCUMENTS
# ============================================================================

def list_documents(
    collection_name: str = "nexus_chunks",
) -> List[Dict]:
    """
    List all documents with chunk counts and tree status.

    Returns:
        List of dicts with document_id, chunk_count, has_tree, tree_depth.
    """
    stats = get_collection_stats(collection_name)
    documents: List[Dict] = []

    for doc_id in stats.get("documents", []):
        tree_stats = get_tree_stats(doc_id, collection_name)
        documents.append({
            "document_id": doc_id,
            "has_tree": tree_stats.get("tree_depth", 0) > 1,
            "tree_depth": tree_stats.get("tree_depth", 0),
            "total_nodes": tree_stats.get("total_nodes", 0),
            "unique_entities": tree_stats.get("unique_entities", 0),
            "layers": tree_stats.get("layers", {}),
        })

    return documents


# ============================================================================
# REBUILD TREE
# ============================================================================

def rebuild_document_tree(
    document_id: str,
    max_depth: int = 3,
    collection_name: str = "nexus_chunks",
) -> Dict:
    """
    Rebuild the T-Retriever tree for a document (keeps base chunks).

    Deletes summary layers ≥ 1, clears graph, and rebuilds from scratch.

    Args:
        document_id:     Document identifier.
        max_depth:       Maximum tree depth.
        collection_name: ChromaDB collection name.

    Returns:
        New tree statistics dict.
    """
    print(f"[REBUILD] Rebuilding tree for '{document_id}'...")
    delete_tree_layers(document_id, min_layer=1, collection_name=collection_name)

    if document_id in _document_graphs:
        del _document_graphs[document_id]

    tree_stats = build_tretriever_tree(
        document_id,
        collection_name=collection_name,
        max_depth=max_depth,
    )
    print(f"[REBUILD] Tree rebuilt: {tree_stats.get('tree_depth', 0)} layers.")
    return tree_stats
