"""
Document Manager - Clean API for adding and removing documents
in the T-Retriever hierarchical RAG system.

Provides a unified lifecycle for documents:
  add_document   : parse → chunk → contextualize → store → build tree
  remove_document: delete all layers + graph metadata + cache cleanup
  incremental_remove_chunks: remove specific chunks with localized tree repair
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
    resummarize_node,
    _document_graphs,
    MIN_CLUSTER_SIZE,
)
from tree_index import (
    get_document_index,
    load_document_index,
    save_document_index,
    _document_indices,
    MIN_CLUSTER_SIZE_AFTER_DELETE,
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

    # 5. Clear tree index cache
    if document_id in _document_indices:
        del _document_indices[document_id]

    print(f"[REMOVE] Document '{document_id}' fully removed ({total} items).")
    return {
        "document_id": document_id,
        "deleted_chunks": total,
        "deleted_graph": graph_deleted,
    }


# ============================================================================
# INCREMENTAL CHUNK REMOVAL (localized tree repair)
# ============================================================================

def incremental_remove_chunks(
    document_id: str,
    chunk_ids: List[str],
    collection_name: str = "nexus_chunks",
) -> Dict:
    """
    Remove specific chunks with localized tree repair.

    Instead of rebuilding the whole tree, this function:
    1. Tombstones each chunk in the membership index
    2. Removes the chunk from ChromaDB
    3. Removes it from its parent's child list
    4. If the parent cluster becomes too small, merges with nearest sibling
    5. Re-summarizes only affected parents
    6. Propagates changes up only the ancestor chain
    7. Triggers background compaction if dirty threshold is exceeded

    Returns:
        Dict with deleted_count, repaired_summaries, compacted info.
    """
    collection = get_or_create_collection(collection_name)
    tree_idx = load_document_index(document_id, collection_name)

    deleted_count = 0
    repaired_summaries: List[str] = []
    merged_clusters: List[str] = []

    for cid in chunk_ids:
        # 1. Tombstone in index
        tree_idx.mark_tombstone(cid)

        # 2. Delete from ChromaDB
        try:
            collection.delete(ids=[cid])
            deleted_count += 1
        except Exception:
            pass

        # 3. Remove from parent's child list; get parent_id
        parent_id = tree_idx.remove_child_from_parent(cid)
        if not parent_id:
            continue

        # 4. Check cluster size
        siblings = tree_idx.get_cluster_siblings(cid)
        remaining_children = [
            c for c in tree_idx.membership.get(parent_id, {}).get("children", [])
            if not tree_idx.is_tombstoned(c)
        ]

        if 0 < len(remaining_children) < MIN_CLUSTER_SIZE_AFTER_DELETE:
            # Cluster is too small — try to merge children into nearest sibling cluster
            # Collect embeddings for nearest-sibling lookup
            emb_lookup: Dict[str, list] = {}
            try:
                parent_entry = tree_idx.membership.get(parent_id, {})
                parent_layer = parent_entry.get("layer", 1)
                sibling_candidates = [
                    nid for nid, m in tree_idx.membership.items()
                    if m["layer"] == parent_layer
                    and nid != parent_id
                    and not tree_idx.is_tombstoned(nid)
                ]
                if sibling_candidates:
                    sib_results = collection.get(
                        ids=sibling_candidates + [parent_id],
                        include=["embeddings"],
                    )
                    for sid, emb in zip(sib_results["ids"], sib_results["embeddings"]):
                        if emb is not None:
                            emb_lookup[sid] = emb
            except Exception:
                pass

            nearest = tree_idx.find_nearest_sibling_cluster(parent_id, emb_lookup)
            if nearest and nearest in tree_idx.membership:
                # Move remaining children to the sibling cluster
                for child_id in remaining_children:
                    tree_idx.set_parent(child_id, nearest)
                    tree_idx.add_child(nearest, child_id)
                # Mark old parent for removal
                tree_idx.mark_tombstone(parent_id)
                try:
                    collection.delete(ids=[parent_id])
                except Exception:
                    pass
                merged_clusters.append(parent_id)
                # Re-summarize the sibling that absorbed the children
                resummarize_node(nearest, document_id, collection_name)
                repaired_summaries.append(nearest)
                # Propagate up the sibling's ancestor chain
                for ancestor_id in tree_idx.ancestor_chain(nearest):
                    if tree_idx.is_tombstoned(ancestor_id):
                        continue
                    resummarize_node(ancestor_id, document_id, collection_name)
                    repaired_summaries.append(ancestor_id)
                continue

        # 5. Re-summarize the parent from remaining children
        if remaining_children:
            resummarize_node(parent_id, document_id, collection_name)
            repaired_summaries.append(parent_id)

            # 6. Propagate up the ancestor chain
            for ancestor_id in tree_idx.ancestor_chain(parent_id):
                if tree_idx.is_tombstoned(ancestor_id):
                    continue
                resummarize_node(ancestor_id, document_id, collection_name)
                repaired_summaries.append(ancestor_id)
        else:
            # Parent has no children left — tombstone it too
            tree_idx.mark_tombstone(parent_id)
            try:
                collection.delete(ids=[parent_id])
            except Exception:
                pass

    # 7. Background compaction check
    compacted = False
    if tree_idx.needs_compaction:
        n_dirty = tree_idx.compact()
        compacted = True
        print(f"[COMPACT] Compacted {n_dirty} dirty nodes; tree_version={tree_idx.tree_version}")

    # Persist updated index
    save_document_index(document_id, collection_name)

    return {
        "document_id": document_id,
        "deleted_count": deleted_count,
        "repaired_summaries": repaired_summaries,
        "merged_clusters": merged_clusters,
        "compacted": compacted,
        "tree_version": tree_idx.tree_version,
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
