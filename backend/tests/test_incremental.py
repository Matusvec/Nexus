"""
Tests for incremental tree updates, localized repair, compaction,
and query correctness before/after incremental updates.
"""
import json
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: build a mini tree in an isolated ChromaDB collection
# ---------------------------------------------------------------------------

def _make_collection(tmp_path, name="test_inc"):
    """Return a fresh ChromaDB collection backed by *tmp_path*."""
    import storage
    import chromadb
    from chromadb.config import Settings

    orig = storage.client
    storage.client = chromadb.PersistentClient(
        path=str(tmp_path / name),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    coll = storage.get_or_create_collection(name)
    return coll, name, orig


def _restore_client(orig):
    import storage
    storage.client = orig


def _seed_tree(coll, coll_name, doc_id="doc1"):
    """Insert L0 chunks, L1 summaries, and build TreeIndex manually.

    Layout:
      L0: chunk0 chunk1 chunk2 chunk3 chunk4 chunk5
      L1: summary0(chunk0,chunk1,chunk2)  summary1(chunk3,chunk4,chunk5)
      L2: root(summary0,summary1)

    Returns tree_idx, chunk_ids.
    """
    from tree_index import TreeIndex, _document_indices

    rng = np.random.RandomState(42)
    chunk_ids = [f"{doc_id}_L0_chunk{i}" for i in range(6)]
    chunk_texts = [
        "Machine learning models learn from data.",
        "Neural networks have multiple layers.",
        "Gradient descent optimizes parameters.",
        "Quantum computing uses qubits.",
        "Entanglement enables fast computation.",
        "Superposition allows parallel states.",
    ]
    chunk_entities = [
        json.dumps([{"name": "ML", "type": "concept", "importance": 5}]),
        json.dumps([{"name": "Neural", "type": "concept", "importance": 5}]),
        json.dumps([{"name": "Gradient", "type": "concept", "importance": 5}]),
        json.dumps([{"name": "Quantum", "type": "concept", "importance": 5}]),
        json.dumps([{"name": "Entanglement", "type": "concept", "importance": 5}]),
        json.dumps([{"name": "Superposition", "type": "concept", "importance": 5}]),
    ]

    coll.add(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=[rng.randn(768).tolist() for _ in range(6)],
        metadatas=[
            {"document_id": doc_id, "layer": 0, "entities": ent, "entity_names": ""}
            for ent in chunk_entities
        ],
    )

    # L1 summaries
    s0_id = f"{doc_id}_L1_summary0"
    s1_id = f"{doc_id}_L1_summary1"
    coll.add(
        ids=[s0_id, s1_id],
        documents=[
            "Summary of ML, neural nets, gradient descent.",
            "Summary of quantum, entanglement, superposition.",
        ],
        embeddings=[rng.randn(768).tolist() for _ in range(2)],
        metadatas=[
            {
                "document_id": doc_id,
                "layer": 1,
                "is_summary": True,
                "child_ids": ",".join(chunk_ids[:3]),
                "entities": "[]",
                "entity_names": "",
                "token_count": 20,
                "content_type": "summary",
            },
            {
                "document_id": doc_id,
                "layer": 1,
                "is_summary": True,
                "child_ids": ",".join(chunk_ids[3:]),
                "entities": "[]",
                "entity_names": "",
                "token_count": 20,
                "content_type": "summary",
            },
        ],
    )

    # L2 root
    root_id = f"{doc_id}_L2_summary0"
    coll.add(
        ids=[root_id],
        documents=["Root summary covering ML and quantum topics."],
        embeddings=[rng.randn(768).tolist()],
        metadatas=[{
            "document_id": doc_id,
            "layer": 2,
            "is_summary": True,
            "child_ids": f"{s0_id},{s1_id}",
            "entities": "[]",
            "entity_names": "",
            "token_count": 15,
            "content_type": "summary",
        }],
    )

    # Build tree index
    idx = TreeIndex()
    for cid in chunk_ids[:3]:
        idx.register_node(cid, layer=0, cluster_id="C0", parent_id=s0_id)
    for cid in chunk_ids[3:]:
        idx.register_node(cid, layer=0, cluster_id="C1", parent_id=s1_id)
    idx.register_node(
        s0_id, layer=1, cluster_id="C0",
        parent_id=root_id, children=list(chunk_ids[:3]),
    )
    idx.register_node(
        s1_id, layer=1, cluster_id="C1",
        parent_id=root_id, children=list(chunk_ids[3:]),
    )
    idx.register_node(
        root_id, layer=2, cluster_id="root",
        children=[s0_id, s1_id],
    )
    _document_indices[doc_id] = idx

    return idx, chunk_ids


# ============================================================================
# TreeIndex unit tests
# ============================================================================

class TestTreeIndex:
    def test_register_and_ancestor_chain(self):
        from tree_index import TreeIndex

        idx = TreeIndex()
        idx.register_node("c0", layer=0, parent_id="s0")
        idx.register_node("s0", layer=1, parent_id="root")
        idx.register_node("root", layer=2)

        chain = idx.ancestor_chain("c0")
        assert chain == ["s0", "root"]

    def test_tombstone(self):
        from tree_index import TreeIndex

        idx = TreeIndex()
        idx.register_node("c0", layer=0)
        idx.mark_tombstone("c0")
        assert idx.is_tombstoned("c0")
        assert "c0" in idx.dirty_nodes

    def test_remove_child_from_parent(self):
        from tree_index import TreeIndex

        idx = TreeIndex()
        idx.register_node("c0", layer=0, parent_id="s0")
        idx.register_node("s0", layer=1, children=["c0", "c1"])

        parent = idx.remove_child_from_parent("c0")
        assert parent == "s0"
        assert "c0" not in idx.membership["s0"]["children"]
        assert "s0" in idx.dirty_nodes

    def test_cluster_siblings(self):
        from tree_index import TreeIndex

        idx = TreeIndex()
        idx.register_node("c0", layer=0, parent_id="s0")
        idx.register_node("c1", layer=0, parent_id="s0")
        idx.register_node("c2", layer=0, parent_id="s0")
        idx.register_node("s0", layer=1, children=["c0", "c1", "c2"])

        siblings = idx.get_cluster_siblings("c0")
        assert set(siblings) == {"c1", "c2"}

    def test_compact_clears_dirty_and_bumps_version(self):
        from tree_index import TreeIndex

        idx = TreeIndex()
        idx.register_node("c0", layer=0)
        idx.mark_tombstone("c0")
        assert idx.tree_version == 1

        idx.compact()
        assert idx.tree_version == 2
        assert len(idx.dirty_nodes) == 0
        assert "c0" not in idx.membership

    def test_serialization_roundtrip(self):
        from tree_index import TreeIndex

        idx = TreeIndex()
        idx.register_node("c0", layer=0, parent_id="s0", children=[])
        idx.dirty_nodes.add("c0")
        idx.tree_version = 5

        data = idx.to_dict()
        restored = TreeIndex.from_dict(data)
        assert "c0" in restored.membership
        assert "c0" in restored.dirty_nodes
        assert restored.tree_version == 5

    def test_needs_compaction_threshold(self):
        from tree_index import TreeIndex, COMPACTION_DIRTY_THRESHOLD

        idx = TreeIndex()
        for i in range(COMPACTION_DIRTY_THRESHOLD - 1):
            idx.dirty_nodes.add(f"n{i}")
        assert not idx.needs_compaction

        idx.dirty_nodes.add("one_more")
        assert idx.needs_compaction


# ============================================================================
# Incremental deletion tests
# ============================================================================

class TestIncrementalDeletion:
    def test_delete_one_chunk_does_not_affect_other_cluster(self, tmp_path):
        """Deleting a chunk from cluster 0 must not touch cluster 1."""
        coll, coll_name, orig = _make_collection(tmp_path)
        try:
            idx, chunk_ids = _seed_tree(coll, coll_name)

            from document_manager import incremental_remove_chunks

            result = incremental_remove_chunks("doc1", [chunk_ids[0]], coll_name)
            assert result["deleted_count"] == 1

            # Cluster 1 (summary1) should be untouched
            s1_id = "doc1_L1_summary1"
            s1_entry = idx.membership.get(s1_id, {})
            assert set(s1_entry.get("children", [])) == set(chunk_ids[3:])
            assert s1_id not in idx.dirty_nodes

            # Cluster 0 (summary0) should have been repaired
            assert "doc1_L1_summary0" in result["repaired_summaries"]
        finally:
            _restore_client(orig)

    def test_delete_chunk_updates_parent_children(self, tmp_path):
        """After incremental delete, parent's children list shrinks."""
        coll, coll_name, orig = _make_collection(tmp_path)
        try:
            idx, chunk_ids = _seed_tree(coll, coll_name)

            from document_manager import incremental_remove_chunks

            incremental_remove_chunks("doc1", [chunk_ids[1]], coll_name)
            s0_entry = idx.membership["doc1_L1_summary0"]
            children = s0_entry["children"]
            assert chunk_ids[1] not in children
            assert chunk_ids[0] in children
            assert chunk_ids[2] in children
        finally:
            _restore_client(orig)

    def test_delete_propagates_up_ancestor_chain(self, tmp_path):
        """Root node should also be repaired when a child summary changes."""
        coll, coll_name, orig = _make_collection(tmp_path)
        try:
            idx, chunk_ids = _seed_tree(coll, coll_name)

            from document_manager import incremental_remove_chunks

            result = incremental_remove_chunks("doc1", [chunk_ids[0]], coll_name)
            # The root should have been re-summarized
            assert "doc1_L2_summary0" in result["repaired_summaries"]
        finally:
            _restore_client(orig)


# ============================================================================
# Cluster merge / neighbor reassignment tests
# ============================================================================

class TestClusterMerge:
    def test_shrunk_cluster_merges_with_nearest_sibling(self, tmp_path):
        """When a cluster drops below MIN_CLUSTER_SIZE_AFTER_DELETE,
        its remaining children should be reassigned to the nearest sibling."""
        import tree_index
        old_threshold = tree_index.MIN_CLUSTER_SIZE_AFTER_DELETE
        tree_index.MIN_CLUSTER_SIZE_AFTER_DELETE = 3  # Force merge when < 3

        coll, coll_name, orig = _make_collection(tmp_path, "test_merge")
        try:
            idx, chunk_ids = _seed_tree(coll, coll_name)

            from document_manager import incremental_remove_chunks

            # Delete 2 out of 3 from cluster 0 → only 1 remains → merge
            result = incremental_remove_chunks(
                "doc1", [chunk_ids[0], chunk_ids[1]], coll_name
            )

            # The remaining chunk2 should now be a child of summary1
            surviving_parent = idx.membership.get(chunk_ids[2], {}).get("parent_id")
            assert surviving_parent == "doc1_L1_summary1"
        finally:
            tree_index.MIN_CLUSTER_SIZE_AFTER_DELETE = old_threshold
            _restore_client(orig)


# ============================================================================
# Compaction tests
# ============================================================================

class TestCompaction:
    def test_dirty_tracking(self, tmp_path):
        """Dirty nodes accumulate and compact clears them."""
        coll, coll_name, orig = _make_collection(tmp_path, "test_compact")
        try:
            idx, chunk_ids = _seed_tree(coll, coll_name)

            from document_manager import incremental_remove_chunks

            incremental_remove_chunks("doc1", [chunk_ids[0]], coll_name)
            assert len(idx.dirty_nodes) > 0

            n = idx.compact()
            assert n > 0
            assert len(idx.dirty_nodes) == 0
        finally:
            _restore_client(orig)

    def test_auto_compaction_triggers(self, tmp_path):
        """When enough dirty nodes accumulate, compaction triggers automatically."""
        import tree_index
        old_threshold = tree_index.COMPACTION_DIRTY_THRESHOLD
        tree_index.COMPACTION_DIRTY_THRESHOLD = 2  # very low threshold

        coll, coll_name, orig = _make_collection(tmp_path, "test_autocomp")
        try:
            idx, chunk_ids = _seed_tree(coll, coll_name)

            from document_manager import incremental_remove_chunks

            result = incremental_remove_chunks(
                "doc1", [chunk_ids[0], chunk_ids[3]], coll_name
            )
            # Auto compaction should have fired
            assert result["compacted"] is True
            assert result["tree_version"] >= 2
        finally:
            tree_index.COMPACTION_DIRTY_THRESHOLD = old_threshold
            _restore_client(orig)


# ============================================================================
# Query correctness before and after incremental update
# ============================================================================

class TestQueryAfterIncremental:
    def test_query_returns_results_after_incremental_delete(self, tmp_path):
        """After deleting a chunk, queries should still return valid results
        from the remaining chunks."""
        coll, coll_name, orig = _make_collection(tmp_path, "test_query")
        try:
            _seed_tree(coll, coll_name)

            from document_manager import incremental_remove_chunks
            from t_query import collapsed_tree_retrieval

            # Delete one chunk
            incremental_remove_chunks("doc1", ["doc1_L0_chunk0"], coll_name)

            # Query should still work and not return the deleted chunk
            results = collapsed_tree_retrieval(
                query="machine learning",
                document_id="doc1",
                top_k=10,
                collection_name=coll_name,
            )
            returned_ids = {r["id"] for r in results}
            assert "doc1_L0_chunk0" not in returned_ids
            # Should still have results from other chunks
            assert len(results) > 0

        finally:
            _restore_client(orig)

    def test_deleted_chunk_not_in_chromadb(self, tmp_path):
        """Incrementally deleted chunks must be absent from ChromaDB."""
        coll, coll_name, orig = _make_collection(tmp_path, "test_absent")
        try:
            _seed_tree(coll, coll_name)

            from document_manager import incremental_remove_chunks

            incremental_remove_chunks("doc1", ["doc1_L0_chunk2"], coll_name)
            result = coll.get(ids=["doc1_L0_chunk2"])
            assert len(result["ids"]) == 0
        finally:
            _restore_client(orig)
