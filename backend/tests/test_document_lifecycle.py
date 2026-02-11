"""
Unit tests for document lifecycle: add, remove, list, update.

Uses an ephemeral ChromaDB collection (via conftest.py fixtures)
and mocked Gemini APIs so no network calls are made.
"""
import pytest
import numpy as np


class TestDocumentManagerRemove:
    """Tests for remove_document."""

    def test_remove_nonexistent_returns_zero(self, chromadb_collection):
        """Removing a doc that doesn't exist should report 0 deletions."""
        from document_manager import remove_document

        result = remove_document("nonexistent", collection_name=chromadb_collection.name)
        assert result["deleted_chunks"] == 0

    def test_remove_clears_all_chunks(self, tmp_path):
        """After removal, no chunks should remain for that doc."""
        import storage
        import chromadb
        from chromadb.config import Settings

        # Patch storage's client to use a temp dir
        orig_client = storage.client
        storage.client = chromadb.PersistentClient(
            path=str(tmp_path / "chroma_remove"),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        try:
            coll_name = "test_remove_chunks"
            coll = storage.get_or_create_collection(coll_name)

            rng = np.random.RandomState(0)
            coll.add(
                ids=["doc1_L0_chunk0", "doc1_L0_chunk1", "doc1_L1_summary0"],
                documents=["chunk0 text", "chunk1 text", "summary text"],
                embeddings=[rng.randn(768).tolist() for _ in range(3)],
                metadatas=[
                    {"document_id": "doc1", "layer": 0},
                    {"document_id": "doc1", "layer": 0},
                    {"document_id": "doc1", "layer": 1},
                ],
            )
            assert coll.count() == 3

            from document_manager import remove_document
            result = remove_document("doc1", collection_name=coll_name)
            assert result["deleted_chunks"] >= 3

            remaining = coll.get(where={"document_id": "doc1"})
            assert len(remaining["ids"]) == 0
        finally:
            storage.client = orig_client


class TestDocumentManagerList:
    """Tests for list_documents."""

    def test_empty_collection_returns_empty(self, chromadb_collection):
        from document_manager import list_documents

        docs = list_documents(collection_name=chromadb_collection.name)
        assert docs == []


class TestRetrievalAPIInterface:
    """Tests for the stable retrieval_api module."""

    def test_remove_documents_batch(self, chromadb_collection):
        """remove_documents should handle multiple IDs."""
        from retrieval_api import remove_documents

        results = remove_documents(["x", "y"], collection_name=chromadb_collection.name)
        assert len(results) == 2
        for r in results:
            assert r["deleted_chunks"] == 0

    def test_explain_retrieval_returns_structure(self, chromadb_collection):
        """explain_retrieval should return classification + path + results."""
        coll = chromadb_collection
        coll_name = coll.name

        # Insert a minimal chunk so the query has something to find
        rng = np.random.RandomState(0)
        coll.add(
            ids=["doc_L0_chunk0"],
            documents=["UMAP is used for dimensionality reduction."],
            embeddings=[rng.randn(768).tolist()],
            metadatas=[{"document_id": "doc", "layer": 0, "entity_names": "umap"}],
        )

        from retrieval_api import explain_retrieval

        explanation = explain_retrieval(
            "What is UMAP?",
            collection_name=coll_name,
        )
        assert "query_classification" in explanation
        assert "retrieval_path" in explanation
        assert "results" in explanation
        assert explanation["query_classification"]["type"] in (
            "simple",
            "complex",
            "exploratory",
        )

    def test_get_tree_info_nonexistent(self, chromadb_collection):
        """get_tree_info for a missing doc should indicate non-existence."""
        from retrieval_api import get_tree_info

        info = get_tree_info("nonexistent", collection_name=chromadb_collection.name)
        # Should either say exists=False or have 0 nodes
        assert info.get("exists") is False or info.get("total_nodes", 0) == 0
