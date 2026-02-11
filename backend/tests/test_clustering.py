"""
Unit tests for T-Retriever clustering and hierarchy construction.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

class TestBuildEntitySimilarityMatrix:
    def test_identical_entities_give_sim_one(self):
        from t_retriever import _build_entity_similarity_matrix

        entities = [
            [{"name": "Alpha", "type": "concept", "importance": 5}],
            [{"name": "Alpha", "type": "concept", "importance": 5}],
        ]
        mat = _build_entity_similarity_matrix(entities, 2)
        assert mat[0, 1] == pytest.approx(1.0)
        assert mat[1, 0] == pytest.approx(1.0)

    def test_disjoint_entities_give_sim_zero(self):
        from t_retriever import _build_entity_similarity_matrix

        entities = [
            [{"name": "Alpha", "type": "concept", "importance": 5}],
            [{"name": "Beta", "type": "concept", "importance": 5}],
        ]
        mat = _build_entity_similarity_matrix(entities, 2)
        assert mat[0, 1] == pytest.approx(0.0)

    def test_partial_overlap(self):
        from t_retriever import _build_entity_similarity_matrix

        entities = [
            [
                {"name": "Alpha", "type": "concept", "importance": 5},
                {"name": "Gamma", "type": "concept", "importance": 5},
            ],
            [
                {"name": "Alpha", "type": "concept", "importance": 5},
                {"name": "Beta", "type": "concept", "importance": 5},
            ],
        ]
        mat = _build_entity_similarity_matrix(entities, 2)
        # Jaccard: {alpha} intersect / {alpha, beta, gamma} union = 1/3
        assert mat[0, 1] == pytest.approx(1 / 3, abs=1e-6)


class TestUMAPReduction:
    def test_reduces_dimensionality(self, sample_embeddings):
        """If umap-learn is installed, dimensions should decrease."""
        from t_retriever import _reduce_with_umap

        reduced = _reduce_with_umap(sample_embeddings, n_components=5)
        # Either reduced or original (if umap not installed)
        assert reduced.shape[0] == sample_embeddings.shape[0]
        assert reduced.shape[1] <= sample_embeddings.shape[1]

    def test_too_few_samples_returns_original(self):
        from t_retriever import _reduce_with_umap

        tiny = np.random.randn(2, 50)
        reduced = _reduce_with_umap(tiny, n_components=5)
        assert reduced.shape == tiny.shape


class TestClusterWithEntities:
    def test_gmm_finds_two_clusters(self, sample_embeddings, sample_entities_two_clusters):
        from t_retriever import cluster_with_entities

        clusters, membership = cluster_with_entities(
            sample_embeddings,
            sample_entities_two_clusters,
            layer=0,
            strategy="gmm",
        )
        assert len(clusters) >= 2
        assert membership.shape[0] == 10

    def test_hdbscan_finds_clusters(self, sample_embeddings, sample_entities_two_clusters):
        from t_retriever import cluster_with_entities

        clusters, membership = cluster_with_entities(
            sample_embeddings,
            sample_entities_two_clusters,
            layer=0,
            strategy="hdbscan",
        )
        # hdbscan may fall back to gmm if not installed
        assert len(clusters) >= 1
        assert membership.shape[0] == 10

    def test_small_input_returns_single_cluster(self):
        from t_retriever import cluster_with_entities

        emb = np.random.randn(2, 50)
        ents = [[{"name": "X", "type": "concept", "importance": 5}]] * 2
        clusters, _ = cluster_with_entities(emb, ents, layer=0)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1]


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

class TestEntityExtraction:
    def test_simple_extracts_capitalized(self):
        from t_retriever import extract_entities_simple

        entities = extract_entities_simple("Machine Learning is used by Google and OpenAI.")
        names = {e["name"] for e in entities}
        # Should find capitalized phrases
        assert len(names) > 0

    def test_extract_entities_respects_max(self):
        from t_retriever import extract_entities_simple

        text = " ".join([f"Entity{i}" for i in range(100)])
        entities = extract_entities_simple(text, max_entities=5)
        assert len(entities) <= 5


class TestDeduplicateEntities:
    def test_removes_exact_duplicates(self):
        from t_retriever import deduplicate_entities

        entities = [
            {"name": "Alpha", "type": "concept", "importance": 5},
            {"name": "alpha", "type": "concept", "importance": 3},
        ]
        deduped = deduplicate_entities(entities)
        assert len(deduped) == 1
        # Should keep the one with higher importance
        assert deduped[0]["importance"] == 5


# ---------------------------------------------------------------------------
# EntityGraph
# ---------------------------------------------------------------------------

class TestEntityGraph:
    def test_add_node_and_get_neighbors(self):
        from t_retriever import EntityGraph

        graph = EntityGraph()
        rng = np.random.RandomState(0)

        for i in range(3):
            graph.add_node(
                f"chunk_{i}",
                entities=[{"name": f"E{i}", "type": "concept", "importance": 5}],
                embedding=rng.randn(10).tolist(),
                layer=0,
            )

        # Before build_edges, no neighbors
        assert graph.get_neighbors("chunk_0") == []

        graph.build_edges(similarity_threshold=0.0)
        # After build_edges with threshold 0, every node connects to others
        neighbors = graph.get_neighbors("chunk_0", top_k=5)
        assert len(neighbors) > 0

    def test_entity_index_lookup(self):
        from t_retriever import EntityGraph

        graph = EntityGraph()
        graph.add_node("c1", [{"name": "Alpha", "type": "concept", "importance": 5}], [0.1] * 10)
        graph.add_node("c2", [{"name": "Alpha", "type": "concept", "importance": 5}], [0.2] * 10)
        graph.add_node("c3", [{"name": "Beta", "type": "concept", "importance": 5}], [0.3] * 10)

        alpha_chunks = graph.get_chunks_by_entity("Alpha")
        assert set(alpha_chunks) == {"c1", "c2"}

    def test_serialization_roundtrip(self):
        from t_retriever import EntityGraph

        graph = EntityGraph()
        graph.add_node("c1", [{"name": "X", "type": "concept", "importance": 5}], [0.1] * 10)
        graph.build_edges(similarity_threshold=0.0)

        data = graph.to_dict()
        restored = EntityGraph.from_dict(data)
        assert "c1" in restored.nodes
        assert "x" in restored.entity_index

    def test_expand_from_nodes(self):
        from t_retriever import EntityGraph

        graph = EntityGraph()
        rng = np.random.RandomState(0)
        for i in range(5):
            graph.add_node(f"c{i}", [{"name": "E", "type": "concept", "importance": 5}], rng.randn(10).tolist())
        graph.build_edges(similarity_threshold=0.0)

        expanded = graph.expand_from_nodes(["c0"], hops=1, top_k_per_hop=2)
        assert len(expanded) > 0
        assert "c0" not in expanded  # seeds are excluded
