"""
Unit tests for T-Retriever query and retrieval correctness.
"""
import pytest


class TestQueryClassification:
    def test_simple_query(self):
        from t_query import classify_query

        result = classify_query("What is machine learning?")
        assert result["type"] in ("simple", "exploratory")
        assert "strategy" in result
        assert "query_entities" in result

    def test_complex_query(self):
        from t_query import classify_query

        result = classify_query(
            "Compare the relationship between transformer architectures "
            "and how does attention mechanism connect to BERT?"
        )
        assert result["type"] == "complex"
        assert result["strategy"]["use_graph"] is True

    def test_exploratory_query(self):
        from t_query import classify_query

        result = classify_query("Tell me about the basics of quantum computing")
        assert result["type"] == "exploratory"

    def test_multihop_enables_graph(self):
        from t_query import classify_query

        result = classify_query("What is the relationship between A and B?")
        strategy = result["strategy"]
        assert strategy.get("use_graph") is True


class TestExtractQueryEntities:
    def test_extracts_content_words(self):
        from t_query import extract_query_entities

        entities = extract_query_entities("How does UMAP relate to clustering?")
        lower = [e.lower() for e in entities]
        assert "umap" in lower or "UMAP" in [e for e in entities]

    def test_filters_stopwords(self):
        from t_query import extract_query_entities

        entities = extract_query_entities("What is the meaning of life?")
        lower = [e.lower() for e in entities]
        assert "what" not in lower
        assert "the" not in lower


class TestBuildContextFromResults:
    def test_truncates_at_max_tokens(self):
        from t_query import build_context_from_results

        results = [
            {
                "text": "word " * 5000,
                "metadata": {},
                "layer": 0,
                "source": "tree",
                "is_summary": False,
            }
        ]
        context = build_context_from_results(results, max_tokens=100)
        # Should be significantly shorter than full text
        assert len(context) < len(results[0]["text"])

    def test_includes_metadata_header(self):
        from t_query import build_context_from_results

        results = [
            {
                "text": "Some content here",
                "metadata": {},
                "layer": 1,
                "source": "graph",
                "is_summary": True,
            }
        ]
        context = build_context_from_results(results, include_metadata=True)
        assert "Summary" in context
        assert "Layer 1" in context


class TestDeduplicateResults:
    def test_removes_seen_ids(self):
        from t_query import deduplicate_results

        results = [
            {"id": "a", "text": "x", "is_summary": False, "metadata": {}, "source": "tree"},
            {"id": "a", "text": "x", "is_summary": False, "metadata": {}, "source": "tree"},
            {"id": "b", "text": "y", "is_summary": False, "metadata": {}, "source": "tree"},
        ]
        deduped = deduplicate_results(results)
        ids = [r["id"] for r in deduped]
        assert ids.count("a") == 1
        assert "b" in ids

    def test_keeps_graph_results_even_if_covered(self):
        from t_query import deduplicate_results

        results = [
            {
                "id": "summary1",
                "text": "summary",
                "is_summary": True,
                "metadata": {"child_ids": "child1,child2"},
                "source": "tree",
            },
            {
                "id": "child1",
                "text": "child content",
                "is_summary": False,
                "metadata": {},
                "source": "graph",
            },
        ]
        deduped = deduplicate_results(results)
        ids = [r["id"] for r in deduped]
        # graph-sourced child should be kept even though it's covered by a summary
        assert "child1" in ids
