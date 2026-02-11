#!/usr/bin/env python
"""
T-Retriever Demo Script
=======================

Demonstrates the full T-Retriever pipeline:
  1. Ingest a document  (add_documents)
  2. Query the index    (query)
  3. Explain retrieval  (explain_retrieval)

Usage:
    python demo_tretrieval.py <path_to_pdf_or_docx>

Without a file argument the script uses a synthetic corpus so you can
run it immediately without any external documents or API keys
(set GEMINI_API_KEY if you want real LLM generation).
"""
from __future__ import annotations

import sys
import json
import textwrap
from pathlib import Path

# Ensure backend/ is importable
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ── helpers ────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _wrap(text: str, width: int = 72) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


# ── demo with a real file ─────────────────────────────────────────────────

def demo_with_file(file_path: str) -> None:
    """Full pipeline: ingest → query → explain using a real document."""
    from retrieval_api import add_documents, query, explain_retrieval, get_tree_info

    _print_section("STEP 1 — Ingest document")
    results = add_documents([file_path])
    doc_id = results[0]["document_id"]
    print(f"Document ID : {doc_id}")
    print(f"Chunks      : {results[0]['chunk_count']}")
    if results[0].get("tree_stats"):
        ts = results[0]["tree_stats"]
        print(f"Tree depth  : {ts.get('tree_depth', 0)} layers")
        print(f"Total nodes : {ts.get('total_nodes', 0)}")
        print(f"Entities    : {ts.get('total_entities', 0)}")

    _print_section("STEP 2 — Query")
    question = "What are the main topics discussed in this document?"
    print(f"Q: {question}\n")
    answer = query(question, document_id=doc_id, show_sources=True)
    print(_wrap(answer["answer"]))
    if answer.get("sources"):
        print(f"\n({len(answer['sources'])} sources used)")

    _print_section("STEP 3 — Explain retrieval")
    explanation = explain_retrieval(question, document_id=doc_id)
    print(f"Query type    : {explanation['query_classification']['type']}")
    print(f"Confidence    : {explanation['query_classification']['confidence']:.2f}")
    print(f"Strategy      : {explanation['query_classification']['strategy'].get('description', '')}")
    print(f"\nRetrieval path:")
    for step in explanation["retrieval_path"]:
        print(f"  → {step['step']}: {json.dumps({k: v for k, v in step.items() if k != 'step'})}")
    print(f"\nTop results ({len(explanation['results'])}):")
    for i, r in enumerate(explanation["results"][:5], 1):
        print(f"  {i}. [{r['source']}] Layer {r['layer']}  entities={r['entity_matches']}")
        print(f"     {r['text_preview'][:80]}...")

    _print_section("STEP 4 — Tree info")
    info = get_tree_info(doc_id)
    print(json.dumps(info, indent=2, default=str))


# ── demo with synthetic data (no file needed) ─────────────────────────────

def demo_synthetic() -> None:
    """
    Demonstrate the T-Retriever pipeline using in-memory synthetic data.

    This runs entirely without a real document or Gemini API key:
    it stores synthetic chunks directly in ChromaDB, builds a mini tree,
    and exercises the query + explain APIs.
    """
    import numpy as np

    _print_section("SYNTHETIC DEMO — no file required")
    print("Inserting synthetic chunks into an ephemeral collection...\n")

    from storage import get_or_create_collection
    from t_retriever import (
        extract_entities_simple,
        store_chunks_with_entities,
        build_tretriever_tree,
        get_tree_stats,
    )
    from retrieval_api import query, explain_retrieval

    collection_name = "demo_synthetic"
    doc_id = "synthetic_doc"
    collection = get_or_create_collection(collection_name)

    # Build synthetic chunks
    synthetic_texts = [
        "Machine learning uses statistical methods to allow computers to learn from data.",
        "Deep learning extends machine learning with neural networks that have many layers.",
        "Natural language processing enables computers to understand human language.",
        "Transformers are a type of neural network architecture used in NLP tasks.",
        "BERT is a pre-trained language model based on the transformer architecture.",
        "GPT models use transformers for text generation and understanding.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
        "Q-learning is a model-free reinforcement learning algorithm.",
        "Computer vision applies deep learning to image recognition tasks.",
        "Convolutional neural networks are commonly used for image classification.",
    ]

    # Store as base chunks with entities
    chunks = []
    for text in synthetic_texts:
        entities = extract_entities_simple(text, max_entities=5)
        chunks.append({"text": text, "entities": entities})

    print(f"Storing {len(chunks)} chunks...")
    store_chunks_with_entities(chunks, doc_id, layer=0, collection_name=collection_name)

    # Build tree
    print("Building T-Retriever tree...")
    stats = build_tretriever_tree(doc_id, collection_name=collection_name, max_depth=2)
    print(f"Tree depth: {stats.get('tree_depth', 0)}")
    print(f"Total nodes: {stats.get('total_nodes', 0)}")

    # Query
    _print_section("QUERY")
    question = "How do transformers relate to BERT?"
    print(f"Q: {question}\n")
    answer = query(question, document_id=doc_id, collection_name=collection_name)
    print(_wrap(answer.get("answer", "(no answer)")))

    # Explain
    _print_section("EXPLAIN RETRIEVAL")
    explanation = explain_retrieval(question, document_id=doc_id, collection_name=collection_name)
    print(f"Query type : {explanation['query_classification']['type']}")
    print(f"Path:")
    for step in explanation["retrieval_path"]:
        print(f"  → {step['step']}: {json.dumps({k: v for k, v in step.items() if k != 'step'})}")
    print(f"\nTop results:")
    for i, r in enumerate(explanation["results"][:3], 1):
        print(f"  {i}. [{r['source']}] L{r['layer']} {r['text_preview'][:60]}...")

    print("\n✓ Demo complete.")


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not Path(file_path).exists():
            print(f"[ERROR] File not found: {file_path}")
            sys.exit(1)
        demo_with_file(file_path)
    else:
        print("No file provided – running synthetic demo.\n")
        print("Usage: python demo_tretrieval.py <path_to_pdf_or_docx>")
        demo_synthetic()
