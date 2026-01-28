"""
Query Module for RAPTOR-enhanced RAG

Implements collapsed tree retrieval method from the RAPTOR paper.
Enhanced with:
- Query classification (from BookRAG paper) for adaptive retrieval
- LLM-guided traversal option (from LATTICE paper)

Queries across all layers of the tree simultaneously for comprehensive retrieval.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from storage import get_or_create_collection
from gemini_client import get_embedding, generate_content
from utils import count_tokens
import config


# ============================================================================
# QUERY CONFIGURATION
# ============================================================================

DEFAULT_TOP_K = 10              # Default number of results to return
MAX_CONTEXT_TOKENS = 8000       # Max tokens for context window
LAYER_BOOST_FACTOR = 0.05       # Boost factor per layer (higher = prefer summaries)

# Query classification settings (from BookRAG paper)
ENABLE_QUERY_CLASSIFICATION = True
SIMPLE_QUERY_TOP_K = 5          # Fewer results for simple queries
COMPLEX_QUERY_TOP_K = 15        # More results for complex queries


# ============================================================================
# QUERY CLASSIFICATION (from BookRAG - arXiv:2512.03413)
# ============================================================================

def classify_query(query: str) -> Dict[str, any]:
    """
    Classify query complexity to adapt retrieval strategy.
    
    Based on Information Foraging Theory from BookRAG paper.
    Classifies queries as:
    - simple: Direct lookup, single fact retrieval
    - complex: Multi-hop reasoning, comparative analysis
    - exploratory: Broad topic exploration
    
    Args:
        query: The user's question
        
    Returns:
        Dict with 'type', 'confidence', 'recommended_strategy'
    """
    # Indicators for complex queries
    complex_indicators = [
        "compare", "contrast", "difference", "between",
        "how does", "why does", "explain why",
        "relationship", "relate", "connection",
        "multiple", "several", "all the",
        "step by step", "process", "procedure",
        "advantages", "disadvantages", "pros and cons",
        "impact", "effect", "influence", "affect"
    ]
    
    # Indicators for simple queries
    simple_indicators = [
        "what is", "define", "who is", "when did",
        "where is", "which", "name the", "list",
        "how many", "how much"
    ]
    
    # Indicators for exploratory queries
    exploratory_indicators = [
        "tell me about", "overview", "summarize",
        "what do you know", "everything about",
        "introduction to", "basics of"
    ]
    
    query_lower = query.lower()
    
    # Count matches
    complex_score = sum(1 for ind in complex_indicators if ind in query_lower)
    simple_score = sum(1 for ind in simple_indicators if ind in query_lower)
    exploratory_score = sum(1 for ind in exploratory_indicators if ind in query_lower)
    
    # Additional complexity heuristics
    word_count = len(query.split())
    has_multiple_questions = query.count("?") > 1
    has_conjunction = any(w in query_lower for w in [" and ", " or ", " but "])
    
    if has_multiple_questions:
        complex_score += 2
    if has_conjunction:
        complex_score += 1
    if word_count > 15:
        complex_score += 1
    if word_count < 6:
        simple_score += 1
    
    # Determine type
    max_score = max(complex_score, simple_score, exploratory_score)
    
    if max_score == 0:
        # Default to simple for very short/unclear queries
        query_type = "simple"
        confidence = 0.5
    elif complex_score >= simple_score and complex_score >= exploratory_score:
        query_type = "complex"
        confidence = min(0.9, 0.5 + complex_score * 0.1)
    elif exploratory_score > simple_score:
        query_type = "exploratory"
        confidence = min(0.9, 0.5 + exploratory_score * 0.15)
    else:
        query_type = "simple"
        confidence = min(0.9, 0.5 + simple_score * 0.1)
    
    # Determine recommended strategy
    strategies = {
        "simple": {
            "retrieval": "collapsed_tree",
            "top_k": SIMPLE_QUERY_TOP_K,
            "prefer_layers": [0, 1],  # Prefer base chunks and first summary layer
            "description": "Direct retrieval from lower tree layers"
        },
        "complex": {
            "retrieval": "collapsed_tree",
            "top_k": COMPLEX_QUERY_TOP_K,
            "prefer_layers": None,  # All layers equally
            "description": "Broad retrieval across all tree layers"
        },
        "exploratory": {
            "retrieval": "collapsed_tree",
            "top_k": DEFAULT_TOP_K,
            "prefer_layers": [1, 2, 3],  # Prefer summary layers
            "description": "Focus on summary layers for overview"
        }
    }
    
    return {
        "type": query_type,
        "confidence": confidence,
        "scores": {
            "simple": simple_score,
            "complex": complex_score,
            "exploratory": exploratory_score
        },
        "strategy": strategies[query_type]
    }


# ============================================================================
# RETRIEVAL METHODS
# ============================================================================

def collapsed_tree_retrieval(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = "raptor_chunks",
    layer_weights: Optional[Dict[int, float]] = None
) -> List[Dict]:
    """
    Collapsed Tree Retrieval - query across ALL layers simultaneously
    
    This is the main RAPTOR retrieval method. It treats all nodes across
    all tree layers as a flat search space, allowing retrieval of both
    specific details (lower layers) and broader context (higher layers).
    
    Args:
        query: The query string
        document_id: Optional - filter to specific document
        top_k: Number of results to return
        collection_name: ChromaDB collection name
        layer_weights: Optional dict of {layer: weight} to boost certain layers
        
    Returns:
        List of result dicts with 'id', 'text', 'metadata', 'distance'
    """
    collection = get_or_create_collection(collection_name)
    
    # Generate query embedding
    query_embedding = get_embedding(query)
    
    # Build where clause
    where_clause = None
    if document_id:
        where_clause = {"document_id": {"$eq": document_id}}
    
    # Query across all layers (collapsed tree)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,  # Over-retrieve for post-filtering
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    
    # Process results
    processed = []
    for i, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        layer = meta.get("layer", 0)
        
        # Apply layer weighting if specified
        adjusted_distance = dist
        if layer_weights and layer in layer_weights:
            adjusted_distance = dist * layer_weights[layer]
        
        processed.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "adjusted_distance": adjusted_distance,
            "layer": layer,
            "is_summary": meta.get("is_summary", False)
        })
    
    # Sort by adjusted distance and return top_k
    processed.sort(key=lambda x: x["adjusted_distance"])
    return processed[:top_k]


def tree_traversal_retrieval(
    query: str,
    document_id: str,
    top_k_per_layer: int = 3,
    collection_name: str = "raptor_chunks"
) -> List[Dict]:
    """
    Tree Traversal Retrieval - top-down query through tree layers
    
    Alternative method that starts at root and traverses down, selecting
    most relevant nodes at each layer. Good for hierarchical exploration.
    
    Args:
        query: The query string
        document_id: Document to search (required for tree traversal)
        top_k_per_layer: Number of nodes to select at each layer
        collection_name: ChromaDB collection name
        
    Returns:
        List of result dicts
    """
    collection = get_or_create_collection(collection_name)
    
    # Generate query embedding
    query_embedding = get_embedding(query)
    
    # Find the maximum layer (root level)
    all_results = collection.get(
        where={"document_id": {"$eq": document_id}},
        include=["metadatas"]
    )
    
    if not all_results["ids"]:
        return []
    
    max_layer = max(m.get("layer", 0) for m in all_results["metadatas"])
    
    # Traverse from root to leaves
    all_retrieved = []
    current_parent_ids = None
    
    for layer in range(max_layer, -1, -1):
        # Build where clause
        where = {
            "$and": [
                {"document_id": {"$eq": document_id}},
                {"layer": {"$eq": layer}}
            ]
        }
        
        # Query this layer
        layer_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k_per_layer,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        if not layer_results["ids"][0]:
            continue
        
        # Add to results
        for doc_id, doc, meta, dist in zip(
            layer_results["ids"][0],
            layer_results["documents"][0],
            layer_results["metadatas"][0],
            layer_results["distances"][0]
        ):
            all_retrieved.append({
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "layer": meta.get("layer", 0),
                "is_summary": meta.get("is_summary", False)
            })
    
    # Sort by distance
    all_retrieved.sort(key=lambda x: x["distance"])
    return all_retrieved


def layer_specific_retrieval(
    query: str,
    layer: int,
    document_id: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = "raptor_chunks"
) -> List[Dict]:
    """
    Retrieve from a specific layer only
    
    Args:
        query: The query string
        layer: Layer to search (0 = base chunks, 1+ = summaries)
        document_id: Optional document filter
        top_k: Number of results
        collection_name: ChromaDB collection name
        
    Returns:
        List of result dicts
    """
    collection = get_or_create_collection(collection_name)
    
    query_embedding = get_embedding(query)
    
    # Build where clause
    if document_id:
        where_clause = {
            "$and": [
                {"document_id": {"$eq": document_id}},
                {"layer": {"$eq": layer}}
            ]
        }
    else:
        where_clause = {"layer": {"$eq": layer}}
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    
    processed = []
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        processed.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "layer": meta.get("layer", 0),
            "is_summary": meta.get("is_summary", False)
        })
    
    return processed


# ============================================================================
# ADAPTIVE RETRIEVAL (combines query classification with retrieval)
# ============================================================================

def adaptive_retrieval(
    query: str,
    document_id: Optional[str] = None,
    collection_name: str = "raptor_chunks",
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Intelligent retrieval that adapts strategy based on query classification.
    
    Based on Information Foraging Theory from BookRAG paper.
    
    Args:
        query: The user's question
        document_id: Optional document filter
        collection_name: ChromaDB collection name
        verbose: Print classification details
        
    Returns:
        Tuple of (results, classification_info)
    """
    # Classify the query
    classification = classify_query(query)
    strategy = classification["strategy"]
    
    if verbose:
        print(f"   Query type: {classification['type']} (confidence: {classification['confidence']:.2f})")
        print(f"   Strategy: {strategy['description']}")
        print(f"   Top-k: {strategy['top_k']}")
    
    # Build layer weights based on preferred layers
    layer_weights = None
    if strategy["prefer_layers"]:
        # Boost preferred layers (lower weight = higher priority)
        layer_weights = {}
        for layer in range(10):  # Support up to 10 layers
            if layer in strategy["prefer_layers"]:
                layer_weights[layer] = 0.8  # 20% boost
            else:
                layer_weights[layer] = 1.2  # 20% penalty
    
    # Execute retrieval with adapted parameters
    results = collapsed_tree_retrieval(
        query=query,
        document_id=document_id,
        top_k=strategy["top_k"],
        collection_name=collection_name,
        layer_weights=layer_weights
    )
    
    return results, classification


# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def build_context_from_results(
    results: List[Dict],
    max_tokens: int = MAX_CONTEXT_TOKENS,
    include_metadata: bool = True
) -> str:
    """
    Build context string from retrieval results
    
    Args:
        results: List of retrieval results
        max_tokens: Maximum tokens for context
        include_metadata: Whether to include metadata in context
        
    Returns:
        Formatted context string
    """
    context_parts = []
    total_tokens = 0
    
    for i, result in enumerate(results):
        text = result["text"]
        meta = result.get("metadata", {})
        layer = result.get("layer", 0)
        
        # Build chunk header
        if include_metadata:
            source_type = "Summary" if result.get("is_summary") else "Chunk"
            header = f"[{source_type} - Layer {layer}]"
        else:
            header = f"[{i+1}]"
        
        chunk_text = f"{header}\n{text}\n"
        chunk_tokens = count_tokens(chunk_text)
        
        if total_tokens + chunk_tokens > max_tokens:
            # Try to fit partial content
            remaining = max_tokens - total_tokens
            if remaining > 100:  # Only if meaningful space left
                truncated = text[:remaining * 4] + "..."  # Rough estimate
                context_parts.append(f"{header}\n{truncated}\n")
            break
        
        context_parts.append(chunk_text)
        total_tokens += chunk_tokens
    
    return "\n---\n".join(context_parts)


def deduplicate_results(results: List[Dict]) -> List[Dict]:
    """
    Remove duplicate or highly overlapping results
    
    Removes child chunks if their parent summary is also in results.
    """
    seen_ids = set()
    parent_covered_ids = set()
    deduplicated = []
    
    # First, collect all child IDs covered by summaries in results
    for result in results:
        if result.get("is_summary"):
            child_ids_str = result.get("metadata", {}).get("child_ids", "")
            if child_ids_str:
                parent_covered_ids.update(child_ids_str.split(","))
    
    # Now filter
    for result in results:
        result_id = result["id"]
        
        # Skip if already seen
        if result_id in seen_ids:
            continue
        
        # Skip if covered by a parent summary
        if result_id in parent_covered_ids and not result.get("is_summary"):
            continue
        
        seen_ids.add(result_id)
        deduplicated.append(result)
    
    return deduplicated


# ============================================================================
# QUESTION ANSWERING
# ============================================================================

def answer_question(
    question: str,
    document_id: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = "raptor_chunks",
    show_sources: bool = True,
    use_adaptive: bool = ENABLE_QUERY_CLASSIFICATION,
    verbose: bool = False
) -> Dict:
    """
    Answer a question using RAPTOR-enhanced retrieval
    
    Args:
        question: The question to answer
        document_id: Optional document filter
        top_k: Number of chunks to retrieve (may be overridden by adaptive)
        collection_name: ChromaDB collection name
        show_sources: Include source references in response
        use_adaptive: Use query classification for adaptive retrieval
        verbose: Print classification details
        
    Returns:
        Dict with 'answer', 'sources', 'context_chunks', 'query_classification'
    """
    classification = None
    
    # Retrieve relevant chunks
    if use_adaptive:
        results, classification = adaptive_retrieval(
            query=question,
            document_id=document_id,
            collection_name=collection_name,
            verbose=verbose
        )
        if verbose:
            print(f"   Retrieved {len(results)} chunks")
    else:
        results = collapsed_tree_retrieval(
            query=question,
            document_id=document_id,
            top_k=top_k,
            collection_name=collection_name
        )
    
    if not results:
        return {
            "answer": "I couldn't find any relevant information to answer this question.",
            "sources": [],
            "context_chunks": [],
            "query_classification": classification
        }
    
    # Deduplicate
    results = deduplicate_results(results)
    
    # Build context
    context = build_context_from_results(results)
    
    # Generate answer
    prompt = f"""You are a helpful assistant answering questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain
enough information to fully answer, say so and provide what you can.

CONTEXT:
{context}

QUESTION: {question}

Provide a comprehensive answer based on the context above. Be specific and cite
relevant details from the sources when appropriate."""

    answer = generate_content(prompt)
    
    # Build source references
    sources = []
    for r in results:
        source_type = "Summary (Layer {})".format(r["layer"]) if r.get("is_summary") else "Chunk"
        sources.append({
            "id": r["id"],
            "type": source_type,
            "layer": r["layer"],
            "preview": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        })
    
    return {
        "answer": answer,
        "sources": sources if show_sources else [],
        "context_chunks": results,
        "query_classification": classification
    }


def multi_hop_query(
    question: str,
    document_id: Optional[str] = None,
    max_hops: int = 3,
    collection_name: str = "raptor_chunks"
) -> Dict:
    """
    Multi-hop reasoning for complex questions
    
    Performs multiple retrieval passes, refining the query based on
    retrieved information.
    
    Args:
        question: The question to answer
        document_id: Optional document filter
        max_hops: Maximum retrieval iterations
        collection_name: ChromaDB collection name
        
    Returns:
        Dict with answer and reasoning trace
    """
    traces = []
    all_results = []
    current_query = question
    
    for hop in range(max_hops):
        # Retrieve
        results = collapsed_tree_retrieval(
            query=current_query,
            document_id=document_id,
            top_k=5,
            collection_name=collection_name
        )
        
        # Track unique results
        new_results = [r for r in results if r["id"] not in {ar["id"] for ar in all_results}]
        all_results.extend(new_results)
        
        if not new_results:
            break
        
        # Build context
        context = build_context_from_results(new_results)
        
        # Generate follow-up or answer
        prompt = f"""Based on this context, either:
1. If you can answer the original question, provide the answer
2. If you need more information, provide a follow-up query

Original question: {question}
Current context: {context}

If answering, start with "ANSWER:" 
If need more info, start with "QUERY:" and provide the follow-up query."""

        response = generate_content(prompt)
        
        traces.append({
            "hop": hop + 1,
            "query": current_query,
            "results_found": len(new_results),
            "response": response
        })
        
        if response.startswith("ANSWER:"):
            return {
                "answer": response[7:].strip(),
                "hops": hop + 1,
                "traces": traces,
                "total_results": len(all_results)
            }
        
        elif response.startswith("QUERY:"):
            current_query = response[6:].strip()
        else:
            # Model gave direct answer without prefix
            return {
                "answer": response,
                "hops": hop + 1,
                "traces": traces,
                "total_results": len(all_results)
            }
    
    # Max hops reached, generate final answer
    context = build_context_from_results(all_results[:10])
    final_answer = generate_content(f"""Answer this question based on the context:
Question: {question}
Context: {context}""")
    
    return {
        "answer": final_answer,
        "hops": max_hops,
        "traces": traces,
        "total_results": len(all_results)
    }


# ============================================================================
# CLI / TEST
# ============================================================================

def interactive_query(collection_name: str = "raptor_chunks"):
    """Interactive query mode for testing"""
    print("\n🔍 RAPTOR Interactive Query")
    print("=" * 50)
    print("Commands:")
    print("  quit/exit - Exit interactive mode")
    print("  doc:<id>  - Filter to specific document")
    print("  layer:<n> - Query specific layer only")
    print("  clear     - Clear document filter")
    print("=" * 50)
    
    document_filter = None
    layer_filter = None
    
    while True:
        try:
            query = input("\n❓ Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not query:
            continue
        
        if query.lower() in ["quit", "exit"]:
            break
        
        if query.startswith("doc:"):
            document_filter = query[4:].strip()
            print(f"📄 Filtering to document: {document_filter}")
            continue
        
        if query.startswith("layer:"):
            layer_filter = int(query[6:].strip())
            print(f"📊 Filtering to layer: {layer_filter}")
            continue
        
        if query.lower() == "clear":
            document_filter = None
            layer_filter = None
            print("🔄 Filters cleared")
            continue
        
        # Execute query
        print("\n⏳ Searching...")
        
        if layer_filter is not None:
            results = layer_specific_retrieval(
                query, layer_filter, document_filter, top_k=5, collection_name=collection_name
            )
        else:
            results = collapsed_tree_retrieval(
                query, document_filter, top_k=5, collection_name=collection_name
            )
        
        if not results:
            print("❌ No results found")
            continue
        
        print(f"\n✅ Found {len(results)} results:\n")
        for i, r in enumerate(results):
            layer = r.get("layer", 0)
            source = "📝 Summary" if r.get("is_summary") else "📄 Chunk"
            print(f"{i+1}. {source} (Layer {layer}, dist: {r['distance']:.3f})")
            print(f"   {r['text'][:200]}...")
            print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("RAPTOR Query System")
        print("\nUsage:")
        print("  python query.py interactive             Interactive query mode")
        print("  python query.py ask '<question>'        Answer a question")
        print("  python query.py ask '<q>' <doc_id>      Answer from specific document")
        print("  python query.py search '<query>'        Search and show results")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "interactive":
        interactive_query()
        
    elif command == "ask" and len(sys.argv) >= 3:
        question = sys.argv[2]
        doc_id = sys.argv[3] if len(sys.argv) >= 4 else None
        
        print(f"\n❓ Question: {question}")
        if doc_id:
            print(f"📄 Document: {doc_id}")
        print("\n⏳ Generating answer...\n")
        
        result = answer_question(question, document_id=doc_id)
        
        print("=" * 60)
        print("💡 ANSWER:")
        print("=" * 60)
        print(result["answer"])
        print("\n" + "=" * 60)
        print("📚 SOURCES:")
        for s in result["sources"][:5]:
            print(f"  - {s['id']} ({s['type']})")
        
    elif command == "search" and len(sys.argv) >= 3:
        query = sys.argv[2]
        doc_id = sys.argv[3] if len(sys.argv) >= 4 else None
        
        print(f"\n🔍 Searching: {query}")
        results = collapsed_tree_retrieval(query, doc_id, top_k=5)
        
        if not results:
            print("No results found")
        else:
            for i, r in enumerate(results):
                print(f"\n{i+1}. [{r['layer']}] {r['text'][:200]}...")
    
    else:
        print(f"Unknown command: {command}")
