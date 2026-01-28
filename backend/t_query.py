"""
T-Retriever Query Module: Hybrid Tree + Graph Retrieval

Implements the T-Retriever query strategy:
1. Tree Retrieval: Collapsed tree search across all layers (like RAPTOR)
2. Graph Expansion: Expand retrieved nodes through entity graph
3. Hybrid Fusion: Combine tree and graph results

Based on "T-Retriever: Tree-based Hierarchical Retrieval Augmented 
Generation for Textual Graphs" (2026)
"""
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from storage import get_or_create_collection
from gemini_client import get_embedding, generate_content
from utils import count_tokens
import config

# Import T-Retriever components
from t_retriever import (
    EntityGraph, 
    get_document_graph, 
    load_document_graph,
    GRAPH_EXPANSION_HOPS,
    GRAPH_EXPANSION_TOP_K,
    TREE_RETRIEVAL_TOP_K,
    HYBRID_ALPHA
)


# ============================================================================
# QUERY CONFIGURATION
# ============================================================================

DEFAULT_TOP_K = 10              # Default number of results to return
MAX_CONTEXT_TOKENS = 8000       # Max tokens for context window

# Query classification settings
ENABLE_QUERY_CLASSIFICATION = True
SIMPLE_QUERY_TOP_K = 5
COMPLEX_QUERY_TOP_K = 15

# T-Retriever specific
ENABLE_GRAPH_EXPANSION = True   # Toggle graph expansion
ENTITY_QUERY_BOOST = 0.2        # Boost for chunks matching query entities


# ============================================================================
# QUERY ENTITY EXTRACTION
# ============================================================================

def extract_query_entities(query: str) -> List[str]:
    """
    Extract key entities/concepts from query for graph-based retrieval
    
    Uses simple heuristics for speed; can be replaced with LLM extraction
    for better accuracy.
    """
    import re
    
    # Remove common question words
    stopwords = {
        'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 
        'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
        'at', 'to', 'for', 'of', 'with', 'by', 'from', 'can', 'could', 
        'would', 'should', 'do', 'does', 'did', 'have', 'has', 'had',
        'this', 'that', 'these', 'those', 'about', 'tell', 'me', 'explain'
    }
    
    # Tokenize and filter
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]*\b', query)
    entities = [w for w in words if w.lower() not in stopwords and len(w) > 2]
    
    # Also find capitalized phrases
    cap_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
    entities.extend(cap_phrases)
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)
    
    return unique


# ============================================================================
# QUERY CLASSIFICATION (Enhanced for T-Retriever)
# ============================================================================

def classify_query(query: str) -> Dict:
    """
    Classify query to determine optimal retrieval strategy
    
    T-Retriever enhancement: Also determines if graph expansion is beneficial
    """
    # Indicators for different query types
    complex_indicators = [
        "compare", "contrast", "difference", "between", "relationship",
        "how does", "why does", "explain why", "connect", "relate",
        "multiple", "several", "all the", "impact", "effect"
    ]
    
    simple_indicators = [
        "what is", "define", "who is", "when did", "where is",
        "which", "name the", "list", "how many", "how much"
    ]
    
    exploratory_indicators = [
        "tell me about", "overview", "summarize", "everything about",
        "introduction to", "basics of"
    ]
    
    # Multi-hop indicators (benefit from graph expansion)
    multihop_indicators = [
        "relationship between", "how does .* relate", "connection",
        "linked to", "associated with", "causes", "leads to",
        "depends on", "influences", "affects"
    ]
    
    query_lower = query.lower()
    
    # Score each type
    complex_score = sum(1 for ind in complex_indicators if ind in query_lower)
    simple_score = sum(1 for ind in simple_indicators if ind in query_lower)
    exploratory_score = sum(1 for ind in exploratory_indicators if ind in query_lower)
    multihop_score = sum(1 for ind in multihop_indicators if ind in query_lower)
    
    # Additional heuristics
    word_count = len(query.split())
    has_multiple_questions = query.count("?") > 1
    has_conjunction = any(w in query_lower for w in [" and ", " or ", " but "])
    
    if has_multiple_questions:
        complex_score += 2
        multihop_score += 1
    if has_conjunction:
        complex_score += 1
    if word_count > 15:
        complex_score += 1
    if word_count < 6:
        simple_score += 1
    
    # Determine type
    max_score = max(complex_score, simple_score, exploratory_score)
    
    if max_score == 0:
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
    
    # Determine if graph expansion is beneficial
    use_graph = multihop_score > 0 or query_type == "complex"
    
    # Build strategy
    strategies = {
        "simple": {
            "retrieval": "tree_only",
            "top_k": SIMPLE_QUERY_TOP_K,
            "prefer_layers": [0, 1],
            "use_graph": False,
            "description": "Direct retrieval from lower tree layers"
        },
        "complex": {
            "retrieval": "hybrid",
            "top_k": COMPLEX_QUERY_TOP_K,
            "prefer_layers": None,
            "use_graph": True,
            "graph_hops": 2,
            "description": "Hybrid tree + graph retrieval for complex reasoning"
        },
        "exploratory": {
            "retrieval": "tree_only",
            "top_k": DEFAULT_TOP_K,
            "prefer_layers": [1, 2, 3],
            "use_graph": False,
            "description": "Focus on summary layers for overview"
        }
    }
    
    strategy = strategies[query_type]
    
    # Override graph usage based on multi-hop score
    if multihop_score > 0 and query_type != "complex":
        strategy["use_graph"] = True
        strategy["retrieval"] = "hybrid"
    
    return {
        "type": query_type,
        "confidence": confidence,
        "scores": {
            "simple": simple_score,
            "complex": complex_score,
            "exploratory": exploratory_score,
            "multihop": multihop_score
        },
        "strategy": strategy,
        "query_entities": extract_query_entities(query)
    }


# ============================================================================
# TREE RETRIEVAL (Collapsed Tree - like RAPTOR)
# ============================================================================

def collapsed_tree_retrieval(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = "raptor_chunks",
    layer_weights: Optional[Dict[int, float]] = None,
    query_entities: Optional[List[str]] = None
) -> List[Dict]:
    """
    Collapsed Tree Retrieval - query across ALL layers simultaneously
    
    Enhanced for T-Retriever with entity-based boosting.
    """
    collection = get_or_create_collection(collection_name)
    
    # Generate query embedding
    query_embedding = get_embedding(query)
    
    # Build where clause
    where_clause = None
    if document_id:
        where_clause = {"document_id": {"$eq": document_id}}
    
    # Exclude graph metadata chunks
    if where_clause:
        where_clause = {
            "$and": [
                where_clause,
                {"layer": {"$gte": 0}}
            ]
        }
    else:
        where_clause = {"layer": {"$gte": 0}}
    
    # Query across all layers
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,  # Over-retrieve for post-processing
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
        
        # Apply layer weighting
        adjusted_distance = dist
        if layer_weights and layer in layer_weights:
            adjusted_distance = dist * layer_weights[layer]
        
        # Entity boost: if chunk contains query entities, reduce distance
        if query_entities:
            chunk_entities = meta.get("entity_names", "").lower()
            matching = sum(1 for e in query_entities if e.lower() in chunk_entities)
            if matching > 0:
                adjusted_distance *= (1 - ENTITY_QUERY_BOOST * min(matching, 3))
        
        processed.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "adjusted_distance": adjusted_distance,
            "layer": layer,
            "is_summary": meta.get("is_summary", False),
            "source": "tree"
        })
    
    # Sort by adjusted distance
    processed.sort(key=lambda x: x["adjusted_distance"])
    return processed[:top_k]


# ============================================================================
# GRAPH EXPANSION RETRIEVAL
# ============================================================================

def graph_expansion_retrieval(
    seed_chunk_ids: List[str],
    document_id: str,
    hops: int = GRAPH_EXPANSION_HOPS,
    top_k_per_hop: int = GRAPH_EXPANSION_TOP_K,
    collection_name: str = "raptor_chunks"
) -> List[Dict]:
    """
    Expand from seed chunks through entity graph
    
    This is the key T-Retriever innovation: using entity relationships
    to find related chunks that pure embedding similarity might miss.
    """
    # Load document graph
    graph = load_document_graph(document_id, collection_name)
    
    if not graph.nodes:
        return []
    
    # Get expanded chunk IDs
    expanded_ids = graph.expand_from_nodes(
        seed_chunk_ids, 
        hops=hops, 
        top_k_per_hop=top_k_per_hop
    )
    
    if not expanded_ids:
        return []
    
    # Fetch chunk data from ChromaDB
    collection = get_or_create_collection(collection_name)
    results = collection.get(
        ids=expanded_ids,
        include=["documents", "metadatas"]
    )
    
    processed = []
    for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        processed.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": 0.5,  # Default distance for graph-expanded results
            "adjusted_distance": 0.5,
            "layer": meta.get("layer", 0),
            "is_summary": meta.get("is_summary", False),
            "source": "graph"
        })
    
    return processed


def entity_based_retrieval(
    query_entities: List[str],
    document_id: str,
    top_k: int = 5,
    collection_name: str = "raptor_chunks"
) -> List[Dict]:
    """
    Retrieve chunks that contain specific entities
    
    Useful for queries that mention specific concepts/terms.
    """
    graph = load_document_graph(document_id, collection_name)
    
    if not graph.entity_index:
        return []
    
    # Find chunks containing query entities
    matching_chunks = set()
    for entity in query_entities:
        chunk_ids = graph.get_chunks_by_entity(entity)
        matching_chunks.update(chunk_ids)
    
    if not matching_chunks:
        return []
    
    # Limit results
    chunk_ids = list(matching_chunks)[:top_k * 2]
    
    # Fetch from ChromaDB
    collection = get_or_create_collection(collection_name)
    results = collection.get(
        ids=chunk_ids,
        include=["documents", "metadatas"]
    )
    
    processed = []
    for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
        # Count matching entities for scoring
        chunk_entities = meta.get("entity_names", "").lower()
        match_count = sum(1 for e in query_entities if e.lower() in chunk_entities)
        
        processed.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "distance": 1.0 - (match_count * 0.1),  # Lower distance for more matches
            "adjusted_distance": 1.0 - (match_count * 0.1),
            "layer": meta.get("layer", 0),
            "is_summary": meta.get("is_summary", False),
            "source": "entity",
            "entity_matches": match_count
        })
    
    # Sort by match count
    processed.sort(key=lambda x: x.get("entity_matches", 0), reverse=True)
    return processed[:top_k]


# ============================================================================
# HYBRID RETRIEVAL (T-Retriever Core)
# ============================================================================

def hybrid_retrieval(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = "raptor_chunks",
    alpha: float = HYBRID_ALPHA,
    graph_hops: int = GRAPH_EXPANSION_HOPS,
    query_entities: Optional[List[str]] = None
) -> List[Dict]:
    """
    T-Retriever Hybrid Retrieval: Tree + Graph
    
    Combines:
    1. Collapsed tree retrieval (embedding similarity across layers)
    2. Graph expansion (entity-based relationship traversal)
    3. Entity-based retrieval (direct entity matching)
    
    Args:
        query: Query string
        document_id: Optional document filter
        top_k: Total results to return
        collection_name: ChromaDB collection
        alpha: Weight for tree vs graph (0.6 = 60% tree, 40% graph)
        graph_hops: Number of hops for graph expansion
        query_entities: Pre-extracted query entities
        
    Returns:
        Fused result list
    """
    # Extract query entities if not provided
    if query_entities is None:
        query_entities = extract_query_entities(query)
    
    # 1. Tree retrieval
    tree_results = collapsed_tree_retrieval(
        query=query,
        document_id=document_id,
        top_k=top_k,
        collection_name=collection_name,
        query_entities=query_entities
    )
    
    # If no document_id, we can't do graph expansion (need specific document)
    if not document_id:
        return tree_results
    
    # 2. Graph expansion from tree results
    seed_ids = [r["id"] for r in tree_results[:5]]  # Use top 5 as seeds
    graph_results = graph_expansion_retrieval(
        seed_chunk_ids=seed_ids,
        document_id=document_id,
        hops=graph_hops,
        collection_name=collection_name
    )
    
    # 3. Entity-based retrieval
    entity_results = entity_based_retrieval(
        query_entities=query_entities,
        document_id=document_id,
        top_k=5,
        collection_name=collection_name
    )
    
    # 4. Fuse results
    all_results = {}
    
    # Add tree results with alpha weight
    for r in tree_results:
        rid = r["id"]
        all_results[rid] = r.copy()
        all_results[rid]["fusion_score"] = alpha * (1 / (1 + r["adjusted_distance"]))
    
    # Add graph results with (1-alpha) weight
    graph_weight = (1 - alpha) * 0.7
    for r in graph_results:
        rid = r["id"]
        if rid in all_results:
            all_results[rid]["fusion_score"] += graph_weight * 0.5
            all_results[rid]["source"] = "tree+graph"
        else:
            all_results[rid] = r.copy()
            all_results[rid]["fusion_score"] = graph_weight * 0.5
    
    # Add entity results with boost
    entity_weight = (1 - alpha) * 0.3
    for r in entity_results:
        rid = r["id"]
        if rid in all_results:
            all_results[rid]["fusion_score"] += entity_weight * r.get("entity_matches", 1) * 0.3
        else:
            all_results[rid] = r.copy()
            all_results[rid]["fusion_score"] = entity_weight * r.get("entity_matches", 1) * 0.3
    
    # Sort by fusion score (higher is better)
    fused = list(all_results.values())
    fused.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)
    
    return fused[:top_k]


# ============================================================================
# ADAPTIVE RETRIEVAL (Auto-selects strategy)
# ============================================================================

def adaptive_retrieval(
    query: str,
    document_id: Optional[str] = None,
    collection_name: str = "raptor_chunks",
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Intelligent retrieval that adapts strategy based on query classification.
    
    This is the main entry point for T-Retriever queries.
    """
    # Classify query
    classification = classify_query(query)
    strategy = classification["strategy"]
    query_entities = classification["query_entities"]
    
    if verbose:
        print(f"   Query type: {classification['type']} (confidence: {classification['confidence']:.2f})")
        print(f"   Strategy: {strategy['description']}")
        print(f"   Query entities: {query_entities}")
        print(f"   Use graph: {strategy.get('use_graph', False)}")
    
    # Build layer weights
    layer_weights = None
    if strategy.get("prefer_layers"):
        layer_weights = {}
        for layer in range(10):
            if layer in strategy["prefer_layers"]:
                layer_weights[layer] = 0.8
            else:
                layer_weights[layer] = 1.2
    
    # Execute appropriate retrieval
    if strategy.get("use_graph") and ENABLE_GRAPH_EXPANSION and document_id:
        results = hybrid_retrieval(
            query=query,
            document_id=document_id,
            top_k=strategy["top_k"],
            collection_name=collection_name,
            graph_hops=strategy.get("graph_hops", GRAPH_EXPANSION_HOPS),
            query_entities=query_entities
        )
    else:
        results = collapsed_tree_retrieval(
            query=query,
            document_id=document_id,
            top_k=strategy["top_k"],
            collection_name=collection_name,
            layer_weights=layer_weights,
            query_entities=query_entities
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
    """Build context string from retrieval results"""
    context_parts = []
    total_tokens = 0
    
    for i, result in enumerate(results):
        text = result["text"]
        meta = result.get("metadata", {})
        layer = result.get("layer", 0)
        source = result.get("source", "tree")
        
        if include_metadata:
            source_type = "Summary" if result.get("is_summary") else "Chunk"
            header = f"[{source_type} - Layer {layer} - Source: {source}]"
        else:
            header = f"[{i+1}]"
        
        chunk_text = f"{header}\n{text}\n"
        chunk_tokens = count_tokens(chunk_text)
        
        if total_tokens + chunk_tokens > max_tokens:
            remaining = max_tokens - total_tokens
            if remaining > 100:
                truncated = text[:remaining * 4] + "..."
                context_parts.append(f"{header}\n{truncated}\n")
            break
        
        context_parts.append(chunk_text)
        total_tokens += chunk_tokens
    
    return "\n---\n".join(context_parts)


def deduplicate_results(results: List[Dict]) -> List[Dict]:
    """Remove duplicate or highly overlapping results"""
    seen_ids = set()
    parent_covered_ids = set()
    deduplicated = []
    
    # Collect child IDs covered by summaries
    for result in results:
        if result.get("is_summary"):
            child_ids_str = result.get("metadata", {}).get("child_ids", "")
            if child_ids_str:
                parent_covered_ids.update(child_ids_str.split(","))
    
    for result in results:
        result_id = result["id"]
        
        if result_id in seen_ids:
            continue
        
        # Skip if covered by a parent summary (but keep if from graph expansion)
        if result_id in parent_covered_ids and not result.get("is_summary"):
            if result.get("source") != "graph":
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
    Answer a question using T-Retriever hybrid retrieval
    
    This is the main Q&A function.
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
        retrieval_method = r.get("source", "tree")
        sources.append({
            "id": r["id"],
            "type": source_type,
            "layer": r["layer"],
            "method": retrieval_method,
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
    Multi-hop reasoning using T-Retriever
    
    Enhanced with graph expansion for better multi-hop support.
    """
    traces = []
    all_results = []
    current_query = question
    
    for hop in range(max_hops):
        # Use hybrid retrieval for multi-hop
        if document_id:
            results = hybrid_retrieval(
                query=current_query,
                document_id=document_id,
                top_k=5,
                collection_name=collection_name
            )
        else:
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
        
        context = build_context_from_results(new_results)
        
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
            return {
                "answer": response,
                "hops": hop + 1,
                "traces": traces,
                "total_results": len(all_results)
            }
    
    # Max hops reached
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
# LAYER-SPECIFIC RETRIEVAL (Utility)
# ============================================================================

def layer_specific_retrieval(
    query: str,
    layer: int,
    document_id: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = "raptor_chunks"
) -> List[Dict]:
    """Retrieve from a specific layer only"""
    collection = get_or_create_collection(collection_name)
    
    query_embedding = get_embedding(query)
    
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
            "is_summary": meta.get("is_summary", False),
            "source": "tree"
        })
    
    return processed


# ============================================================================
# CLI / TEST
# ============================================================================

def interactive_query(collection_name: str = "raptor_chunks"):
    """Interactive query mode for testing"""
    print("\nT-Retriever Interactive Query")
    print("=" * 50)
    print("Commands:")
    print("  quit/exit - Exit interactive mode")
    print("  doc:<id>  - Filter to specific document")
    print("  layer:<n> - Query specific layer only")
    print("  graph:on/off - Toggle graph expansion")
    print("  clear     - Clear filters")
    print("=" * 50)
    
    document_filter = None
    layer_filter = None
    use_graph = True
    
    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not query:
            continue
        
        if query.lower() in ["quit", "exit"]:
            break
        
        if query.startswith("doc:"):
            document_filter = query[4:].strip()
            print(f"Filtering to document: {document_filter}")
            continue
        
        if query.startswith("layer:"):
            layer_filter = int(query[6:].strip())
            print(f"Filtering to layer: {layer_filter}")
            continue
        
        if query.startswith("graph:"):
            use_graph = query[6:].strip().lower() == "on"
            print(f"Graph expansion: {'ON' if use_graph else 'OFF'}")
            continue
        
        if query.lower() == "clear":
            document_filter = None
            layer_filter = None
            print("Filters cleared")
            continue
        
        # Execute query
        print("\nSearching...")
        
        if layer_filter is not None:
            results = layer_specific_retrieval(
                query, layer_filter, document_filter, top_k=5, collection_name=collection_name
            )
        elif use_graph and document_filter:
            results = hybrid_retrieval(
                query, document_filter, top_k=5, collection_name=collection_name
            )
        else:
            results = collapsed_tree_retrieval(
                query, document_filter, top_k=5, collection_name=collection_name
            )
        
        if not results:
            print("[ERROR] No results found")
            continue
        
        print(f"\n[OK] Found {len(results)} results:\n")
        for i, r in enumerate(results):
            layer = r.get("layer", 0)
            source = r.get("source", "tree")
            icon = "[S]" if r.get("is_summary") else "[C]"
            print(f"{i+1}. {icon} Layer {layer} | Source: {source} | dist: {r.get('distance', 'N/A'):.3f}")
            print(f"   {r['text'][:200]}...")
            print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("T-Retriever Query System")
        print("\nUsage:")
        print("  python t_query.py interactive             Interactive query mode")
        print("  python t_query.py ask '<question>'        Answer a question")
        print("  python t_query.py ask '<q>' <doc_id>      Answer from specific document")
        print("  python t_query.py search '<query>'        Search and show results")
        print("  python t_query.py hybrid '<query>' <doc>  Use hybrid retrieval")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "interactive":
        interactive_query()
        
    elif command == "ask" and len(sys.argv) >= 3:
        question = sys.argv[2]
        doc_id = sys.argv[3] if len(sys.argv) >= 4 else None
        
        print(f"\nQuestion: {question}")
        if doc_id:
            print(f"Document: {doc_id}")
        print("\nGenerating answer...\n")
        
        result = answer_question(question, document_id=doc_id, verbose=True)
        
        print("=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(result["answer"])
        print("\n" + "=" * 60)
        print("SOURCES:")
        for s in result["sources"][:5]:
            method = s.get("method", "tree")
            print(f"  - {s['id']} ({s['type']}) via {method}")
        
    elif command == "search" and len(sys.argv) >= 3:
        query = sys.argv[2]
        doc_id = sys.argv[3] if len(sys.argv) >= 4 else None
        
        print(f"\nSearching: {query}")
        results = collapsed_tree_retrieval(query, doc_id, top_k=5)
        
        if not results:
            print("No results found")
        else:
            for i, r in enumerate(results):
                print(f"\n{i+1}. [L{r['layer']}] {r['text'][:200]}...")
    
    elif command == "hybrid" and len(sys.argv) >= 4:
        query = sys.argv[2]
        doc_id = sys.argv[3]
        
        print(f"\nHybrid search: {query}")
        print(f"Document: {doc_id}")
        
        results = hybrid_retrieval(query, doc_id, top_k=5)
        
        if not results:
            print("No results found")
        else:
            for i, r in enumerate(results):
                source = r.get("source", "tree")
                score = r.get("fusion_score", 0)
                print(f"\n{i+1}. [L{r['layer']}] Source: {source} | Score: {score:.3f}")
                print(f"   {r['text'][:200]}...")
    
    else:
        print(f"Unknown command: {command}")
