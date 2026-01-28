"""
T-Retriever: Tree-based Hierarchical Retrieval for Textual Graphs

Implementation based on the 2026 paper "T-Retriever: Tree-based Hierarchical 
Retrieval Augmented Generation for Textual Graphs" by Wei et al.

Key innovations over RAPTOR:
1. Combines hierarchical tree retrieval with graph relationships
2. Entity extraction and relationship modeling
3. Graph-aware clustering that respects entity connections
4. Hybrid retrieval: tree traversal + graph expansion
5. Better multi-hop reasoning through graph structure

Architecture:
- Layer 0: Raw chunks with extracted entities
- Layer 1+: Summaries (like RAPTOR) BUT with entity aggregation
- Entity Graph: Connects chunks/summaries via shared entities
- Retrieval: Tree search + graph expansion for related context

This replaces RAPTOR with a more powerful graph-augmented approach.
"""
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
import json
import re

warnings.filterwarnings('ignore', category=UserWarning)

# Import from existing modules
from gemini_client import generate_content, get_embedding
from embeddings import get_embeddings
from storage import get_or_create_collection
from utils import count_tokens
import config


# ============================================================================
# CONFIGURATION
# ============================================================================

# Tree building parameters (similar to RAPTOR)
MAX_TREE_DEPTH = 4              # Maximum layers in tree
MIN_CLUSTER_SIZE = 2            # Minimum nodes to form a cluster
MAX_SUMMARY_TOKENS = 500        # Max tokens for summary context
MIN_NODES_FOR_CLUSTERING = 3    # Minimum nodes needed to cluster

# Entity extraction parameters
MAX_ENTITIES_PER_CHUNK = 15     # Maximum entities to extract per chunk
MIN_ENTITY_FREQUENCY = 2        # Minimum occurrences to keep entity in graph
ENTITY_SIMILARITY_THRESHOLD = 0.85  # Threshold for entity deduplication

# Graph parameters
GRAPH_EDGE_SIMILARITY_THRESHOLD = 0.3   # Min similarity for graph edges
GRAPH_KNN_NEIGHBORS = 8                  # k for k-NN graph
ENTITY_EDGE_WEIGHT = 1.5                 # Boost for entity-based connections

# Retrieval parameters
GRAPH_EXPANSION_HOPS = 2        # How many hops to expand in graph
GRAPH_EXPANSION_TOP_K = 3       # Top neighbors per hop
TREE_RETRIEVAL_TOP_K = 10       # Initial tree retrieval count
HYBRID_ALPHA = 0.6              # Weight for tree vs graph (0.6 = 60% tree)


# ============================================================================
# ENTITY EXTRACTION MODULE
# ============================================================================

def extract_entities(text: str, max_entities: int = MAX_ENTITIES_PER_CHUNK) -> List[Dict]:
    """
    Extract entities using configured mode (fast or LLM)
    
    Industry standard: Use fast extraction (spaCy/rules) by default.
    LLM extraction is 1000x slower and not necessary for good RAG performance.
    """
    try:
        from config import ENTITY_EXTRACTION_MODE
    except ImportError:
        ENTITY_EXTRACTION_MODE = "fast"
    
    if ENTITY_EXTRACTION_MODE == "llm":
        return extract_entities_llm(text, max_entities)
    else:
        return extract_entities_simple(text, max_entities)


def extract_entities_llm(text: str, max_entities: int = MAX_ENTITIES_PER_CHUNK) -> List[Dict]:
    """
    Extract entities from text using LLM (SLOW - use only if needed)
    
    Returns list of entities with:
    - name: Entity name
    - type: Entity type (person, concept, method, organization, etc.)
    - importance: Score 1-10
    
    Args:
        text: Text to extract entities from
        max_entities: Maximum entities to extract
        
    Returns:
        List of entity dicts
    """
    prompt = f"""Extract the most important entities from this text. 
Focus on: people, organizations, concepts, methods, technologies, and key terms.

Return a JSON array with up to {max_entities} entities, each with:
- "name": exact entity name as it appears
- "type": one of [person, organization, concept, method, technology, location, other]
- "importance": 1-10 score (10 = most important)

Text:
{text[:4000]}

Return ONLY valid JSON array, no other text:"""

    try:
        response = generate_content(prompt)
        
        # Parse JSON from response
        # Handle potential markdown code blocks
        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = re.sub(r'^```\w*\n?', '', json_str)
            json_str = re.sub(r'\n?```$', '', json_str)
        
        entities = json.loads(json_str)
        
        # Validate and clean
        valid_entities = []
        for e in entities[:max_entities]:
            if isinstance(e, dict) and "name" in e:
                valid_entities.append({
                    "name": str(e.get("name", "")).strip(),
                    "type": str(e.get("type", "other")).lower(),
                    "importance": min(10, max(1, int(e.get("importance", 5))))
                })
        
        return valid_entities
        
    except Exception as ex:
        print(f"   [WARN] Entity extraction failed: {ex}")
        return extract_entities_simple(text)


def extract_entities_simple(text: str, max_entities: int = MAX_ENTITIES_PER_CHUNK) -> List[Dict]:
    """
    Fast rule-based entity extraction (fallback)
    
    Uses simple heuristics: capitalized phrases, technical terms, etc.
    """
    import re
    
    # Find capitalized phrases (potential proper nouns)
    cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    caps = re.findall(cap_pattern, text)
    
    # Find technical terms (ALL CAPS or camelCase)
    tech_pattern = r'\b([A-Z]{2,}|[a-z]+[A-Z][a-zA-Z]*)\b'
    techs = re.findall(tech_pattern, text)
    
    # Count frequencies
    from collections import Counter
    all_entities = Counter(caps + techs)
    
    # Filter and format
    entities = []
    for name, count in all_entities.most_common(max_entities):
        if len(name) > 2 and count >= 1:
            etype = "technology" if name.isupper() or re.match(r'[a-z]+[A-Z]', name) else "concept"
            entities.append({
                "name": name,
                "type": etype,
                "importance": min(10, count + 3)
            })
    
    return entities


def deduplicate_entities(entities: List[Dict], embeddings: Optional[np.ndarray] = None) -> List[Dict]:
    """
    Deduplicate entities by name similarity and embedding similarity
    
    Args:
        entities: List of entity dicts
        embeddings: Optional precomputed embeddings for entities
        
    Returns:
        Deduplicated entity list
    """
    if not entities:
        return []
    
    # First pass: exact match deduplication (case-insensitive)
    seen = {}
    for e in entities:
        key = e["name"].lower().strip()
        if key not in seen:
            seen[key] = e
        else:
            # Keep the one with higher importance
            if e.get("importance", 0) > seen[key].get("importance", 0):
                seen[key] = e
    
    deduped = list(seen.values())
    
    # Second pass: embedding similarity (if we have enough entities)
    if len(deduped) > 5 and embeddings is None:
        try:
            names = [e["name"] for e in deduped]
            embeddings = np.array(get_embeddings(names))
            
            # Find similar pairs
            sims = cosine_similarity(embeddings)
            to_remove = set()
            
            for i in range(len(deduped)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(deduped)):
                    if j in to_remove:
                        continue
                    if sims[i, j] > ENTITY_SIMILARITY_THRESHOLD:
                        # Remove the less important one
                        if deduped[i].get("importance", 0) >= deduped[j].get("importance", 0):
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break
            
            deduped = [e for idx, e in enumerate(deduped) if idx not in to_remove]
            
        except Exception:
            pass
    
    return deduped


# ============================================================================
# GRAPH BUILDING MODULE
# ============================================================================

class EntityGraph:
    """
    Graph structure connecting chunks via shared entities
    
    Nodes: chunk_ids
    Edges: weighted by shared entities and embedding similarity
    """
    
    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # chunk_id -> {entities, embedding, layer, ...}
        self.edges: Dict[str, Dict[str, float]] = {}  # chunk_id -> {neighbor_id: weight}
        self.entity_index: Dict[str, Set[str]] = {}  # entity_name -> set of chunk_ids
        
    def add_node(self, chunk_id: str, entities: List[Dict], embedding: List[float], 
                 layer: int = 0, metadata: Optional[Dict] = None):
        """Add a chunk node to the graph"""
        self.nodes[chunk_id] = {
            "entities": entities,
            "embedding": embedding,
            "layer": layer,
            "metadata": metadata or {}
        }
        
        # Update entity index
        for e in entities:
            name = e["name"].lower()
            if name not in self.entity_index:
                self.entity_index[name] = set()
            self.entity_index[name].add(chunk_id)
    
    def build_edges(self, similarity_threshold: float = GRAPH_EDGE_SIMILARITY_THRESHOLD):
        """
        Build edges between nodes based on:
        1. Shared entities (weighted by importance)
        2. Embedding similarity
        """
        print(f"   Building graph edges ({len(self.nodes)} nodes)...")
        
        chunk_ids = list(self.nodes.keys())
        n = len(chunk_ids)
        
        if n < 2:
            return
        
        # Get embeddings matrix
        embeddings = np.array([self.nodes[cid]["embedding"] for cid in chunk_ids])
        
        # Compute embedding similarities
        embed_sims = cosine_similarity(embeddings)
        
        # Build edges
        for i in range(n):
            cid_i = chunk_ids[i]
            if cid_i not in self.edges:
                self.edges[cid_i] = {}
            
            entities_i = {e["name"].lower(): e.get("importance", 5) 
                         for e in self.nodes[cid_i]["entities"]}
            
            for j in range(n):
                if i == j:
                    continue
                
                cid_j = chunk_ids[j]
                
                # Compute combined weight
                embed_sim = embed_sims[i, j]
                
                # Entity overlap score
                entities_j = {e["name"].lower(): e.get("importance", 5) 
                             for e in self.nodes[cid_j]["entities"]}
                shared = set(entities_i.keys()) & set(entities_j.keys())
                
                if shared:
                    # Weight by importance of shared entities
                    entity_score = sum(
                        (entities_i[e] + entities_j[e]) / 2 
                        for e in shared
                    ) / (10 * max(len(entities_i), len(entities_j), 1))
                else:
                    entity_score = 0
                
                # Combined weight (entity connections boost similarity)
                weight = embed_sim + entity_score * ENTITY_EDGE_WEIGHT
                
                if weight > similarity_threshold:
                    self.edges[cid_i][cid_j] = weight
        
        # Count edges
        total_edges = sum(len(neighbors) for neighbors in self.edges.values()) // 2
        print(f"   [OK] Built {total_edges} edges")
    
    def get_neighbors(self, chunk_id: str, top_k: int = GRAPH_EXPANSION_TOP_K) -> List[Tuple[str, float]]:
        """Get top-k neighbors sorted by edge weight"""
        if chunk_id not in self.edges:
            return []
        
        neighbors = list(self.edges[chunk_id].items())
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]
    
    def expand_from_nodes(
        self, 
        seed_ids: List[str], 
        hops: int = GRAPH_EXPANSION_HOPS,
        top_k_per_hop: int = GRAPH_EXPANSION_TOP_K
    ) -> List[str]:
        """
        Expand from seed nodes through graph
        
        Returns additional chunk_ids discovered through graph traversal
        """
        visited = set(seed_ids)
        frontier = set(seed_ids)
        
        for hop in range(hops):
            next_frontier = set()
            
            for node_id in frontier:
                neighbors = self.get_neighbors(node_id, top_k_per_hop)
                for neighbor_id, weight in neighbors:
                    if neighbor_id not in visited:
                        next_frontier.add(neighbor_id)
            
            visited.update(next_frontier)
            frontier = next_frontier
            
            if not frontier:
                break
        
        # Return only the expanded nodes (not original seeds)
        return list(visited - set(seed_ids))
    
    def get_chunks_by_entity(self, entity_name: str) -> List[str]:
        """Get all chunk_ids containing a specific entity"""
        return list(self.entity_index.get(entity_name.lower(), set()))
    
    def to_dict(self) -> Dict:
        """Serialize graph to dict for storage"""
        return {
            "nodes": {k: {**v, "embedding": None} for k, v in self.nodes.items()},
            "edges": dict(self.edges),
            "entity_index": {k: list(v) for k, v in self.entity_index.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EntityGraph':
        """Deserialize graph from dict"""
        graph = cls()
        graph.nodes = data.get("nodes", {})
        graph.edges = data.get("edges", {})
        graph.entity_index = {k: set(v) for k, v in data.get("entity_index", {}).items()}
        return graph


# Global graph instance (loaded/saved per document)
_document_graphs: Dict[str, EntityGraph] = {}


def get_document_graph(document_id: str) -> EntityGraph:
    """Get or create entity graph for a document"""
    if document_id not in _document_graphs:
        _document_graphs[document_id] = EntityGraph()
    return _document_graphs[document_id]


def save_document_graph(document_id: str, collection_name: str = "raptor_chunks"):
    """Save graph metadata to ChromaDB collection metadata"""
    graph = _document_graphs.get(document_id)
    if not graph:
        return
    
    # Store graph structure as JSON in a special metadata chunk
    collection = get_or_create_collection(collection_name)
    
    graph_data = json.dumps(graph.to_dict())
    graph_chunk_id = f"{document_id}_graph_metadata"
    
    # Check if exists and update, or add new
    try:
        existing = collection.get(ids=[graph_chunk_id])
        if existing["ids"]:
            collection.update(
                ids=[graph_chunk_id],
                documents=[graph_data],
                metadatas=[{
                    "document_id": document_id,
                    "type": "graph_metadata",
                    "layer": -1
                }]
            )
        else:
            # Need a dummy embedding for ChromaDB
            collection.add(
                ids=[graph_chunk_id],
                documents=[graph_data],
                metadatas=[{
                    "document_id": document_id,
                    "type": "graph_metadata",
                    "layer": -1
                }],
                embeddings=[[0.0] * 768]  # Dummy embedding
            )
    except Exception as e:
        print(f"   [WARN] Failed to save graph: {e}")


def load_document_graph(document_id: str, collection_name: str = "raptor_chunks") -> EntityGraph:
    """Load graph from ChromaDB"""
    collection = get_or_create_collection(collection_name)
    graph_chunk_id = f"{document_id}_graph_metadata"
    
    try:
        result = collection.get(ids=[graph_chunk_id], include=["documents"])
        if result["ids"] and result["documents"]:
            data = json.loads(result["documents"][0])
            graph = EntityGraph.from_dict(data)
            _document_graphs[document_id] = graph
            return graph
    except Exception:
        pass
    
    return get_document_graph(document_id)


# ============================================================================
# CLUSTERING MODULE (Similar to RAPTOR but entity-aware)
# ============================================================================

def cluster_with_entities(
    embeddings: np.ndarray,
    entities_per_node: List[List[Dict]],
    layer: int = 0
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Entity-aware clustering using Leiden algorithm
    
    Clusters nodes considering both embedding similarity and entity overlap.
    This is the key improvement over standard RAPTOR clustering.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        print("   [WARN] Leiden not available, using GMM fallback")
        return cluster_embeddings_gmm(embeddings)
    
    n_samples = embeddings.shape[0]
    
    if n_samples < MIN_NODES_FOR_CLUSTERING:
        return [[i for i in range(n_samples)]], np.ones((n_samples, 1))
    
    # Build similarity matrix combining embeddings + entity overlap
    embed_sims = cosine_similarity(embeddings)
    
    # Entity overlap matrix
    entity_sims = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        ents_i = {e["name"].lower() for e in entities_per_node[i]}
        for j in range(i + 1, n_samples):
            ents_j = {e["name"].lower() for e in entities_per_node[j]}
            if ents_i and ents_j:
                overlap = len(ents_i & ents_j) / len(ents_i | ents_j)
                entity_sims[i, j] = entity_sims[j, i] = overlap
    
    # Combined similarity
    combined_sims = embed_sims * 0.7 + entity_sims * 0.3
    
    # Build graph for Leiden
    edges = []
    weights = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if combined_sims[i, j] > GRAPH_EDGE_SIMILARITY_THRESHOLD:
                edges.append((i, j))
                weights.append(combined_sims[i, j])
    
    if not edges:
        # No edges, return single cluster
        return [[i for i in range(n_samples)]], np.ones((n_samples, 1))
    
    g = ig.Graph(n=n_samples, edges=edges, directed=False)
    g.es['weight'] = weights
    
    # Layer-adaptive resolution (broader clusters at higher layers)
    resolution = 1.0 * (0.7 ** layer)
    
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=42
    )
    
    clusters = [list(community) for community in partition]
    
    # Build membership matrix
    membership = np.zeros((n_samples, len(clusters)))
    for cluster_idx, members in enumerate(clusters):
        for node_idx in members:
            membership[node_idx, cluster_idx] = 1.0
    
    clusters = [c for c in clusters if len(c) > 0]
    
    return clusters, membership


def cluster_embeddings_gmm(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None
) -> Tuple[List[List[int]], np.ndarray]:
    """Fallback GMM clustering (from RAPTOR)"""
    from sklearn.mixture import GaussianMixture
    
    n_samples = embeddings.shape[0]
    
    if n_samples < MIN_NODES_FOR_CLUSTERING:
        return [[i for i in range(n_samples)]], np.ones((n_samples, 1))
    
    # Auto-detect cluster count using BIC
    if n_clusters is None:
        best_bic = float('inf')
        best_n = 2
        for n in range(2, min(10, n_samples)):
            try:
                gmm = GaussianMixture(n_components=n, random_state=42)
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
            except:
                continue
        n_clusters = best_n
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(embeddings)
    
    labels = gmm.predict(embeddings)
    probabilities = gmm.predict_proba(embeddings)
    
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(i)
    
    clusters = [c for c in clusters if len(c) > 0]
    
    return clusters, probabilities


# ============================================================================
# SUMMARIZATION MODULE (Entity-aware)
# ============================================================================

def summarize_cluster_with_entities(
    texts: List[str],
    entities_per_text: List[List[Dict]],
    layer: int
) -> Tuple[str, List[Dict]]:
    """
    Generate summary for a cluster, aggregating and prioritizing entities
    
    Returns:
        Tuple of (summary_text, aggregated_entities)
    """
    # Aggregate entities from all texts in cluster
    entity_counts = {}
    for ents in entities_per_text:
        for e in ents:
            name = e["name"].lower()
            if name not in entity_counts:
                entity_counts[name] = {
                    "name": e["name"],
                    "type": e["type"],
                    "importance": e.get("importance", 5),
                    "count": 0
                }
            entity_counts[name]["count"] += 1
            entity_counts[name]["importance"] = max(
                entity_counts[name]["importance"],
                e.get("importance", 5)
            )
    
    # Sort by importance * count
    sorted_entities = sorted(
        entity_counts.values(),
        key=lambda x: x["importance"] * x["count"],
        reverse=True
    )[:MAX_ENTITIES_PER_CHUNK]
    
    # Build entity list string for prompt
    key_entities = ", ".join(e["name"] for e in sorted_entities[:10])
    
    # Combine texts
    combined = "\n\n---\n\n".join(texts)
    if len(combined) > MAX_SUMMARY_TOKENS * 4:
        combined = combined[:MAX_SUMMARY_TOKENS * 4] + "..."
    
    prompt = f"""You are creating a summary for a hierarchical document retrieval system.
This is layer {layer + 1} of the tree.

Key entities in these texts: {key_entities}

Summarize the following text chunks into a coherent summary that:
1. Captures the main themes and key information
2. Preserves important entities, facts, and relationships
3. Emphasizes connections between the key entities mentioned
4. Is self-contained and understandable

Text chunks to summarize:
{combined}

Provide a clear, comprehensive summary (2-4 paragraphs):"""

    try:
        summary = generate_content(prompt)
    except Exception as e:
        print(f"   [WARN] Summarization failed: {e}")
        summary = " ".join([t.split('.')[0] + '.' for t in texts[:3]])
    
    # Clean up aggregated entities
    aggregated = [
        {"name": e["name"], "type": e["type"], "importance": e["importance"]}
        for e in sorted_entities
    ]
    
    return summary, aggregated


# ============================================================================
# TREE BUILDING MODULE (T-Retriever style)
# ============================================================================

def get_layer_chunks_with_entities(
    document_id: str,
    layer: int,
    collection_name: str = "raptor_chunks"
) -> Tuple[List[str], List[str], List[Dict], List[List[float]], List[List[Dict]]]:
    """
    Get all chunks at a layer with their entities
    
    Returns:
        (chunk_ids, texts, metadatas, embeddings, entities_list)
    """
    collection = get_or_create_collection(collection_name)
    
    results = collection.get(
        where={
            "$and": [
                {"document_id": {"$eq": document_id}},
                {"layer": {"$eq": layer}}
            ]
        },
        include=["documents", "metadatas", "embeddings"]
    )
    
    # Extract entities from metadata
    entities_list = []
    for meta in results.get("metadatas", []):
        entities_json = meta.get("entities", "[]")
        try:
            entities = json.loads(entities_json) if isinstance(entities_json, str) else entities_json
        except:
            entities = []
        entities_list.append(entities)
    
    return (
        results.get("ids", []),
        results.get("documents", []),
        results.get("metadatas", []),
        results.get("embeddings", []),
        entities_list
    )


def store_chunks_with_entities(
    chunks: List[Dict],
    document_id: str,
    layer: int,
    collection_name: str = "raptor_chunks"
) -> List[str]:
    """
    Store chunks with entity metadata
    
    Args:
        chunks: List of dicts with 'text', 'entities', 'child_ids' (optional)
        document_id: Document identifier
        layer: Tree layer
        collection_name: Collection name
        
    Returns:
        List of chunk IDs
    """
    collection = get_or_create_collection(collection_name)
    graph = get_document_graph(document_id)
    
    chunk_ids = []
    chunk_texts = []
    chunk_embeddings = []
    chunk_metadatas = []
    
    for idx, chunk in enumerate(chunks):
        text = chunk["text"]
        entities = chunk.get("entities", [])
        
        is_summary = layer > 0 or chunk.get("is_summary", False)
        chunk_id = f"{document_id}_L{layer}_{'summary' if is_summary else 'chunk'}{idx}"
        
        metadata = {
            "document_id": document_id,
            "layer": layer,
            "chunk_index": idx,
            "is_summary": is_summary,
            "entities": json.dumps(entities),
            "entity_names": ",".join(e["name"] for e in entities[:10]),
            "token_count": count_tokens(text),
            "content_type": "summary" if is_summary else "text"
        }
        
        if "child_ids" in chunk:
            metadata["child_ids"] = ",".join(chunk["child_ids"])
        
        chunk_ids.append(chunk_id)
        chunk_texts.append(text)
        chunk_metadatas.append(metadata)
    
    # Generate embeddings
    print(f"   Generating embeddings for {len(chunks)} chunks...")
    chunk_embeddings = get_embeddings(chunk_texts)
    
    # Add to graph
    for cid, emb, meta, chunk in zip(chunk_ids, chunk_embeddings, chunk_metadatas, chunks):
        entities = chunk.get("entities", [])
        graph.add_node(cid, entities, emb, layer, meta)
    
    # Store in ChromaDB
    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=chunk_embeddings,
        metadatas=chunk_metadatas
    )
    
    return chunk_ids


def build_layer_tretriever(
    document_id: str,
    current_layer: int,
    collection_name: str = "raptor_chunks"
) -> Tuple[int, List[str]]:
    """
    Build one layer of the T-Retriever tree
    
    Entity-aware clustering + entity-preserving summarization
    """
    print(f"\n[BUILD] Building layer {current_layer + 1} from layer {current_layer}...")

    # Get chunks at current layer
    chunk_ids, texts, metadatas, embeddings, entities_list = get_layer_chunks_with_entities(
        document_id, current_layer, collection_name
    )
    
    if not chunk_ids:
        print(f"   No chunks found at layer {current_layer}")
        return 0, []
    
    print(f"   Found {len(chunk_ids)} chunks at layer {current_layer}")
    
    if len(chunk_ids) < MIN_NODES_FOR_CLUSTERING:
        print(f"   Too few chunks for clustering. Stopping.")
        return 0, []
    
    # Convert embeddings
    if embeddings is None or len(embeddings) == 0:
        print(f"   Fetching embeddings...")
        collection = get_or_create_collection(collection_name)
        results = collection.get(ids=chunk_ids, include=["embeddings"])
        embeddings = results["embeddings"]
    
    embeddings_array = np.array(embeddings)
    
    # Entity-aware clustering
    print(f"   Clustering with entity awareness...")
    clusters, _ = cluster_with_entities(embeddings_array, entities_list, current_layer)
    
    print(f"   Created {len(clusters)} clusters")
    
    # Generate summaries
    summaries = []
    for cluster_idx, cluster_indices in enumerate(tqdm(clusters, desc="   Summarizing", unit="cluster")):
        if len(cluster_indices) < MIN_CLUSTER_SIZE:
            continue
        
        cluster_texts = [texts[i] for i in cluster_indices]
        cluster_entities = [entities_list[i] for i in cluster_indices]
        cluster_chunk_ids = [chunk_ids[i] for i in cluster_indices]
        
        # Generate entity-aware summary
        summary_text, aggregated_entities = summarize_cluster_with_entities(
            cluster_texts, cluster_entities, current_layer
        )
        
        summaries.append({
            "text": summary_text,
            "entities": aggregated_entities,
            "child_ids": cluster_chunk_ids,
            "cluster_id": f"L{current_layer + 1}_C{cluster_idx}",
            "is_summary": True
        })
    
    if not summaries:
        print(f"   No valid clusters formed")
        return 0, []
    
    # Store summaries
    summary_ids = store_chunks_with_entities(
        summaries, document_id, current_layer + 1, collection_name
    )
    
    print(f"   [OK] Created {len(summary_ids)} summaries at layer {current_layer + 1}")
    
    return len(summary_ids), summary_ids


def build_tretriever_tree(
    document_id: str,
    collection_name: str = "raptor_chunks",
    max_depth: int = MAX_TREE_DEPTH
) -> Dict:
    """
    Build complete T-Retriever tree for a document
    
    This is the main tree-building function that replaces RAPTOR's build_raptor_tree.
    
    Process:
    1. Extract entities from base chunks (layer 0)
    2. Build entity graph
    3. Recursively cluster (entity-aware) and summarize
    4. Save graph structure
    
    Args:
        document_id: Document identifier
        collection_name: ChromaDB collection name
        max_depth: Maximum tree depth
        
    Returns:
        Dict with tree statistics
    """
    print("=" * 60)
    print(f"BUILDING T-RETRIEVER TREE: {document_id}")
    print("=" * 60)
    
    stats = {
        "document_id": document_id,
        "layers": {},
        "total_nodes": 0,
        "tree_depth": 0,
        "total_entities": 0
    }
    
    # Get base chunks
    chunk_ids, texts, metadatas, embeddings, entities_list = get_layer_chunks_with_entities(
        document_id, 0, collection_name
    )
    
    if not chunk_ids:
        print(f"[ERROR] No base chunks found for document: {document_id}")
        return stats
    
    # Check if entities already extracted
    has_entities = any(len(e) > 0 for e in entities_list)
    
    if not has_entities:
        print(f"\nExtracting entities from {len(chunk_ids)} base chunks (fast mode)...")
        
        # Extract entities for base chunks using configured mode
        collection = get_or_create_collection(collection_name)
        
        for i, (cid, text, meta) in enumerate(tqdm(
            zip(chunk_ids, texts, metadatas),
            desc="   Extracting entities",
            total=len(chunk_ids)
        )):
            entities = extract_entities(text)  # Uses fast mode by default
            entities_list[i] = entities
            
            # Update metadata in ChromaDB
            meta["entities"] = json.dumps(entities)
            meta["entity_names"] = ",".join(e["name"] for e in entities[:10])
            
            collection.update(
                ids=[cid],
                metadatas=[meta]
            )
    
    # Build entity graph from base chunks
    print(f"\nBuilding entity graph...")
    graph = get_document_graph(document_id)
    
    # Get embeddings if needed
    if embeddings is None or len(embeddings) == 0:
        collection = get_or_create_collection(collection_name)
        results = collection.get(ids=chunk_ids, include=["embeddings"])
        embeddings = results["embeddings"]
    
    # Add nodes to graph
    for cid, emb, ents, meta in zip(chunk_ids, embeddings, entities_list, metadatas):
        graph.add_node(cid, ents, emb, 0, meta)
    
    # Build graph edges
    graph.build_edges()
    
    # Count entities
    all_entities = set()
    for ents in entities_list:
        for e in ents:
            all_entities.add(e["name"].lower())
    stats["total_entities"] = len(all_entities)
    
    stats["layers"][0] = len(chunk_ids)
    stats["total_nodes"] = len(chunk_ids)
    
    print(f"\nBase layer: {len(chunk_ids)} chunks, {stats['total_entities']} unique entities")
    
    # Build tree layers
    current_layer = 0
    while current_layer < max_depth:
        num_summaries, summary_ids = build_layer_tretriever(
            document_id, current_layer, collection_name
        )
        
        if num_summaries == 0:
            break
        
        current_layer += 1
        stats["layers"][current_layer] = num_summaries
        stats["total_nodes"] += num_summaries
        
        if num_summaries <= MIN_CLUSTER_SIZE:
            print(f"   Reached root level with {num_summaries} node(s)")
            break
    
    stats["tree_depth"] = current_layer + 1
    
    # Save graph
    save_document_graph(document_id, collection_name)
    
    # Print summary
    print("\n" + "=" * 60)
    print("T-RETRIEVER TREE COMPLETE")
    print("=" * 60)
    print(f"Document: {document_id}")
    print(f"Tree depth: {stats['tree_depth']} layers")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Unique entities: {stats['total_entities']}")
    print(f"Graph edges: {sum(len(n) for n in graph.edges.values()) // 2}")
    print("\nLayer breakdown:")
    for layer, count in sorted(stats["layers"].items()):
        layer_type = "base chunks" if layer == 0 else "summaries"
        print(f"  Layer {layer}: {count} {layer_type}")
    print("=" * 60)
    
    return stats


def delete_tree_layers(
    document_id: str,
    min_layer: int = 1,
    collection_name: str = "raptor_chunks"
) -> int:
    """Delete summary layers for rebuilding"""
    collection = get_or_create_collection(collection_name)
    
    results = collection.get(
        where={
            "$and": [
                {"document_id": {"$eq": document_id}},
                {"layer": {"$gte": min_layer}}
            ]
        }
    )
    
    if not results["ids"]:
        return 0
    
    collection.delete(ids=results["ids"])
    print(f"[DEL] Deleted {len(results['ids'])} summary chunks (layer >= {min_layer})")
    
    return len(results["ids"])


def rebuild_tree(
    document_id: str,
    collection_name: str = "raptor_chunks"
) -> Dict:
    """Rebuild T-Retriever tree"""
    print(f"[REBUILD] Rebuilding T-Retriever tree for: {document_id}")
    
    delete_tree_layers(document_id, min_layer=1, collection_name=collection_name)
    
    # Clear graph
    if document_id in _document_graphs:
        del _document_graphs[document_id]
    
    return build_tretriever_tree(document_id, collection_name)


def get_tree_stats(
    document_id: str,
    collection_name: str = "raptor_chunks"
) -> Dict:
    """Get T-Retriever tree statistics"""
    collection = get_or_create_collection(collection_name)
    
    results = collection.get(
        where={"document_id": {"$eq": document_id}},
        include=["metadatas"]
    )
    
    if not results["ids"]:
        return {"document_id": document_id, "exists": False}
    
    layers = {}
    total_entities = set()
    
    for meta in results["metadatas"]:
        layer = meta.get("layer", 0)
        if layer >= 0:  # Exclude graph metadata (layer -1)
            layers[layer] = layers.get(layer, 0) + 1
        
        # Count entities
        entity_names = meta.get("entity_names", "")
        if entity_names:
            total_entities.update(entity_names.split(","))
    
    # Load graph stats
    graph = load_document_graph(document_id, collection_name)
    graph_edges = sum(len(n) for n in graph.edges.values()) // 2 if graph.edges else 0
    
    return {
        "document_id": document_id,
        "exists": True,
        "total_nodes": len([m for m in results["metadatas"] if m.get("layer", 0) >= 0]),
        "tree_depth": max(layers.keys()) + 1 if layers else 0,
        "layers": layers,
        "unique_entities": len(total_entities),
        "graph_edges": graph_edges
    }


# ============================================================================
# CLI / TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("T-Retriever Tree Builder")
        print("\nUsage:")
        print("  python t_retriever.py build <document_id>    Build tree for document")
        print("  python t_retriever.py rebuild <document_id>  Rebuild tree")
        print("  python t_retriever.py stats <document_id>    Show tree statistics")
        print("  python t_retriever.py delete <document_id>   Delete summary layers")
        print("  python t_retriever.py extract <document_id>  Extract entities only")
        print("\nNote: Document must be uploaded first with `python main.py upload <file>`")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "build" and len(sys.argv) >= 3:
        doc_id = sys.argv[2]
        build_tretriever_tree(doc_id)
        
    elif command == "rebuild" and len(sys.argv) >= 3:
        doc_id = sys.argv[2]
        rebuild_tree(doc_id)
        
    elif command == "stats" and len(sys.argv) >= 3:
        doc_id = sys.argv[2]
        stats = get_tree_stats(doc_id)
        if stats.get("exists"):
            print(f"\nT-Retriever Tree Stats: {doc_id}")
            print(f"   Total nodes: {stats['total_nodes']}")
            print(f"   Tree depth: {stats['tree_depth']} layers")
            print(f"   Unique entities: {stats['unique_entities']}")
            print(f"   Graph edges: {stats['graph_edges']}")
            for layer, count in sorted(stats["layers"].items()):
                print(f"   Layer {layer}: {count} nodes")
        else:
            print(f"[ERROR] No tree found for: {doc_id}")
            
    elif command == "delete" and len(sys.argv) >= 3:
        doc_id = sys.argv[2]
        delete_tree_layers(doc_id)
        
    elif command == "extract" and len(sys.argv) >= 3:
        doc_id = sys.argv[2]
        print(f"Extracting entities for: {doc_id}")
        # Just build tree which will extract entities
        build_tretriever_tree(doc_id)
        
    else:
        print(f"Unknown command: {command}")
        print("Run without arguments for help")
