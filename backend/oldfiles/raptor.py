"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

Implementation based on the ICLR 2024 paper by Stanford researchers.
Enhanced with techniques from:
- LATTICE (arXiv:2510.13217): LLM-guided hierarchical retrieval
- BookRAG (arXiv:2512.03413): Query classification and adaptive retrieval
- Frontiers 2026 paper: Leiden clustering with layer-aware parameters

Key components:
1. Clustering: GMM + UMAP OR Leiden algorithm (configurable)
2. Summarization: LLM-generated summaries of clusters
3. Tree building: Recursive bottom-up construction
4. Layer-aware parameters: Adaptive clustering granularity per tree level
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import warnings

# Suppress UMAP warnings about numba
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

# Tree building parameters
MAX_TREE_DEPTH = 5              # Maximum layers in tree
MIN_CLUSTER_SIZE = 2            # Minimum nodes to form a cluster
MAX_SUMMARY_TOKENS = 500        # Max tokens for summary context
MIN_NODES_FOR_CLUSTERING = 3    # Minimum nodes needed to cluster
SOFT_CLUSTER_THRESHOLD = 0.1    # GMM probability threshold for soft clustering

# UMAP parameters
UMAP_N_COMPONENTS = 10          # Reduce to this many dimensions
UMAP_MIN_DIST = 0.0             # Minimum distance for UMAP
UMAP_METRIC = 'cosine'          # Distance metric
UMAP_RECOMMENDED_MIN_SAMPLES = 15  # Minimum samples for reliable UMAP (matches n_neighbors default)

# Leiden clustering parameters (from Frontiers 2026 paper)
LEIDEN_BASE_RESOLUTION = 1.0    # Base resolution for Leiden algorithm
LEIDEN_RESOLUTION_DECAY = 0.7   # Resolution multiplier per layer (lower = broader clusters)
LEIDEN_KNN_NEIGHBORS = 10       # k for k-NN graph construction
LEIDEN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity to create edge

# Clustering method selection
CLUSTERING_METHOD = "leiden"    # Options: "gmm", "leiden"


# ============================================================================
# CLUSTERING MODULE
# ============================================================================

def reduce_dimensions(embeddings: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """
    Reduce embedding dimensions using UMAP
    
    Args:
        embeddings: High-dimensional embedding matrix (n_samples, n_features)
        n_neighbors: Controls local vs global structure preservation
                     Higher = more global structure, Lower = more local
    
    Returns:
        Reduced dimension embeddings
    """
    try:
        import umap
        
        n_samples = embeddings.shape[0]
        
        # UMAP needs at least n_neighbors samples
        if n_samples < n_neighbors:
            n_neighbors = max(2, n_samples - 1)
        
        # Adjust n_components if we have few samples
        n_components = min(UMAP_N_COMPONENTS, n_samples - 1)
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=42
        )
        
        reduced = reducer.fit_transform(embeddings)
        return reduced
        
    except Exception as e:
        print(f"   ⚠️ UMAP reduction failed: {e}")
        # Fallback: use PCA
        from sklearn.decomposition import PCA
        n_components = min(UMAP_N_COMPONENTS, embeddings.shape[0] - 1, embeddings.shape[1])
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embeddings)


def find_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10) -> int:
    """
    Find optimal number of clusters using BIC (Bayesian Information Criterion)
    
    Args:
        embeddings: Embedding matrix
        max_clusters: Maximum clusters to try
        
    Returns:
        Optimal number of clusters
    """
    n_samples = embeddings.shape[0]
    max_clusters = min(max_clusters, n_samples - 1)
    
    if max_clusters < 2:
        return 1
    
    best_bic = float('inf')
    best_n = 2
    
    for n in range(2, max_clusters + 1):
        try:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception:
            continue
    
    return best_n


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    use_soft_clustering: bool = True
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Cluster embeddings using GMM with optional soft clustering
    
    Args:
        embeddings: Embedding matrix (n_samples, n_features)
        n_clusters: Number of clusters (None = auto-detect with BIC)
        use_soft_clustering: If True, nodes can belong to multiple clusters
        
    Returns:
        Tuple of (cluster_assignments, probabilities)
        - cluster_assignments: List of lists, each inner list contains indices for that cluster
        - probabilities: GMM probability matrix (n_samples, n_clusters)
    """
    n_samples = embeddings.shape[0]
    
    if n_samples < MIN_NODES_FOR_CLUSTERING:
        # Not enough nodes to cluster - return single cluster with all
        return [[i for i in range(n_samples)]], np.ones((n_samples, 1))
    
    # Reduce dimensions first
    print(f"   Reducing dimensions ({embeddings.shape[1]}D → {UMAP_N_COMPONENTS}D)...")
    reduced = reduce_dimensions(embeddings)
    
    # Find optimal cluster count if not specified
    if n_clusters is None:
        n_clusters = find_optimal_clusters(reduced)
    
    print(f"   Clustering into {n_clusters} groups...")
    
    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        random_state=42,
        max_iter=100
    )
    gmm.fit(reduced)
    
    # Get probabilities for each sample belonging to each cluster
    probabilities = gmm.predict_proba(reduced)
    
    # Build cluster assignments
    clusters = [[] for _ in range(n_clusters)]
    
    if use_soft_clustering:
        # Soft clustering: assign to all clusters where probability > threshold
        for i in range(n_samples):
            for j in range(n_clusters):
                if probabilities[i, j] > SOFT_CLUSTER_THRESHOLD:
                    clusters[j].append(i)
    else:
        # Hard clustering: assign to highest probability cluster only
        labels = gmm.predict(reduced)
        for i, label in enumerate(labels):
            clusters[label].append(i)
    
    # Remove empty clusters
    clusters = [c for c in clusters if len(c) > 0]
    
    return clusters, probabilities


# ============================================================================
# LEIDEN CLUSTERING MODULE (from Frontiers 2026 paper)
# ============================================================================

def build_similarity_graph(embeddings: np.ndarray, k: int = LEIDEN_KNN_NEIGHBORS) -> 'igraph.Graph':
    """
    Build a k-NN similarity graph from embeddings for Leiden clustering.
    
    Uses cosine similarity to connect similar nodes.
    
    Args:
        embeddings: Embedding matrix (n_samples, n_features)
        k: Number of nearest neighbors per node
        
    Returns:
        igraph.Graph with similarity weights on edges
    """
    import igraph as ig
    from sklearn.metrics.pairwise import cosine_similarity
    
    n_samples = embeddings.shape[0]
    
    # Compute pairwise cosine similarity
    similarities = cosine_similarity(embeddings)
    
    # Build edge list from k-NN
    edges = []
    weights = []
    
    for i in range(n_samples):
        # Get k most similar nodes (excluding self)
        sim_scores = similarities[i].copy()
        sim_scores[i] = -1  # Exclude self
        
        # Get top-k neighbors
        top_k_indices = np.argsort(sim_scores)[-k:]
        
        for j in top_k_indices:
            if sim_scores[j] > LEIDEN_SIMILARITY_THRESHOLD:
                # Add edge (avoid duplicates by only adding i < j)
                if i < j:
                    edges.append((i, j))
                    weights.append(float(sim_scores[j]))
    
    # Create graph
    g = ig.Graph(n=n_samples, edges=edges, directed=False)
    g.es['weight'] = weights
    
    return g


def get_layer_resolution(layer: int) -> float:
    """
    Calculate Leiden resolution parameter for a given layer.
    
    Higher layers use lower resolution → broader clusters.
    This implements "layer-aware dual-adaptive parameters" from the Frontiers paper.
    
    Args:
        layer: Current tree layer (0 = base chunks)
        
    Returns:
        Resolution parameter for Leiden algorithm
    """
    return LEIDEN_BASE_RESOLUTION * (LEIDEN_RESOLUTION_DECAY ** layer)


def cluster_with_leiden(
    embeddings: np.ndarray,
    layer: int = 0,
    use_soft_clustering: bool = True
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Cluster embeddings using Leiden algorithm with layer-aware resolution.
    
    Leiden is a community detection algorithm that:
    - Scales as O(n log n) vs O(n²) for GMM
    - Naturally finds graph communities (arbitrary shapes)
    - Adapts cluster granularity via resolution parameter
    
    Based on: "Enhancing RAPTOR with Semantic Chunking and Adaptive Graph Clustering"
    (Frontiers in Computer Science 2026)
    
    Args:
        embeddings: Embedding matrix (n_samples, n_features)
        layer: Current tree layer (affects resolution)
        use_soft_clustering: If True, nodes can belong to overlapping clusters
        
    Returns:
        Tuple of (cluster_assignments, membership_matrix)
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        print("   ⚠️ Leiden clustering requires: pip install python-igraph leidenalg")
        print("   Falling back to GMM clustering...")
        return cluster_embeddings(embeddings)
    
    n_samples = embeddings.shape[0]
    
    if n_samples < MIN_NODES_FOR_CLUSTERING:
        return [[i for i in range(n_samples)]], np.ones((n_samples, 1))
    
    # Get layer-aware resolution
    resolution = get_layer_resolution(layer)
    print(f"   Building similarity graph ({n_samples} nodes)...")
    
    # Build k-NN graph
    graph = build_similarity_graph(embeddings)
    
    print(f"   Leiden clustering (resolution={resolution:.3f}, layer={layer})...")
    
    # Run Leiden algorithm
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=42
    )
    
    # Extract clusters
    n_clusters = len(partition)
    clusters = [list(community) for community in partition]
    
    # Build membership matrix (for compatibility with GMM output)
    membership = np.zeros((n_samples, n_clusters))
    for cluster_idx, members in enumerate(clusters):
        for node_idx in members:
            membership[node_idx, cluster_idx] = 1.0
    
    # Soft clustering: add nodes to neighboring clusters if similarity is high
    if use_soft_clustering and n_clusters > 1:
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        for i in range(n_samples):
            home_cluster = np.argmax(membership[i])
            
            # Check if this node should also belong to other clusters
            for cluster_idx, members in enumerate(clusters):
                if cluster_idx == home_cluster:
                    continue
                    
                # Calculate average similarity to cluster members
                if len(members) > 0:
                    avg_sim = np.mean([similarities[i, j] for j in members])
                    
                    # If highly similar to another cluster, add soft membership
                    if avg_sim > SOFT_CLUSTER_THRESHOLD + 0.2:  # Slightly higher threshold
                        membership[i, cluster_idx] = avg_sim
                        if i not in clusters[cluster_idx]:
                            clusters[cluster_idx].append(i)
    
    # Remove empty clusters
    non_empty = [(c, membership[:, idx]) for idx, c in enumerate(clusters) if len(c) > 0]
    clusters = [c for c, _ in non_empty]
    
    print(f"   ✓ Found {len(clusters)} communities")
    
    return clusters, membership


def cluster_adaptive(
    embeddings: np.ndarray,
    layer: int = 0,
    method: str = CLUSTERING_METHOD,
    use_soft_clustering: bool = True
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Unified clustering interface with method selection.
    
    Automatically chooses between GMM and Leiden based on configuration,
    with fallback handling.
    
    Args:
        embeddings: Embedding matrix
        layer: Current tree layer
        method: "gmm" or "leiden"
        use_soft_clustering: Allow nodes in multiple clusters
        
    Returns:
        Tuple of (cluster_assignments, probabilities/membership)
    """
    if method == "leiden":
        try:
            return cluster_with_leiden(embeddings, layer, use_soft_clustering)
        except Exception as e:
            print(f"   ⚠️ Leiden failed: {e}")
            print(f"   Falling back to GMM...")
            return cluster_embeddings(embeddings, use_soft_clustering=use_soft_clustering)
    else:
        return cluster_embeddings(embeddings, use_soft_clustering=use_soft_clustering)


# ============================================================================
# SUMMARIZATION MODULE
# ============================================================================

def summarize_cluster(texts: List[str], layer: int) -> str:
    """
    Generate a summary of clustered texts using Gemini
    
    Args:
        texts: List of text chunks in the cluster
        layer: Current tree layer (for context)
        
    Returns:
        Summary text
    """
    # Combine texts with clear separation
    combined = "\n\n---\n\n".join(texts)
    
    # Truncate if too long
    max_chars = MAX_SUMMARY_TOKENS * 4  # Rough estimate
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "..."
    
    prompt = f"""You are creating a summary for a hierarchical document retrieval system.
This is layer {layer + 1} of the tree (higher layers = more abstract summaries).

Summarize the following text chunks into a coherent summary that:
1. Captures the main themes and key information
2. Preserves important details, names, numbers, and facts
3. Maintains connections between different topics
4. Is self-contained and understandable without the original text

Text chunks to summarize:
{combined}

Provide a clear, comprehensive summary (2-4 paragraphs):"""

    try:
        summary = generate_content(prompt)
        return summary
    except Exception as e:
        print(f"   ⚠️ Summarization failed: {e}")
        # Fallback: concatenate first sentences
        fallback = " ".join([t.split('.')[0] + '.' for t in texts[:3]])
        return fallback


# ============================================================================
# TREE BUILDING MODULE
# ============================================================================

def get_layer_chunks(
    document_id: str, 
    layer: int, 
    collection_name: str = "raptor_chunks"
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Get all chunks for a document at a specific layer
    
    Returns:
        Tuple of (chunk_ids, texts, metadatas)
    """
    collection = get_or_create_collection(collection_name)
    
    # Query for all chunks at this layer for this document
    results = collection.get(
        where={
            "$and": [
                {"document_id": {"$eq": document_id}},
                {"layer": {"$eq": layer}}
            ]
        },
        include=["documents", "metadatas", "embeddings"]
    )
    
    return results["ids"], results["documents"], results["metadatas"], results.get("embeddings", [])


def store_summary_chunks(
    summaries: List[Dict],
    document_id: str,
    layer: int,
    collection_name: str = "raptor_chunks"
) -> List[str]:
    """
    Store summary chunks at a higher layer
    
    Args:
        summaries: List of dicts with 'text', 'child_ids', 'cluster_id'
        document_id: Document identifier
        layer: Layer number for these summaries
        collection_name: ChromaDB collection name
        
    Returns:
        List of stored chunk IDs
    """
    collection = get_or_create_collection(collection_name)
    
    chunk_ids = []
    chunk_texts = []
    chunk_embeddings = []
    chunk_metadatas = []
    
    for idx, summary in enumerate(summaries):
        chunk_id = f"{document_id}_L{layer}_summary{idx}"
        text = summary["text"]
        
        metadata = {
            "document_id": document_id,
            "layer": layer,
            "chunk_index": idx,
            "is_summary": True,
            "child_ids": ",".join(summary["child_ids"]),
            "cluster_id": summary.get("cluster_id", f"cluster_{idx}"),
            "token_count": count_tokens(text),
            "content_type": "summary",
            "content_types": "summary",
            "has_images": False,
            "has_tables": False,
        }
        
        chunk_ids.append(chunk_id)
        chunk_texts.append(text)
        chunk_metadatas.append(metadata)
    
    # Generate embeddings for summaries
    print(f"   Generating embeddings for {len(summaries)} summaries...")
    chunk_embeddings = get_embeddings(chunk_texts)
    
    # Store in ChromaDB
    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=chunk_embeddings,
        metadatas=chunk_metadatas
    )
    
    return chunk_ids


def build_layer(
    document_id: str,
    current_layer: int,
    collection_name: str = "raptor_chunks"
) -> Tuple[int, List[str]]:
    """
    Build one layer of the RAPTOR tree
    
    Takes all chunks at current_layer, clusters them, summarizes clusters,
    and stores summaries at current_layer + 1.
    
    Args:
        document_id: Document identifier
        current_layer: Layer to cluster (summaries go to current_layer + 1)
        collection_name: ChromaDB collection name
        
    Returns:
        Tuple of (number of summaries created, list of summary IDs)
    """
    print(f"\n🔨 Building layer {current_layer + 1} from layer {current_layer}...")
    
    # Get all chunks at current layer
    chunk_ids, texts, metadatas, embeddings = get_layer_chunks(
        document_id, current_layer, collection_name
    )
    
    if not chunk_ids:
        print(f"   No chunks found at layer {current_layer}")
        return 0, []
    
    print(f"   Found {len(chunk_ids)} chunks at layer {current_layer}")
    
    # Check if we have enough nodes to cluster
    if len(chunk_ids) < MIN_NODES_FOR_CLUSTERING:
        print(f"   Too few chunks ({len(chunk_ids)}) for clustering. Stopping.")
        return 0, []
    
    # Get embeddings (fetch from ChromaDB if not returned)
    if embeddings is None or len(embeddings) == 0:
        print(f"   Fetching embeddings...")
        collection = get_or_create_collection(collection_name)
        results = collection.get(ids=chunk_ids, include=["embeddings"])
        embeddings = results["embeddings"]
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Cluster the chunks using layer-aware adaptive clustering
    clusters, probabilities = cluster_adaptive(
        embeddings_array, 
        layer=current_layer,
        method=CLUSTERING_METHOD
    )
    
    print(f"   Created {len(clusters)} clusters")
    
    # Generate summaries for each cluster
    summaries = []
    for cluster_idx, cluster_indices in enumerate(tqdm(clusters, desc="   Summarizing clusters", unit="cluster")):
        if len(cluster_indices) < MIN_CLUSTER_SIZE:
            continue
            
        # Get texts for this cluster
        cluster_texts = [texts[i] for i in cluster_indices]
        cluster_chunk_ids = [chunk_ids[i] for i in cluster_indices]
        
        # Generate summary
        summary_text = summarize_cluster(cluster_texts, current_layer)
        
        summaries.append({
            "text": summary_text,
            "child_ids": cluster_chunk_ids,
            "cluster_id": f"L{current_layer + 1}_C{cluster_idx}"
        })
    
    if not summaries:
        print(f"   No valid clusters formed")
        return 0, []
    
    # Store summaries at next layer
    summary_ids = store_summary_chunks(
        summaries, document_id, current_layer + 1, collection_name
    )
    
    print(f"   ✓ Created {len(summary_ids)} summaries at layer {current_layer + 1}")
    
    return len(summary_ids), summary_ids


def build_raptor_tree(
    document_id: str,
    collection_name: str = "raptor_chunks",
    max_depth: int = MAX_TREE_DEPTH
) -> Dict[str, any]:
    """
    Build complete RAPTOR tree for a document
    
    Starting from layer 0 (base chunks), recursively clusters and summarizes
    until reaching max depth or too few nodes to cluster.
    
    Args:
        document_id: Document identifier
        collection_name: ChromaDB collection name
        max_depth: Maximum tree depth
        
    Returns:
        Dict with tree statistics
    """
    print("=" * 60)
    print(f"🌲 BUILDING RAPTOR TREE: {document_id}")
    print("=" * 60)
    
    stats = {
        "document_id": document_id,
        "layers": {},
        "total_nodes": 0,
        "tree_depth": 0
    }
    
    # Check layer 0 exists
    chunk_ids, _, _, _ = get_layer_chunks(document_id, 0, collection_name)
    if not chunk_ids:
        print(f"❌ No base chunks found for document: {document_id}")
        print(f"   Upload the document first with: python main.py upload <file>")
        return stats
    
    stats["layers"][0] = len(chunk_ids)
    stats["total_nodes"] = len(chunk_ids)
    
    print(f"📊 Starting with {len(chunk_ids)} base chunks (layer 0)")
    
    # Warn if below recommended minimum for UMAP
    if len(chunk_ids) < UMAP_RECOMMENDED_MIN_SAMPLES:
        print(f"\n⚠️  Note: Only {len(chunk_ids)} chunks (recommended: {UMAP_RECOMMENDED_MIN_SAMPLES}+)")
        print(f"   UMAP may fall back to PCA for dimensionality reduction.")
        print(f"   Tree will still be built, but clustering may be less optimal.\n")
    
    # Build layers recursively
    current_layer = 0
    while current_layer < max_depth:
        num_summaries, summary_ids = build_layer(
            document_id, current_layer, collection_name
        )
        
        if num_summaries == 0:
            break
        
        current_layer += 1
        stats["layers"][current_layer] = num_summaries
        stats["total_nodes"] += num_summaries
        
        # Stop if we've condensed to a single summary (root)
        if num_summaries <= MIN_CLUSTER_SIZE:
            print(f"   Reached root level with {num_summaries} node(s)")
            break
    
    stats["tree_depth"] = current_layer + 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ RAPTOR TREE COMPLETE")
    print("=" * 60)
    print(f"Document: {document_id}")
    print(f"Tree depth: {stats['tree_depth']} layers")
    print(f"Total nodes: {stats['total_nodes']}")
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
    """
    Delete summary layers (layer >= min_layer) for a document
    Useful for rebuilding the tree without re-uploading base chunks
    
    Args:
        document_id: Document identifier
        min_layer: Minimum layer to delete (default 1 = keep base chunks)
        collection_name: ChromaDB collection name
        
    Returns:
        Number of chunks deleted
    """
    collection = get_or_create_collection(collection_name)
    
    # Get all chunks for this document with layer >= min_layer
    results = collection.get(
        where={
            "$and": [
                {"document_id": {"$eq": document_id}},
                {"layer": {"$gte": min_layer}}
            ]
        }
    )
    
    if not results["ids"]:
        print(f"No summary layers found for {document_id}")
        return 0
    
    # Delete them
    collection.delete(ids=results["ids"])
    print(f"🗑️ Deleted {len(results['ids'])} summary chunks (layer >= {min_layer})")
    
    return len(results["ids"])


def rebuild_tree(
    document_id: str,
    collection_name: str = "raptor_chunks"
) -> Dict[str, any]:
    """
    Rebuild RAPTOR tree by deleting summaries and rebuilding
    
    Args:
        document_id: Document identifier
        collection_name: ChromaDB collection name
        
    Returns:
        Tree statistics
    """
    print(f"♻️ Rebuilding RAPTOR tree for: {document_id}")
    
    # Delete existing summary layers
    delete_tree_layers(document_id, min_layer=1, collection_name=collection_name)
    
    # Rebuild from base chunks
    return build_raptor_tree(document_id, collection_name)


def get_tree_stats(
    document_id: str,
    collection_name: str = "raptor_chunks"
) -> Dict[str, any]:
    """
    Get statistics about the RAPTOR tree for a document
    
    Args:
        document_id: Document identifier
        collection_name: ChromaDB collection name
        
    Returns:
        Dict with tree statistics
    """
    collection = get_or_create_collection(collection_name)
    
    # Get all chunks for this document
    results = collection.get(
        where={"document_id": {"$eq": document_id}},
        include=["metadatas"]
    )
    
    if not results["ids"]:
        return {"document_id": document_id, "exists": False}
    
    # Count by layer
    layers = {}
    for meta in results["metadatas"]:
        layer = meta.get("layer", 0)
        layers[layer] = layers.get(layer, 0) + 1
    
    return {
        "document_id": document_id,
        "exists": True,
        "total_nodes": len(results["ids"]),
        "tree_depth": max(layers.keys()) + 1 if layers else 0,
        "layers": layers
    }


# ============================================================================
# TEST / CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("RAPTOR Tree Builder")
        print("\nUsage:")
        print("  python raptor.py build <document_id>    Build tree for document")
        print("  python raptor.py rebuild <document_id>  Rebuild tree (delete + build)")
        print("  python raptor.py stats <document_id>    Show tree statistics")
        print("  python raptor.py delete <document_id>   Delete summary layers")
        print("\nNote: Document must be uploaded first with `python main.py upload <file>`")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "build" and len(sys.argv) >= 3:
        doc_id = sys.argv[2]
        build_raptor_tree(doc_id)
        
    elif command == "rebuild" and len(sys.argv) >= 3:
        doc_id = sys.argv[2]
        rebuild_tree(doc_id)
        
    elif command == "stats" and len(sys.argv) >= 3:
        doc_id = sys.argv[2]
        stats = get_tree_stats(doc_id)
        if stats.get("exists"):
            print(f"\n📊 RAPTOR Tree Stats: {doc_id}")
            print(f"   Total nodes: {stats['total_nodes']}")
            print(f"   Tree depth: {stats['tree_depth']} layers")
            for layer, count in sorted(stats["layers"].items()):
                print(f"   Layer {layer}: {count} nodes")
        else:
            print(f"❌ No tree found for: {doc_id}")
            
    elif command == "delete" and len(sys.argv) >= 3:
        doc_id = sys.argv[2]
        delete_tree_layers(doc_id)
        
    else:
        print(f"Unknown command: {command}")
        print("Run without arguments for help")
