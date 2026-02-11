"""
TreeIndex — Membership index for incremental tree updates.

Maintains a mapping from every chunk/summary ID to its cluster and ancestor
chain, enabling O(depth) deletion instead of O(n) full-tree rebuild.

Concepts:
    membership:  chunk_id → { cluster_id, parent_id, layer }
    dirty_nodes: set of node IDs modified since last compaction
    tree_version: monotonically increasing integer

Persistence:
    Serialised as JSON in ChromaDB with layer = -2 and
    type = "tree_index_metadata".
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMPACTION_DIRTY_THRESHOLD = 10     # compact when this many nodes are dirty
COMPACTION_ENABLED = True           # toggle background compaction
MIN_CLUSTER_SIZE_AFTER_DELETE = 2   # merge if cluster shrinks below this

# ---------------------------------------------------------------------------
# TreeIndex
# ---------------------------------------------------------------------------

class TreeIndex:
    """In-memory index that tracks every node's position in the tree.

    Fields
    ------
    membership : dict[str, MemberEntry]
        chunk_id → {"cluster_id": str, "parent_id": str | None,
                     "layer": int, "children": list[str]}
    dirty_nodes : set[str]
        IDs that changed since last compaction.
    tree_version : int
        Incremented on every compaction swap.
    """

    def __init__(self) -> None:
        self.membership: Dict[str, Dict] = {}
        self.dirty_nodes: Set[str] = set()
        self.tree_version: int = 1
        self._tombstones: Set[str] = set()       # soft-deleted IDs

    # ------------------------------------------------------------------
    # Build / maintain
    # ------------------------------------------------------------------

    def register_node(
        self,
        node_id: str,
        *,
        layer: int,
        cluster_id: str = "",
        parent_id: str = "",
        children: Optional[List[str]] = None,
    ) -> None:
        """Register a node (chunk or summary) in the index."""
        self.membership[node_id] = {
            "cluster_id": cluster_id,
            "parent_id": parent_id,
            "layer": layer,
            "children": children or [],
        }

    def set_parent(self, node_id: str, parent_id: str) -> None:
        if node_id in self.membership:
            self.membership[node_id]["parent_id"] = parent_id

    # ------------------------------------------------------------------
    # Ancestor chain
    # ------------------------------------------------------------------

    def ancestor_chain(self, node_id: str) -> List[str]:
        """Return [parent, grandparent, …] up to the root."""
        chain: List[str] = []
        current = node_id
        visited: Set[str] = set()
        while current in self.membership:
            parent = self.membership[current].get("parent_id", "")
            if not parent or parent in visited:
                break
            chain.append(parent)
            visited.add(parent)
            current = parent
        return chain

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_child(self, parent_id: str, child_id: str) -> None:
        """Add *child_id* to *parent_id*'s children list."""
        if parent_id in self.membership:
            children = self.membership[parent_id].setdefault("children", [])
            if child_id not in children:
                children.append(child_id)

    # ------------------------------------------------------------------
    # Deletion helpers
    # ------------------------------------------------------------------

    def mark_tombstone(self, node_id: str) -> None:
        """Soft-delete a leaf node."""
        self._tombstones.add(node_id)
        self.dirty_nodes.add(node_id)

    def is_tombstoned(self, node_id: str) -> bool:
        return node_id in self._tombstones

    def get_cluster_siblings(self, node_id: str) -> List[str]:
        """Return other nodes in the same cluster (same parent)."""
        entry = self.membership.get(node_id)
        if not entry:
            return []
        parent_id = entry.get("parent_id", "")
        if not parent_id or parent_id not in self.membership:
            return []
        parent_children = self.membership[parent_id].get("children", [])
        return [c for c in parent_children
                if c != node_id and c not in self._tombstones]

    def remove_child_from_parent(self, node_id: str) -> Optional[str]:
        """Remove *node_id* from its parent's children list.

        Returns the parent_id (or None).
        """
        entry = self.membership.get(node_id)
        if not entry:
            return None
        parent_id = entry.get("parent_id", "")
        if parent_id and parent_id in self.membership:
            children = self.membership[parent_id].get("children", [])
            if node_id in children:
                children.remove(node_id)
            self.dirty_nodes.add(parent_id)
            return parent_id
        return None

    def find_nearest_sibling_cluster(
        self, parent_id: str, embeddings_lookup: Dict[str, List[float]]
    ) -> Optional[str]:
        """Find the nearest sibling summary node to *parent_id* at the same layer.

        Uses centroid cosine similarity via the provided embeddings lookup.
        Returns the sibling's ID or None.
        """
        entry = self.membership.get(parent_id)
        if not entry:
            return None
        layer = entry["layer"]
        grandparent = entry.get("parent_id", "")

        # Candidates: other nodes at the same layer under the same grandparent
        candidates: List[str] = []
        if grandparent and grandparent in self.membership:
            candidates = [
                c for c in self.membership[grandparent].get("children", [])
                if c != parent_id and c not in self._tombstones
            ]
        if not candidates:
            # Fallback: any node at the same layer
            candidates = [
                nid for nid, m in self.membership.items()
                if m["layer"] == layer
                and nid != parent_id
                and nid not in self._tombstones
            ]
        if not candidates:
            return None

        # Pick the one with the highest cosine similarity
        parent_emb = embeddings_lookup.get(parent_id)
        if parent_emb is None:
            return candidates[0]

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        parent_vec = np.array(parent_emb).reshape(1, -1)
        best_id: Optional[str] = None
        best_sim = -2.0
        for cid in candidates:
            c_emb = embeddings_lookup.get(cid)
            if c_emb is None:
                continue
            sim = float(cos_sim(parent_vec, np.array(c_emb).reshape(1, -1))[0, 0])
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    @property
    def needs_compaction(self) -> bool:
        return (
            COMPACTION_ENABLED
            and len(self.dirty_nodes) >= COMPACTION_DIRTY_THRESHOLD
        )

    def compact(self) -> int:
        """Clear dirty set and bump version.  Returns number of dirty nodes cleared."""
        n = len(self.dirty_nodes)
        # Purge tombstoned entries from membership entirely
        for tid in list(self._tombstones):
            self.membership.pop(tid, None)
        self._tombstones.clear()
        self.dirty_nodes.clear()
        self.tree_version += 1
        return n

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "membership": self.membership,
            "dirty_nodes": list(self.dirty_nodes),
            "tombstones": list(self._tombstones),
            "tree_version": self.tree_version,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TreeIndex":
        idx = cls()
        idx.membership = data.get("membership", {})
        idx.dirty_nodes = set(data.get("dirty_nodes", []))
        idx._tombstones = set(data.get("tombstones", []))
        idx.tree_version = data.get("tree_version", 1)
        return idx


# ---------------------------------------------------------------------------
# Per-document index cache (mirrors _document_graphs pattern)
# ---------------------------------------------------------------------------
_document_indices: Dict[str, TreeIndex] = {}


def get_document_index(document_id: str) -> TreeIndex:
    if document_id not in _document_indices:
        _document_indices[document_id] = TreeIndex()
    return _document_indices[document_id]


def save_document_index(
    document_id: str, collection_name: str = "nexus_chunks"
) -> None:
    """Persist index to ChromaDB as a special metadata chunk (layer = -2)."""
    from storage import get_or_create_collection

    idx = _document_indices.get(document_id)
    if idx is None:
        return

    collection = get_or_create_collection(collection_name)
    chunk_id = f"{document_id}_tree_index"
    payload = json.dumps(idx.to_dict())

    try:
        existing = collection.get(ids=[chunk_id])
        if existing["ids"]:
            collection.update(
                ids=[chunk_id],
                documents=[payload],
                metadatas=[{
                    "document_id": document_id,
                    "type": "tree_index_metadata",
                    "layer": -2,
                    "tree_version": idx.tree_version,
                }],
            )
        else:
            collection.add(
                ids=[chunk_id],
                documents=[payload],
                metadatas=[{
                    "document_id": document_id,
                    "type": "tree_index_metadata",
                    "layer": -2,
                    "tree_version": idx.tree_version,
                }],
                embeddings=[[0.0] * 768],
            )
    except Exception as exc:
        print(f"   [WARN] Failed to save tree index: {exc}")


def load_document_index(
    document_id: str, collection_name: str = "nexus_chunks"
) -> TreeIndex:
    """Load index from ChromaDB; create fresh if missing."""
    from storage import get_or_create_collection

    collection = get_or_create_collection(collection_name)
    chunk_id = f"{document_id}_tree_index"

    try:
        result = collection.get(ids=[chunk_id], include=["documents"])
        if result["ids"] and result["documents"]:
            data = json.loads(result["documents"][0])
            idx = TreeIndex.from_dict(data)
            _document_indices[document_id] = idx
            return idx
    except Exception:
        pass

    return get_document_index(document_id)
