"""
ChromaDB setup and basic database operations
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional


# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create/get collection
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)


def add_document(
    text: str,
    doc_id: str,
    group: str,
    subgroup: str = None,
    title: str = None,
    tags: List[str] = None,
    linked_to: List[str] = None,
    **extra_metadata
):
    """Add a document to the database with metadata"""
    metadata = {
        "group": group,
        "title": title or doc_id,
    }
    
    if subgroup:
        metadata["subgroup"] = subgroup
    if tags:
        metadata["tags"] = ",".join(tags)
    if linked_to:
        metadata["linked_to"] = ",".join(linked_to)
    
    metadata.update(extra_metadata)
    
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id]
    )
    return doc_id


def query_documents(
    query_text: str,
    n_results: int = 5,
    group: str = None,
    subgroup: str = None,
    include_groups: List[str] = None
):
    """Query documents with optional group/subgroup filtering"""
    where_filter = None
    
    if include_groups:
        where_filter = {"group": {"$in": include_groups}}
    elif group and subgroup:
        where_filter = {"$and": [{"group": group}, {"subgroup": subgroup}]}
    elif group:
        where_filter = {"group": group}
    elif subgroup:
        where_filter = {"subgroup": subgroup}
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter
    )
    
    return results


def get_document(doc_id: str):
    """Get a specific document by ID"""
    return collection.get(ids=[doc_id])


def update_document(doc_id: str, text: str = None, metadata: Dict = None):
    """Update a document's text or metadata"""
    if text:
        collection.update(ids=[doc_id], documents=[text])
    if metadata:
        collection.update(ids=[doc_id], metadatas=[metadata])


def delete_document(doc_id: str):
    """Delete a document by ID"""
    collection.delete(ids=[doc_id])


def list_groups():
    """List all unique groups"""
    all_docs = collection.get()
    groups = set()
    for metadata in all_docs['metadatas']:
        if 'group' in metadata:
            groups.add(metadata['group'])
    return sorted(list(groups))


def list_subgroups(group: str = None):
    """List all subgroups, optionally filtered by group"""
    all_docs = collection.get()
    subgroups = set()
    for metadata in all_docs['metadatas']:
        if group and metadata.get('group') != group:
            continue
        if 'subgroup' in metadata:
            subgroups.add(metadata['subgroup'])
    return sorted(list(subgroups))


def get_stats():
    """Get database statistics"""
    count = collection.count()
    groups = list_groups()
    return {
        "total_documents": count,
        "groups": groups,
        "num_groups": len(groups)
    }


def reset_collection():
    """Delete all documents in the collection"""
    client.delete_collection("documents")
    global collection
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )