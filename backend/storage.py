"""
ChromaDB storage for RAPTOR chunks with image/table metadata tracking
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
import config
from embeddings import get_embeddings
from utils import count_tokens, extract_content_references
from tqdm import tqdm


# Initialize ChromaDB client
client = chromadb.PersistentClient(
    path=config.CHROMA_PERSIST_DIR,
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)


def get_or_create_collection(collection_name: str = "raptor_chunks"):
    """
    Get or create a ChromaDB collection for storing chunks
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        ChromaDB collection object
    """
    print(f"📦 Initializing collection: {collection_name}")
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": "RAPTOR hierarchical chunks with image/table metadata",
            "embedding_model": "text-embedding-004"
        }
    )
    
    print(f"   Collection '{collection_name}' ready ({collection.count()} existing chunks)")
    return collection


def store_chunks(
    chunks: List[str],
    document_id: str,
    collection_name: str = "raptor_chunks",
    layer: int = 0,
    parent_id: Optional[str] = None
) -> List[str]:
    """
    Store text chunks in ChromaDB with metadata
    
    Args:
        chunks: List of text chunks
        document_id: Identifier for source document
        collection_name: ChromaDB collection name
        layer: RAPTOR layer (0 = base chunks, 1+ = summaries)
        parent_id: Parent chunk ID for RAPTOR tree structure
        
    Returns:
        List of chunk IDs
    """
    print(f"\n💾 Storing {len(chunks)} chunks in ChromaDB...")
    print(f"   Document: {document_id}")
    print(f"   Layer: {layer}")
    
    collection = get_or_create_collection(collection_name)
    
    # Prepare data for batch insert
    chunk_ids = []
    chunk_texts = []
    chunk_metadatas = []
    
    for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunk")):
        # Generate unique chunk ID
        chunk_id = f"{document_id}_L{layer}_chunk{idx}"
        
        # Extract content metadata using shared utility
        content_meta = extract_content_references(chunk)
        
        # Build metadata
        metadata = {
            "document_id": document_id,
            "chunk_index": idx,
            "layer": layer,
            "token_count": count_tokens(chunk),
            "content_types": ",".join(content_meta["content_types"]),  # Store as comma-separated string
            "has_images": content_meta["has_images"],
            "has_tables": content_meta["has_tables"],
        }
        
        # Add image/table references if present
        if content_meta["image_refs"]:
            metadata["image_refs"] = ",".join(content_meta["image_refs"])
        if content_meta["table_refs"]:
            metadata["table_refs"] = ",".join(content_meta["table_refs"])
        
        # Add parent reference for RAPTOR tree
        if parent_id:
            metadata["parent_id"] = parent_id
        
        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk)
        chunk_metadatas.append(metadata)
    
    # Generate embeddings
    print(f"   Generating embeddings for {len(chunks)} chunks...")
    embeddings = get_embeddings(chunk_texts)
    
    # Store in ChromaDB
    print(f"   Storing in ChromaDB...")
    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=chunk_metadatas
    )
    
    print(f"   ✓ Stored {len(chunk_ids)} chunks successfully")
    return chunk_ids


def store_contextualized_chunks(
    chunks: List[Dict],
    document_id: str,
    doc_summary: str = "",
    collection_name: str = "raptor_chunks",
    layer: int = 0
) -> List[str]:
    """
    Store contextualized chunks in ChromaDB (Anthropic's Contextual Retrieval)
    
    Key difference from store_chunks:
    - Embeds the CONTEXTUALIZED version (better retrieval)
    - Stores the ORIGINAL version (for display to user)
    - Tracks contextual metadata
    
    Args:
        chunks: List of dicts with 'original', 'contextualized', 'context', 'chunk_index'
        document_id: Identifier for source document
        doc_summary: Document summary for metadata
        collection_name: ChromaDB collection name
        layer: RAPTOR layer (0 = base chunks, 1+ = summaries)
        
    Returns:
        List of chunk IDs
    """
    print(f"\n💾 Storing {len(chunks)} contextualized chunks in ChromaDB...")
    print(f"   Document: {document_id}")
    print(f"   Layer: {layer}")
    print(f"   Using: Contextual Embeddings (Anthropic's method)")
    
    collection = get_or_create_collection(collection_name)
    
    # Prepare data for batch insert
    chunk_ids = []
    chunk_texts = []  # Original text for storage
    contextual_texts = []  # Contextualized text for embedding
    chunk_metadatas = []
    
    for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
        idx = chunk.get("chunk_index", 0)
        original = chunk.get("original", "")
        contextualized = chunk.get("contextualized", original)
        context = chunk.get("context", "")
        content_type = chunk.get("content_type", "text")
        
        # Generate unique chunk ID
        chunk_id = f"{document_id}_L{layer}_chunk{idx}"
        
        # Extract content metadata using shared utility
        content_meta = extract_content_references(original)
        
        # Build metadata
        metadata = {
            "document_id": document_id,
            "chunk_index": idx,
            "layer": layer,
            "token_count": count_tokens(original),
            "contextual_token_count": count_tokens(contextualized),
            "content_type": content_type,
            "content_types": ",".join(content_meta["content_types"]),
            "has_images": content_meta["has_images"],
            "has_tables": content_meta["has_tables"],
            "has_context": bool(context),
            "doc_summary": doc_summary[:200] if doc_summary else "",
        }
        
        # Add image/table references if present
        if content_meta["image_refs"]:
            metadata["image_refs"] = ",".join(content_meta["image_refs"])
        if content_meta["table_refs"]:
            metadata["table_refs"] = ",".join(content_meta["table_refs"])
        if chunk.get("ref_id"):
            metadata["ref_id"] = chunk["ref_id"]
        
        chunk_ids.append(chunk_id)
        chunk_texts.append(original)  # Store original for display
        contextual_texts.append(contextualized)  # Embed contextual version
        chunk_metadatas.append(metadata)
    
    # Generate embeddings from CONTEXTUALIZED versions (key for retrieval improvement)
    print(f"   Generating embeddings from contextualized text...")
    embeddings = get_embeddings(contextual_texts)
    
    # Store in ChromaDB (original text, but contextualized embeddings)
    print(f"   Storing in ChromaDB...")
    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,  # Store original text
        embeddings=embeddings,  # Use contextualized embeddings
        metadatas=chunk_metadatas
    )
    
    print(f"   ✓ Stored {len(chunk_ids)} chunks with contextual embeddings")
    return chunk_ids


def query_chunks(
    query_text: str,
    collection_name: str = "raptor_chunks",
    n_results: int = 5,
    filter_layer: Optional[int] = None,
    filter_content_type: Optional[str] = None
) -> Dict[str, any]:
    """
    Query ChromaDB for relevant chunks
    
    NOTE: This is a helper function for query.py
    Use query.py for actual querying, this is just for storage operations
    
    Args:
        query_text: Query text
        collection_name: ChromaDB collection name
        n_results: Number of results to return
        filter_layer: Filter by RAPTOR layer (None = all layers)
        filter_content_type: Filter by content type ("image", "table", None = all)
        
    Returns:
        Dict with ids, documents, distances, metadatas
    """
    collection = get_or_create_collection(collection_name)
    
    # Build where filter
    where = {}
    if filter_layer is not None:
        where["layer"] = filter_layer
    if filter_content_type == "image":
        where["has_images"] = True
    elif filter_content_type == "table":
        where["has_tables"] = True
    
    # Generate query embedding
    query_embedding = get_embeddings([query_text])[0]
    
    # Query collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where if where else None
    )
    
    return results


def get_chunk_by_id(chunk_id: str, collection_name: str = "raptor_chunks") -> Optional[Dict]:
    """
    Retrieve a specific chunk by ID
    
    Args:
        chunk_id: Chunk identifier
        collection_name: ChromaDB collection name
        
    Returns:
        Dict with chunk data or None if not found
    """
    collection = get_or_create_collection(collection_name)
    
    result = collection.get(
        ids=[chunk_id],
        include=["documents", "metadatas", "embeddings"]
    )
    
    if result["ids"]:
        return {
            "id": result["ids"][0],
            "document": result["documents"][0],
            "metadata": result["metadatas"][0],
            "embedding": result["embeddings"][0] if result["embeddings"] else None
        }
    return None


def delete_document_chunks(document_id: str, collection_name: str = "raptor_chunks"):
    """
    Delete all chunks for a specific document
    
    Args:
        document_id: Document identifier
        collection_name: ChromaDB collection name
    """
    collection = get_or_create_collection(collection_name)
    
    # Get all chunks for this document
    results = collection.get(
        where={"document_id": document_id}
    )
    
    if results["ids"]:
        print(f"🗑️  Deleting {len(results['ids'])} chunks for document: {document_id}")
        collection.delete(ids=results["ids"])
        print(f"   ✓ Deleted successfully")
    else:
        print(f"   No chunks found for document: {document_id}")


def get_collection_stats(collection_name: str = "raptor_chunks") -> Dict[str, any]:
    """
    Get statistics about the collection
    
    Args:
        collection_name: ChromaDB collection name
        
    Returns:
        Dict with collection statistics
    """
    collection = get_or_create_collection(collection_name)
    
    # Get all chunks
    all_chunks = collection.get(include=["metadatas"])
    
    if not all_chunks["ids"]:
        return {
            "total_chunks": 0,
            "documents": [],
            "layers": [],
            "content_types": {}
        }
    
    # Extract statistics
    documents = set()
    layers = set()
    content_type_counts = {"text": 0, "image": 0, "table": 0}
    
    for metadata in all_chunks["metadatas"]:
        documents.add(metadata.get("document_id"))
        layers.add(metadata.get("layer"))
        
        if metadata.get("has_images"):
            content_type_counts["image"] += 1
        if metadata.get("has_tables"):
            content_type_counts["table"] += 1
        content_type_counts["text"] += 1  # All chunks have text
    
    return {
        "total_chunks": len(all_chunks["ids"]),
        "documents": sorted(list(documents)),
        "layers": sorted(list(layers)),
        "content_types": content_type_counts
    }


# CLI interface for viewing database contents
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("ChromaDB Viewer - See what's in your database\n")
        print("Usage:")
        print("  python storage.py stats          - Show database statistics")
        print("  python storage.py list           - List all documents")
        print("  python storage.py view <doc_id>  - View all chunks from a document")
        print("  python storage.py delete <doc_id> - Delete a document")
        print("\nExamples:")
        print("  python storage.py stats")
        print("  python storage.py view PHYS11320_SensorCatalog")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "stats":
        print("📊 ChromaDB Statistics\n")
        stats = get_collection_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Documents: {len(stats['documents'])}")
        print(f"Layers: {stats['layers']}")
        
        if stats['total_chunks'] == 0:
            print("\n⚠️  Database is empty. Upload a document to see content breakdown.")
        else:
            print(f"\nContent breakdown:")
            print(f"  Text chunks: {stats['content_types'].get('text', 0)}")
            print(f"  With images: {stats['content_types'].get('image', 0)}")
            print(f"  With tables: {stats['content_types'].get('table', 0)}")
            
            if stats['documents']:
                print(f"\nStored documents:")
                for doc in stats['documents']:
                    # Count chunks per document
                    collection = get_or_create_collection()
                    doc_results = collection.get(where={"document_id": doc})
                    print(f"  • {doc} ({len(doc_results['ids'])} chunks)")
    
    elif command == "list":
        print("📚 All Documents\n")
        stats = get_collection_stats()
        if not stats['documents']:
            print("Database is empty. Upload a document with:")
            print("  python document_parser.py <file.pdf>")
        else:
            for i, doc in enumerate(stats['documents'], 1):
                collection = get_or_create_collection()
                doc_results = collection.get(where={"document_id": doc})
                print(f"{i}. {doc} ({len(doc_results['ids'])} chunks)")
    
    elif command == "view":
        if len(sys.argv) < 3:
            print("❌ Error: Please specify document ID")
            print("Usage: python storage.py view <doc_id>")
            print("\nAvailable documents:")
            stats = get_collection_stats()
            for doc in stats['documents']:
                print(f"  - {doc}")
            sys.exit(1)
        
        doc_id = sys.argv[2]
        collection = get_or_create_collection()
        
        results = collection.get(
            where={"document_id": doc_id},
            include=["documents", "metadatas"]
        )
        
        if not results["ids"]:
            print(f"❌ No chunks found for: {doc_id}")
            print("\nAvailable documents:")
            stats = get_collection_stats()
            for doc in stats['documents']:
                print(f"  - {doc}")
        else:
            print(f"📄 Document: {doc_id}")
            print(f"Total chunks: {len(results['ids'])}")
            print("="*70 + "\n")
            
            for i, (chunk_id, doc, meta) in enumerate(zip(results["ids"], results["documents"], results["metadatas"]), 1):
                print(f"[Chunk {i}/{len(results['ids'])}] {chunk_id}")
                print(f"  Layer: {meta.get('layer', 0)} | Tokens: {meta.get('token_count', 'N/A')} | Content: {meta.get('content_types', 'text')}")
                
                if meta.get('image_refs'):
                    print(f"  🖼️  Images: {meta['image_refs']}")
                if meta.get('table_refs'):
                    print(f"  📊 Tables: {meta['table_refs']}")
                
                print(f"\n  Text:")
                # Show first 250 chars
                preview = doc[:250].replace('\n', '\n  ')
                print(f"  {preview}")
                if len(doc) > 250:
                    print(f"  ... ({len(doc) - 250} more characters)")
                print("\n" + "-"*70 + "\n")
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("❌ Error: Please specify document ID")
            print("Usage: python storage.py delete <doc_id>")
            sys.exit(1)
        
        doc_id = sys.argv[2]
        collection = get_or_create_collection()
        check = collection.get(where={"document_id": doc_id})
        
        if not check["ids"]:
            print(f"❌ Document not found: {doc_id}")
        else:
            print(f"⚠️  This will delete {len(check['ids'])} chunks from '{doc_id}'")
            confirm = input("Type 'yes' to confirm: ")
            
            if confirm.lower() == "yes":
                delete_document_chunks(doc_id)
                print(f"✓ Deleted '{doc_id}'")
            else:
                print("Cancelled.")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python storage.py' for help.")
