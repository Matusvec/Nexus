#!/usr/bin/env python
"""
Nexus CLI - Central command-line interface for document processing

This is the main entry point that orchestrates all modules:
- document_parser: Extract text, images, tables from documents
- chunking: Semantic text chunking with contextual retrieval
- storage: ChromaDB vector storage
- raptor: Hierarchical tree building with UMAP + GMM clustering
- query: Collapsed tree retrieval for comprehensive Q&A

Usage:
    python main.py upload <file>          Upload and process a document
    python main.py build-tree -d <doc>    Build RAPTOR tree for document
    python main.py query <question>       Query the knowledge base
    python main.py stats                  Show database statistics
    python main.py list                   List all documents
    python main.py view <doc_id>          View document chunks
    python main.py delete <doc_id>        Delete a document
"""
import sys
import argparse
from pathlib import Path


def cmd_upload(args):
    """Upload and process a document through the full pipeline with contextual retrieval"""
    from document_parser import parse_document
    from chunking import chunk_text, contextualize_chunks
    from gemini_client import generate_document_summary
    from storage import store_contextualized_chunks, get_collection_stats
    
    file_path = args.file
    
    print("=" * 60)
    print("📥 NEXUS DOCUMENT UPLOAD (Contextual Retrieval)")
    print("=" * 60)
    
    # Step 1: Parse document
    print("\n[STEP 1/4] Parsing document...")
    result = parse_document(file_path)
    
    print(f"\n✓ Parsing complete:")
    print(f"  - Text: {len(result['text'])} characters")
    print(f"  - Images: {result['images_found']}")
    print(f"  - Tables: {result['tables_found']}")
    
    # Step 2: Generate document summary (for contextual retrieval)
    print("\n[STEP 2/4] Generating document summary...")
    doc_summary = generate_document_summary(result['text'], result['metadata']['filename'])
    print(f"  Summary: {doc_summary[:100]}...")
    
    # Step 3: Chunk text with overlap
    print("\n[STEP 3/4] Chunking text (with overlap)...")
    raw_chunks = chunk_text(
        result['text'],
        similarity_threshold=args.similarity_threshold,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        group_size=args.group_size,
        overlap_tokens=args.overlap_tokens
    )
    
    print(f"  Created {len(raw_chunks)} semantic chunks with {args.overlap_tokens}-token overlap")
    
    # Step 4: Add contextual information to each chunk (Anthropic's method)
    print("\n[STEP 4/5] Adding contextual embeddings...")
    contextualized = contextualize_chunks(
        raw_chunks, 
        doc_summary, 
        result['metadata']['filename'],
        use_llm_context=args.llm_context
    )
    
    # Step 5: Store in ChromaDB
    print("\n[STEP 5/5] Storing in ChromaDB...")
    document_id = Path(file_path).stem
    
    chunk_ids = store_contextualized_chunks(
        chunks=contextualized,
        document_id=document_id,
        doc_summary=doc_summary,
        collection_name="raptor_chunks",
        layer=0
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ UPLOAD COMPLETE")
    print("=" * 60)
    print(f"Document: {result['metadata']['filename']}")
    print(f"Chunks stored: {len(chunk_ids)}")
    
    # Warn if too few chunks for optimal tree building
    from t_retriever import MIN_NODES_FOR_CLUSTERING
    if len(chunk_ids) < 10:
        print(f"\n⚠️  WARNING: Only {len(chunk_ids)} chunks created.")
        print(f"   T-Retriever tree building works best with 10+ chunks.")
        print(f"   With fewer chunks, clustering may be less optimal.")
        print(f"   Consider uploading a longer document or lowering --max-tokens.")
    print(f"Features used:")
    print(f"  ✓ Semantic chunking with {args.overlap_tokens}-token overlap")
    print(f"  ✓ Contextual embeddings (Anthropic's method)")
    if args.llm_context:
        print(f"  ✓ LLM-generated chunk context")
    
    stats = get_collection_stats()
    print(f"\n📊 Database Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Documents: {len(stats['documents'])}")
    print(f"  Chunks with images: {stats['content_types'].get('image', 0)}")
    print(f"  Chunks with tables: {stats['content_types'].get('table', 0)}")
    print("=" * 60)


def cmd_query(args):
    """Query the knowledge base using T-Retriever hybrid retrieval"""
    from t_query import answer_question, collapsed_tree_retrieval
    from storage import get_collection_stats
    
    question = args.question
    doc_id = args.document if hasattr(args, 'document') and args.document else None
    top_k = args.top_k
    
    print("=" * 60)
    print("🔍 NEXUS QUERY (T-Retriever Hybrid)")
    print("=" * 60)
    print(f"Question: {question}")
    if doc_id:
        print(f"Document filter: {doc_id}")
    print()
    
    # Check if database has content
    stats = get_collection_stats()
    if stats['total_chunks'] == 0:
        print("❌ Database is empty. Upload a document first:")
        print("   python main.py upload <file.pdf>")
        return
    
    print("⏳ Searching and generating answer...\n")
    
    result = answer_question(
        question=question,
        document_id=doc_id,
        top_k=top_k,
        show_sources=True
    )
    
    print("=" * 60)
    print("💡 ANSWER")
    print("=" * 60)
    print(result["answer"])
    
    if result["sources"] and args.show_sources:
        print("\n" + "-" * 60)
        print("📚 SOURCES")
        print("-" * 60)
        for i, source in enumerate(result["sources"][:5], 1):
            method = source.get('method', 'tree')
            print(f"\n{i}. {source['type']} - {source['id']} (via {method})")
            print(f"   {source['preview'][:150]}...")
    
    print("\n" + "=" * 60)
    

def cmd_stats(args):
    """Show database statistics"""
    from storage import get_collection_stats, get_or_create_collection
    
    print("📊 Nexus Database Statistics\n")
    stats = get_collection_stats()
    
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Documents: {len(stats['documents'])}")
    print(f"Layers: {stats['layers']}")
    
    if stats['total_chunks'] == 0:
        print("\n⚠️  Database is empty. Upload a document with:")
        print("   python main.py upload <file.pdf>")
    else:
        print(f"\nContent breakdown:")
        print(f"  Text chunks: {stats['content_types'].get('text', 0)}")
        print(f"  With images: {stats['content_types'].get('image', 0)}")
        print(f"  With tables: {stats['content_types'].get('table', 0)}")
        
        if stats['documents']:
            print(f"\nStored documents:")
            collection = get_or_create_collection()
            for doc in stats['documents']:
                doc_results = collection.get(where={"document_id": doc})
                print(f"  • {doc} ({len(doc_results['ids'])} chunks)")


def cmd_list(args):
    """List all documents"""
    from storage import get_collection_stats, get_or_create_collection
    
    print("📚 All Documents\n")
    stats = get_collection_stats()
    
    if not stats['documents']:
        print("Database is empty. Upload a document with:")
        print("  python main.py upload <file.pdf>")
    else:
        collection = get_or_create_collection()
        for i, doc in enumerate(stats['documents'], 1):
            doc_results = collection.get(where={"document_id": doc})
            print(f"{i}. {doc} ({len(doc_results['ids'])} chunks)")


def cmd_delete(args):
    """Delete a document from the database"""
    from storage import delete_document_chunks, get_or_create_collection
    
    doc_id = args.doc_id
    collection = get_or_create_collection()
    check = collection.get(where={"document_id": doc_id})
    
    if not check["ids"]:
        print(f"❌ Document not found: {doc_id}")
        return
    
    if not args.force:
        print(f"⚠️  This will delete {len(check['ids'])} chunks from '{doc_id}'")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            return
    
    delete_document_chunks(doc_id)
    print(f"✓ Deleted '{doc_id}'")


def cmd_view(args):
    """View chunks from a document"""
    from storage import get_or_create_collection, get_collection_stats
    
    doc_id = args.doc_id
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
        return
    
    print(f"📄 Document: {doc_id}")
    print(f"Total chunks: {len(results['ids'])}")
    print("=" * 70 + "\n")
    
    limit = args.limit or len(results['ids'])
    for i, (chunk_id, doc, meta) in enumerate(zip(
        results["ids"][:limit], 
        results["documents"][:limit], 
        results["metadatas"][:limit]
    ), 1):
        print(f"[Chunk {i}/{len(results['ids'])}] {chunk_id}")
        print(f"  Layer: {meta.get('layer', 0)} | Tokens: {meta.get('token_count', 'N/A')} | Content: {meta.get('content_types', 'text')}")
        
        if meta.get('image_refs'):
            print(f"  🖼️  Images: {meta['image_refs']}")
        if meta.get('table_refs'):
            print(f"  📊 Tables: {meta['table_refs']}")
        
        print(f"\n  Text:")
        preview = doc[:250].replace('\n', '\n  ')
        print(f"  {preview}")
        if len(doc) > 250:
            print(f"  ... ({len(doc) - 250} more characters)")
        print("\n" + "-" * 70 + "\n")


def cmd_build_tree(args):
    """Build T-Retriever hierarchical tree for a document"""
    from t_retriever import build_tretriever_tree, rebuild_tree, get_tree_stats
    from storage import get_collection_stats
    
    doc_id = args.document if hasattr(args, 'document') and args.document else None
    
    print("=" * 60)
    print("🌲 T-RETRIEVER TREE BUILDER")
    print("=" * 60)
    
    # Get available documents
    stats = get_collection_stats()
    
    if stats['total_chunks'] == 0:
        print("❌ Database is empty. Upload a document first:")
        print("   python main.py upload <file.pdf>")
        return
    
    # If no document specified, show list and prompt
    if not doc_id:
        if len(stats['documents']) == 1:
            doc_id = stats['documents'][0]
            print(f"📄 Using document: {doc_id}\n")
        else:
            print("Available documents:")
            for i, doc in enumerate(stats['documents'], 1):
                # Check if tree exists
                tree_stats = get_tree_stats(doc)
                tree_info = f"[Tree: {tree_stats['tree_depth']} layers, {tree_stats.get('unique_entities', 0)} entities]" if tree_stats.get('tree_depth', 0) > 1 else "[No tree]"
                print(f"  {i}. {doc} {tree_info}")
            print("\nSpecify document with: python main.py build-tree -d <document_id>")
            return
    
    # Check if document exists
    if doc_id not in stats['documents']:
        print(f"❌ Document not found: {doc_id}")
        print("\nAvailable documents:")
        for doc in stats['documents']:
            print(f"  - {doc}")
        return
    
    # Check if tree already exists
    existing_stats = get_tree_stats(doc_id)
    if existing_stats.get('tree_depth', 0) > 1:
        print(f"⚠️  Tree already exists for '{doc_id}' ({existing_stats['tree_depth']} layers, {existing_stats.get('unique_entities', 0)} entities)")
        if args.rebuild:
            print("   Rebuilding tree...\n")
            rebuild_tree(doc_id)
        else:
            print("   Use --rebuild to rebuild the tree")
            print(f"\nCurrent tree stats:")
            for layer, count in sorted(existing_stats['layers'].items()):
                layer_type = "base chunks" if layer == 0 else "summaries"
                print(f"  Layer {layer}: {count} {layer_type}")
            print(f"  Unique entities: {existing_stats.get('unique_entities', 0)}")
            print(f"  Graph edges: {existing_stats.get('graph_edges', 0)}")
        return
    
    # Build the tree
    build_tretriever_tree(doc_id, max_depth=args.max_layers)


def main():
    parser = argparse.ArgumentParser(
        description="Nexus - AI-powered document research workspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py upload research_paper.pdf
  python main.py query "What are the main findings?"
  python main.py stats
  python main.py list
  python main.py view my_document
  python main.py delete my_document
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload and process a document")
    upload_parser.add_argument("file", help="Path to PDF or DOCX file")
    upload_parser.add_argument("--similarity-threshold", type=float, default=0.7,
                               help="Cosine similarity threshold for chunking (default: 0.7)")
    upload_parser.add_argument("--min-tokens", type=int, default=100,
                               help="Minimum tokens per chunk (default: 100)")
    upload_parser.add_argument("--max-tokens", type=int, default=500,
                               help="Maximum tokens per chunk (default: 500)")
    upload_parser.add_argument("--group-size", type=int, default=2,
                               help="Sentences to group before embedding (default: 2)")
    upload_parser.add_argument("--overlap-tokens", type=int, default=50,
                               help="Token overlap between chunks (default: 50)")
    upload_parser.add_argument("--llm-context", action="store_true",
                               help="Use LLM to generate per-chunk context (slower but better)")
    upload_parser.set_defaults(func=cmd_upload)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("-d", "--document", help="Filter to specific document")
    query_parser.add_argument("--top-k", type=int, default=5,
                              help="Number of chunks to retrieve (default: 5)")
    query_parser.add_argument("--show-sources", action="store_true", default=True,
                              help="Show source references (default: True)")
    query_parser.add_argument("--no-sources", dest="show_sources", action="store_false",
                              help="Hide source references")
    query_parser.set_defaults(func=cmd_query)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all documents")
    list_parser.set_defaults(func=cmd_list)
    
    # View command
    view_parser = subparsers.add_parser("view", help="View chunks from a document")
    view_parser.add_argument("doc_id", help="Document ID to view")
    view_parser.add_argument("--limit", type=int, help="Limit number of chunks shown")
    view_parser.set_defaults(func=cmd_view)
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a document")
    delete_parser.add_argument("doc_id", help="Document ID to delete")
    delete_parser.add_argument("-f", "--force", action="store_true",
                               help="Skip confirmation prompt")
    delete_parser.set_defaults(func=cmd_delete)
    
    # Build tree command
    tree_parser = subparsers.add_parser("build-tree", help="Build T-Retriever hierarchical tree with entity graph")
    tree_parser.add_argument("-d", "--document", help="Document ID to build tree for")
    tree_parser.add_argument("--max-layers", type=int, default=3,
                             help="Maximum tree layers (default: 3)")
    tree_parser.add_argument("--rebuild", action="store_true",
                             help="Rebuild tree if it already exists")
    tree_parser.set_defaults(func=cmd_build_tree)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python main.py upload your_document.pdf")
        print("   python main.py stats")
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
