#!/usr/bin/env python3
"""
Vector Database Management Script
Utilities to manage the aviation vector database
"""

import os
import json
from pathlib import Path
from simple_rag import SimpleAviationRAG
import argparse

def check_database_status():
    """Check the status of the vector database."""
    print("🔍 Vector Database Status")
    print("=" * 40)
    
    vector_db_path = "aviation_vector_db.faiss"
    documents_path = "aviation_documents.json"
    
    # Check if files exist
    vector_exists = Path(vector_db_path).exists()
    docs_exists = Path(documents_path).exists()
    
    if vector_exists and docs_exists:
        print("✅ Vector database exists")
        
        # Get file sizes
        vector_size = Path(vector_db_path).stat().st_size / 1024  # KB
        docs_size = Path(documents_path).stat().st_size / 1024   # KB
        
        print(f"📁 Vector index: {vector_size:.1f} KB")
        print(f"📄 Documents: {docs_size:.1f} KB")
        
        # Get document count
        try:
            with open(documents_path, 'r') as f:
                documents = json.load(f)
            print(f"📊 Total documents: {len(documents)}")
            
            # Show source breakdown
            sources = {}
            for doc in documents:
                source = doc.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            print("\n📚 Documents by source:")
            for source, count in sorted(sources.items()):
                print(f"  - {source}: {count} chunks")
                
        except Exception as e:
            print(f"❌ Error reading documents: {e}")
            
        # Check file timestamps
        import time
        vector_time = os.path.getmtime(vector_db_path)
        docs_time = os.path.getmtime(documents_path)
        
        print(f"\n🕒 Last updated:")
        print(f"  - Vector DB: {time.ctime(vector_time)}")
        print(f"  - Documents: {time.ctime(docs_time)}")
        
    else:
        print("❌ Vector database not found")
        if not vector_exists:
            print("  - Missing: aviation_vector_db.faiss")
        if not docs_exists:
            print("  - Missing: aviation_documents.json")

def rebuild_database():
    """Rebuild the vector database from scratch."""
    print("🔄 Rebuilding Vector Database")
    print("=" * 40)
    
    try:
        rag = SimpleAviationRAG()
        rag.rebuild_database()
        print("✅ Database rebuilt successfully!")
        
    except Exception as e:
        print(f"❌ Failed to rebuild database: {e}")

def delete_database():
    """Delete the vector database files."""
    print("🗑️  Deleting Vector Database")
    print("=" * 40)
    
    vector_db_path = "aviation_vector_db.faiss"
    documents_path = "aviation_documents.json"
    
    deleted = []
    
    if Path(vector_db_path).exists():
        Path(vector_db_path).unlink()
        deleted.append("aviation_vector_db.faiss")
    
    if Path(documents_path).exists():
        Path(documents_path).unlink()
        deleted.append("aviation_documents.json")
    
    if deleted:
        print(f"✅ Deleted: {', '.join(deleted)}")
    else:
        print("ℹ️  No database files found to delete")

def test_query(query: str):
    """Test a query against the database."""
    print(f"🔍 Testing Query: '{query}'")
    print("=" * 50)
    
    try:
        rag = SimpleAviationRAG()
        result = rag.query(query)
        
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['num_sources']}")
        
        for i, source in enumerate(result['sources']):
            print(f"  {i+1}. {source['source']} (score: {source['similarity_score']:.3f})")
            
    except Exception as e:
        print(f"❌ Query failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage Aviation Vector Database")
    parser.add_argument("command", choices=["status", "rebuild", "delete", "test"], 
                       help="Command to execute")
    parser.add_argument("--query", type=str, help="Query to test (for test command)")
    
    args = parser.parse_args()
    
    if args.command == "status":
        check_database_status()
    elif args.command == "rebuild":
        rebuild_database()
    elif args.command == "delete":
        delete_database()
    elif args.command == "test":
        if args.query:
            test_query(args.query)
        else:
            test_query("What is ETOPS?")  # Default test query

if __name__ == "__main__":
    main()
