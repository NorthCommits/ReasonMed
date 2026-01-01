"""
Test script for the RAG pipeline.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore


def test_rag():
    """Test the RAG pipeline with sample queries."""
    print("=" * 60)
    print("CliniScribe AI - RAG Pipeline Test")
    print("=" * 60)
    
    vector_store = VectorStore()
    doc_count = vector_store.get_collection_count()
    
    if doc_count == 0:
        print("\n⚠️  Vector store is empty!")
        print("Please run the setup script first:")
        print("  python scripts/setup_vectorstore.py")
        return
    
    print(f"\n✓ Vector store contains {doc_count} documents")
    
    pipeline = RAGPipeline(model_name="gpt-3.5-turbo", top_k=3)
    
    test_queries = [
        "65-year-old female presents with persistent cough, weight loss, and fatigue",
        "45-year-old male with chest pain and shortness of breath",
        "30-year-old patient with headache, fever, and neck stiffness"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 60)
        print(f"Test Query {i}:")
        print("=" * 60)
        print(f"Query: {query}\n")
        
        try:
            result = pipeline.run(query)
            
            print(f"Retrieved {len(result['retrieved_documents'])} similar cases:\n")
            
            for j, doc in enumerate(result["retrieved_documents"], 1):
                print(f"  {j}. Similarity: {doc['similarity_score']:.3f}")
                print(f"     Question: {doc['metadata'].get('full_question', 'N/A')[:100]}...")
            
            print("\nGenerated Response:")
            print("-" * 60)
            print(result["response"])
            print("-" * 60)
        
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_rag()

