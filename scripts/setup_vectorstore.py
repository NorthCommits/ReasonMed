"""
One-time setup script to download dataset, generate embeddings, and populate vector store.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_pipeline import DataPipeline
from src.embeddings import EmbeddingGenerator
from src.vectorstore import VectorStore
import argparse


def setup_vectorstore(limit: int = None, batch_size: int = 100):
    """
    Set up the vector store with medical case data.
    
    Args:
        limit: Optional limit on number of records to process (None for all)
        batch_size: Batch size for embedding generation
    """
    print("=" * 60)
    print("ReasonMed - Vector Store Setup")
    print("=" * 60)
    
    print("\nStep 1: Loading dataset from HuggingFace...")
    data_pipeline = DataPipeline()
    processed_records = data_pipeline.process_all(limit=limit)
    print(f"✓ Processed {len(processed_records)} records")
    
    print("\nStep 2: Generating embeddings...")
    embedding_generator = EmbeddingGenerator()
    
    texts = [record["text"] for record in processed_records]
    embeddings = embedding_generator.generate_embeddings_batch(texts, batch_size=batch_size)
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    print("\nStep 3: Creating vector store and adding documents...")
    vector_store = VectorStore()
    
    ids = [record["question_id"] for record in processed_records]
    metadatas = [
        {
            "full_question": record["full_question"],
            "full_reasoning": record["full_reasoning"],
            "full_response": record["full_response"],
            "medical_keywords": record["medical_keywords"],
        }
        for record in processed_records
    ]
    
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    final_count = vector_store.get_collection_count()
    print(f"✓ Vector store setup complete!")
    print(f"✓ Total documents in vector store: {final_count}")
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up the vector store with medical case data")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of records to process (default: process all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation (default: 100)"
    )
    
    args = parser.parse_args()
    
    try:
        setup_vectorstore(limit=args.limit, batch_size=args.batch_size)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

