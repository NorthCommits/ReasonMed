"""
Vector database operations using ChromaDB.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, collection_name: str = "medical_cases", 
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = None
        self._get_or_create_collection()
    
    def _get_or_create_collection(self) -> None:
        """Get existing collection or create a new one."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Medical case documentation and reasoning patterns"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries for each document
            ids: List of unique identifiers for each document
        """
        if not all(len(l) == len(texts) for l in [embeddings, metadatas, ids]):
            raise ValueError("All input lists must have the same length")
        
        print(f"Adding {len(texts)} documents to vector store...")
        
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            if (i + batch_size) % 1000 == 0 or i + batch_size >= len(texts):
                print(f"Added {min(i + batch_size, len(texts))}/{len(texts)} documents...")
        
        print(f"Successfully added {len(texts)} documents to vector store")
    
    def query(self, query_embedding: List[float], n_results: int = 5,
              filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            Dictionary containing ids, documents, metadatas, and distances
        """
        where_clause = filter_dict if filter_dict else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents in the collection
        """
        return self.collection.count()
    
    def delete_collection(self) -> None:
        """Delete the collection (use with caution)."""
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
        self._get_or_create_collection()


if __name__ == "__main__":
    store = VectorStore()
    print(f"Collection count: {store.get_collection_count()}")

