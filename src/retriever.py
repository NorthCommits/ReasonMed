"""
Retriever for similarity search in the vector store.
"""

from typing import List, Dict, Any, Optional
from src.embeddings import EmbeddingGenerator
from src.vectorstore import VectorStore


class Retriever:
    """Handles similarity search and retrieval of relevant medical cases."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None,
                 embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance (creates new if not provided)
            embedding_generator: EmbeddingGenerator instance (creates new if not provided)
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
    
    def retrieve(self, query: str, top_k: int = 5, 
                 filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar medical cases for a given query.
        
        Args:
            query: Query text (e.g., patient case description)
            top_k: Number of similar cases to retrieve
            filter_dict: Optional metadata filter for retrieval
            
        Returns:
            List of retrieved documents with metadata and similarity scores
        """
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            filter_dict=filter_dict
        )
        
        retrieved_docs = []
        for i in range(len(results["ids"])):
            doc = {
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
                "similarity_score": 1 - results["distances"][i]
            }
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc["metadata"]
            context_parts.append(f"Similar Case {i} (Similarity: {doc['similarity_score']:.3f}):")
            context_parts.append(f"Question: {metadata.get('full_question', 'N/A')}")
            context_parts.append(f"Reasoning: {metadata.get('full_reasoning', 'N/A')[:300]}...")
            context_parts.append(f"Diagnosis: {metadata.get('full_response', 'N/A')}")
            context_parts.append("")
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    retriever = Retriever()
    test_query = "Patient presents with chest pain and shortness of breath"
    results = retriever.retrieve(test_query, top_k=3)
    print(f"Retrieved {len(results)} documents")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['metadata'].get('full_question', 'N/A')[:100]}...")
        print(f"   Similarity: {doc['similarity_score']:.3f}")

