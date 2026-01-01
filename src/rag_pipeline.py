"""
Complete RAG (Retrieval-Augmented Generation) pipeline.
"""

from typing import List, Dict, Any, Optional
from src.retriever import Retriever
from src.generator import ResponseGenerator


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""
    
    def __init__(self, retriever: Optional[Retriever] = None,
                 generator: Optional[ResponseGenerator] = None,
                 top_k: int = 5, model_name: str = "gpt-4"):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Retriever instance (creates new if not provided)
            generator: ResponseGenerator instance (creates new if not provided)
            top_k: Number of similar cases to retrieve
            model_name: OpenAI model name for generation
        """
        self.retriever = retriever or Retriever()
        self.generator = generator or ResponseGenerator(model_name=model_name)
        self.top_k = top_k
    
    def run(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.
        
        Args:
            query: User query or patient case description
            filter_dict: Optional metadata filter for retrieval
            
        Returns:
            Dictionary containing retrieved documents, context, and generated response
        """
        retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k, filter_dict=filter_dict)
        
        context = self.retriever.format_context(retrieved_docs)
        
        response = self.generator.generate(query, context)
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "context": context,
            "response": response
        }
    
    def run_streaming(self, query: str, filter_dict: Optional[Dict[str, Any]] = None):
        """
        Run the RAG pipeline with streaming response.
        
        Args:
            query: User query or patient case description
            filter_dict: Optional metadata filter for retrieval
            
        Yields:
            Response chunks and retrieved documents info
        """
        retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k, filter_dict=filter_dict)
        context = self.retriever.format_context(retrieved_docs)
        
        yield {
            "type": "retrieved",
            "data": {
                "num_documents": len(retrieved_docs),
                "documents": retrieved_docs
            }
        }
        
        for chunk in self.generator.generate_streaming(query, context):
            yield {
                "type": "chunk",
                "data": chunk
            }
        
        yield {
            "type": "complete",
            "data": {}
        }


if __name__ == "__main__":
    pipeline = RAGPipeline(model_name="gpt-3.5-turbo", top_k=3)
    
    test_query = "65-year-old female presents with persistent cough, weight loss, and fatigue"
    result = pipeline.run(test_query)
    
    print(f"Query: {result['query']}\n")
    print(f"Retrieved {len(result['retrieved_documents'])} documents\n")
    print(f"Generated Response:\n{result['response']}")

