"""
FastAPI endpoints for ReasonMed.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore

app = FastAPI(
    title="ReasonMed API",
    description="Medical clinical documentation assistant API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = None
vector_store = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or initialize the RAG pipeline."""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline


def get_vector_store() -> VectorStore:
    """Get or initialize the vector store."""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    top_k: Optional[int] = 5
    model_name: Optional[str] = "gpt-4"


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    response: str
    retrieved_documents: List[Dict[str, Any]]
    num_retrieved: int


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ReasonMed API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Query the RAG pipeline",
            "/health": "GET - Health check",
            "/stats": "GET - Vector store statistics",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        store = get_vector_store()
        count = store.get_collection_count()
        return {
            "status": "healthy",
            "vector_store_count": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    try:
        store = get_vector_store()
        count = store.get_collection_count()
        return {
            "collection_name": store.collection_name,
            "document_count": count,
            "status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG pipeline with a medical case description.
    
    Args:
        request: Query request with query text, top_k, and optional model name
        
    Returns:
        Query response with generated documentation and retrieved documents
    """
    try:
        pipeline = get_rag_pipeline()
        pipeline.top_k = request.top_k
        
        if request.model_name and request.model_name != "gpt-4":
            from src.generator import ResponseGenerator
            pipeline.generator = ResponseGenerator(model_name=request.model_name)
        
        result = pipeline.run(request.query)
        
        return QueryResponse(
            query=result["query"],
            response=result["response"],
            retrieved_documents=result["retrieved_documents"],
            num_retrieved=len(result["retrieved_documents"])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Query the RAG pipeline with streaming response.
    
    Args:
        request: Query request with query text, top_k, and optional model name
        
    Yields:
        Streaming response chunks
    """
    try:
        pipeline = get_rag_pipeline()
        pipeline.top_k = request.top_k
        
        if request.model_name and request.model_name != "gpt-4":
            from src.generator import ResponseGenerator
            pipeline.generator = ResponseGenerator(model_name=request.model_name)
        
        for chunk in pipeline.run_streaming(request.query):
            yield chunk
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

