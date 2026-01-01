"""
Embedding generation using OpenAI's text-embedding-3-small model.
"""

import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()


class EmbeddingGenerator:
    """Generates embeddings using OpenAI's embedding model."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: OpenAI embedding model name
            api_key: Optional API key (if not provided, uses environment variable)
        """
        self.model_name = model_name
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100, 
                                   delay: float = 0.1) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with rate limiting.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            delay: Delay in seconds between batches to respect rate limits
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        total = len(texts)
        
        print(f"Generating embeddings for {total} texts...")
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if (i + batch_size) % 1000 == 0 or i + batch_size >= total:
                    print(f"Generated embeddings for {min(i + batch_size, total)}/{total} texts...")
                
                if i + batch_size < total:
                    time.sleep(delay)
            
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                raise
        
        print(f"Embedding generation complete. Total embeddings: {len(all_embeddings)}")
        return all_embeddings


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    test_texts = [
        "Patient presents with chest pain and shortness of breath.",
        "Diagnosis: Acute myocardial infarction. Treatment: Aspirin and statin therapy."
    ]
    embeddings = generator.generate_embeddings_batch(test_texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

