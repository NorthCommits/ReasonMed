"""
LLM response generation using OpenAI API.
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class ResponseGenerator:
    """Generates responses using OpenAI's GPT models."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the response generator.
        
        Args:
            model_name: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: Optional API key (if not provided, uses environment variable)
        """
        self.model_name = model_name
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, query: str, context: str, 
                 system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            query: User query or patient case description
            context: Retrieved context from similar cases
            system_prompt: Optional system prompt (uses default if not provided)
            
        Returns:
            Generated response text
        """
        if system_prompt is None:
            system_prompt = """You are a medical clinical documentation assistant. 
Your role is to help doctors write clinical notes by analyzing similar cases and 
providing structured documentation suggestions based on the retrieved medical cases.

Use the provided similar cases as reference, but adapt your response to the specific 
patient case. Provide clear, concise, and clinically accurate documentation."""
        
        user_prompt = f"""Based on the following similar medical cases, provide clinical 
documentation suggestions for this patient case:

Patient Case:
{query}

Similar Cases for Reference:
{context}

Please provide:
1. Key findings and observations
2. Differential diagnosis considerations
3. Recommended diagnostic approach
4. Treatment considerations

Format your response as clear, professional clinical documentation."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error generating response: {e}")
            raise
    
    def generate_streaming(self, query: str, context: str,
                          system_prompt: Optional[str] = None):
        """
        Generate a streaming response using the LLM.
        
        Args:
            query: User query or patient case description
            context: Retrieved context from similar cases
            system_prompt: Optional system prompt (uses default if not provided)
            
        Yields:
            Response chunks as they are generated
        """
        if system_prompt is None:
            system_prompt = """You are a medical clinical documentation assistant. 
Your role is to help doctors write clinical notes by analyzing similar cases and 
providing structured documentation suggestions based on the retrieved medical cases.

Use the provided similar cases as reference, but adapt your response to the specific 
patient case. Provide clear, concise, and clinically accurate documentation."""
        
        user_prompt = f"""Based on the following similar medical cases, provide clinical 
documentation suggestions for this patient case:

Patient Case:
{query}

Similar Cases for Reference:
{context}

Please provide:
1. Key findings and observations
2. Differential diagnosis considerations
3. Recommended diagnostic approach
4. Treatment considerations

Format your response as clear, professional clinical documentation."""
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            print(f"Error generating streaming response: {e}")
            raise


if __name__ == "__main__":
    generator = ResponseGenerator(model_name="gpt-3.5-turbo")
    test_query = "45-year-old male presents with chest pain"
    test_context = "Similar case: Patient with chest pain diagnosed with acute myocardial infarction..."
    response = generator.generate(test_query, test_context)
    print("Generated response:")
    print(response)

