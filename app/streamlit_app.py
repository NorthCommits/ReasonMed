"""
Streamlit UI for CliniScribe AI.
"""

import streamlit as st
import sys
import os
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore

st.set_page_config(
    page_title="CliniScribe AI",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_rag_pipeline(model_name: str = "gpt-4"):
    """Load the RAG pipeline (cached)."""
    return RAGPipeline(model_name=model_name)

@st.cache_resource
def load_vector_store():
    """Load the vector store (cached)."""
    return VectorStore()

def main():
    """Main Streamlit application."""
    st.title("🏥 CliniScribe AI")
    st.markdown("### Medical Clinical Documentation Assistant")
    st.markdown("---")
    
    sidebar = st.sidebar
    sidebar.header("Settings")
    
    model_name = sidebar.selectbox(
        "OpenAI Model",
        ["gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Select the OpenAI model for generation"
    )
    
    top_k = sidebar.slider(
        "Number of Similar Cases",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of similar cases to retrieve"
    )
    
    sidebar.markdown("---")
    
    try:
        vector_store = load_vector_store()
        doc_count = vector_store.get_collection_count()
        sidebar.metric("Vector Store Documents", doc_count)
    except Exception as e:
        sidebar.error(f"Error loading vector store: {e}")
        st.stop()
    
    if doc_count == 0:
        st.warning("⚠️ Vector store is empty. Please run the setup script first:")
        st.code("python scripts/setup_vectorstore.py", language="bash")
        st.stop()
    
    tab1, tab2 = st.tabs(["Query", "About"])
    
    with tab1:
        st.header("Enter Patient Case")
        
        query_text = st.text_area(
            "Patient Case Description",
            height=150,
            placeholder="Enter the patient case description, symptoms, and relevant medical information..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.button("Generate Documentation", type="primary", use_container_width=True)
        
        if submit_button and query_text:
            with st.spinner("Retrieving similar cases and generating documentation..."):
                try:
                    pipeline = load_rag_pipeline(model_name=model_name)
                    pipeline.top_k = top_k
                    
                    result = pipeline.run(query_text)
                    
                    st.markdown("---")
                    st.header("Generated Clinical Documentation")
                    st.markdown(result["response"])
                    
                    st.markdown("---")
                    st.header(f"Similar Cases ({len(result['retrieved_documents'])})")
                    
                    for i, doc in enumerate(result["retrieved_documents"], 1):
                        with st.expander(f"Case {i} - Similarity: {doc['similarity_score']:.3f}"):
                            st.markdown("**Question:**")
                            st.markdown(doc["metadata"].get("full_question", "N/A"))
                            
                            st.markdown("**Reasoning:**")
                            reasoning = doc["metadata"].get("full_reasoning", "N/A")
                            st.markdown(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning)
                            
                            st.markdown("**Diagnosis:**")
                            st.markdown(doc["metadata"].get("full_response", "N/A"))
                            
                            st.markdown("**Keywords:**")
                            st.markdown(doc["metadata"].get("medical_keywords", "N/A"))
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.exception(e)
        
        elif submit_button and not query_text:
            st.warning("Please enter a patient case description.")
    
    with tab2:
        st.header("About CliniScribe AI")
        st.markdown("""
        CliniScribe AI is a medical clinical documentation assistant that helps doctors 
        write clinical notes by retrieving similar cases and diagnostic reasoning patterns 
        from a medical dataset.
        
        ### Features
        - **Retrieval-Augmented Generation (RAG)**: Combines similarity search with LLM generation
        - **Medical Case Database**: Uses the FreedomIntelligence medical-o1-reasoning-SFT dataset
        - **Similar Case Retrieval**: Finds relevant medical cases based on patient presentation
        - **Structured Documentation**: Generates clear, professional clinical documentation
        
        ### How It Works
        1. Enter a patient case description
        2. The system retrieves similar cases from the medical dataset
        3. The LLM generates clinical documentation based on the retrieved cases
        4. Review the generated documentation and similar cases
        
        ### Technology
        - **Vector Database**: ChromaDB for efficient similarity search
        - **Embeddings**: OpenAI text-embedding-3-small
        - **LLM**: OpenAI GPT-4 or GPT-3.5-turbo
        - **Framework**: Streamlit for the UI
        
        ### Disclaimer
        This tool is for educational and research purposes. It should not be used as a 
        substitute for professional medical judgment, diagnosis, or treatment.
        """)


if __name__ == "__main__":
    main()

