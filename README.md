# CliniScribe AI

A medical clinical documentation assistant that helps doctors write clinical notes by retrieving similar cases and diagnostic reasoning patterns from a medical dataset using RAG (Retrieval-Augmented Generation).

## Features

- Retrieval-Augmented Generation (RAG) for medical case analysis
- Similar case retrieval from medical dataset
- Diagnostic reasoning pattern matching
- FastAPI REST API
- Streamlit web interface
- ChromaDB vector database for efficient similarity search

## Technology Stack

- **Backend**: Python 3.10+, FastAPI
- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **LLM**: OpenAI API (GPT-4 or GPT-3.5-turbo)
- **Embeddings**: OpenAI text-embedding-3-small
- **Data Source**: FreedomIntelligence/medical-o1-reasoning-SFT from HuggingFace

## Setup

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key: `OPENAI_API_KEY=sk-...`

4. **Initialize the vector database**:
   ```bash
   python scripts/setup_vectorstore.py
   ```
   This will download the dataset from HuggingFace, process it, generate embeddings, and store them in ChromaDB.

## Usage

### Streamlit UI

Run the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

### FastAPI Server

Start the API server:
```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the FastAPI server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
reasonmed/
├── .env                          # API keys (gitignored)
├── .env.example                  # Template for environment variables
├── .gitignore
├── requirements.txt              # Python dependencies
├── README.md
├── data/
│   ├── raw/                      # Original HuggingFace data
│   └── processed/                # Processed embeddings/vectorstore
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # Data loading & processing
│   ├── embeddings.py             # Embedding generation
│   ├── vectorstore.py            # Vector database operations
│   ├── retriever.py              # Similarity search logic
│   ├── generator.py              # LLM response generation
│   └── rag_pipeline.py           # Complete RAG workflow
├── api/
│   ├── __init__.py
│   └── main.py                   # FastAPI endpoints
├── app/
│   └── streamlit_app.py          # Streamlit UI
└── scripts/
    ├── setup_vectorstore.py      # One-time setup script
    └── test_rag.py               # Testing script
```

## Testing

Run the test script to verify the RAG pipeline:
```bash
python scripts/test_rag.py
```

## License

This project is for educational and research purposes.

