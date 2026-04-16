# Python Port

## Architecture

- FastAPI backend
- LangChain for embeddings, vector retrieval, and answer chains
- LangGraph for ingestion and QA workflows
- Qdrant Cloud for semantic vector storage and retrieval

## Prerequisites

1. Create a Qdrant Cloud cluster (free tier works for testing)
2. Copy credentials into the root `.env` file (`QDRANT_URL`, `QDRANT_API_KEY`)
3. Set `EMBEDDING_PROVIDER=huggingface` (default) or `openai`
4. Set `GEMINI_API_KEY` if you want LLM answer synthesis and visual-page analysis

Global configuration lives in `../rag_config.yaml` and can be overridden by `.env` variables.

## Run

```powershell
cd python_port
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.main:app --reload --port 8010
```

Open:

- `http://127.0.0.1:8010/`
- `http://127.0.0.1:8010/chat`

## Test Flow

1. Ingest a local folder from the home page (`pdf_inputs` by default)
2. Open chat page and ask questions about paper content, figures, and tables
3. Verify `/api/ingest/documents` returns document metadata from Qdrant

## Key Endpoints

- `POST /api/ingest/pdf` - upload and index one PDF
- `POST /api/ingest/folder` - index every PDF in a folder
- `GET /api/ingest/documents` - list indexed documents
- `POST /api/chat` - semantic retrieval + grounded answer generation
