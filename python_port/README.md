# Python Port

## Architecture

- FastAPI backend
- LangChain for CLIP text embeddings, vector retrieval, and answer chains
- LangGraph for ingestion and QA workflows
- Qdrant Cloud for semantic vector storage and retrieval
- Separate Qdrant collection for visual CLIP vectors

## Prerequisites

1. Create a Qdrant Cloud cluster (free tier works for testing)
2. Copy credentials into the root `.env` file (`QDRANT_URL`, `QDRANT_API_KEY`)
3. Set `EMBEDDING_PROVIDER=clip` and `RAG_VISUAL_EMBEDDINGS_ENABLED=true` to enable the multimodal pipeline
4. Set `GEMINI_API_KEY` if you want LLM answer synthesis; CLIP is used for ingestion-time visual indexing and fast hybrid retrieval
5. Install the CLIP runtime dependencies from `requirements.txt` (`open_clip_torch`, `torch`, `torchvision`)

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
- `http://127.0.0.1:8010/docs`

Run Streamlit frontend (optional):

```powershell
streamlit run streamlit_app.py
```

## Test Flow

1. Clear existing vectors if needed with `DELETE /api/api/ingest/documents?session_id=...`
2. Upload/ingest a PDF through `POST /api/api/ingest/pdf` or Streamlit
3. Query with `POST /api/api/chat` using the same `session_id` (and optional `doc_id`)
4. Verify `GET /api/api/ingest/documents` returns session-scoped document metadata with text and visual counts

## Key Endpoints

- `POST /api/api/ingest/pdf` - upload and index one PDF
- `POST /api/api/ingest/folder` - index every PDF in a folder
- `GET /api/api/ingest/documents` - list indexed documents
- `DELETE /api/api/ingest/documents` - delete session/doc points from Qdrant
- `POST /api/api/chat` - hybrid text + visual retrieval and grounded answer generation
