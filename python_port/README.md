# Query-based Search — Python PDF RAG

FastAPI service that ingests PDFs into **Qdrant Cloud**, retrieves with **hybrid text + visual (CLIP) search**, and answers with **Google Gemini** (optional Groq fallback). Scanned pages, figures, and tables are supported via **Gemini vision transcription** at ingest time when native PDF text is sparse.

---

## What this stack does

1. **Text layer** — PyMuPDF extracts embedded text; chunks are embedded (default: **CLIP text**) and stored in the main Qdrant collection (`research_chunks` by default).

2. **Visual layer** — Each page is rendered to an image; **CLIP image** vectors go to a separate collection (`research_visual_chunks`) so retrieval can still find relevant pages when the text layer is empty or thin.

3. **Page vision (Gemini)** — If a page has **few characters** of native text (default threshold: 64), the same page image is sent to **Gemini** once to produce a plain-text transcript and short descriptions of charts/diagrams. That text is:
   - chunked into **`from_image`** documents in the main collection (good for keyword-style questions), and  
   - merged into the visual point’s **`page_content`** so the chat model sees real context when a visual hit is returned.

4. **Answering** — Retrieved chunks (text, `from_image`, and visual summaries) are passed to an LLM with a grounded prompt. **GEMINI_API_KEY** is required for synthesis; **GROQ_API_KEY** can be used as fallback if configured.

Global defaults live in **`../rag_config.yaml`** (repo root). Environment variables override YAML (see `src/lib/settings.py`).

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10+ recommended (matches LangChain / torch stacks). |
| **Qdrant Cloud** | Free tier is fine for development. You need `QDRANT_URL` and `QDRANT_API_KEY`. |
| **RAM / disk** | CLIP + PyTorch: allow a few GB for first model load; first run downloads OpenCLIP weights. |
| **Gemini** | Required for **answers** and for **page vision transcription** on low-text pages. |

Optional: **Groq** API key for fallback chat models.

---

## Configuration

### 1. Environment file

Create **`python_port/.env`** (or run the API from `python_port` so a local `.env` is picked up). `get_settings()` also attempts the repo-root `.env` if present.

**Typical variables:**

| Variable | Purpose |
|----------|---------|
| `QDRANT_URL` | Qdrant Cloud HTTPS URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `GEMINI_API_KEY` | Chat + page vision transcription |
| `EMBEDDING_PROVIDER` | `clip` (recommended for hybrid visual retrieval) |
| `RAG_VISUAL_EMBEDDINGS_ENABLED` | `true` to index page images in the visual collection |
| `GROQ_API_KEY` | Optional fallback for LLM |
| `GOOGLE_GENAI_MODEL` | Override Gemini model id (must support vision for page transcription) |

**Page vision tuning (optional):**

| Variable | Default | Meaning |
|----------|---------|---------|
| `RAG_PAGE_VISION_ENABLED` | `true` | Set `false` to skip Gemini-on-image at ingest |
| `RAG_PAGE_VISION_TEXT_MAX_CHARS` | `64` | If native `page.get_text()` is longer than this, **skip** Gemini for that page (saves API calls on text-heavy PDFs) |
| `RAG_PAGE_VISION_MODEL` | unset | If set, overrides `GOOGLE_GENAI_MODEL` **only** for page transcription |
| `RAG_VISUAL_PAGE_CONTENT_MAX_CHARS` | `16000` | Max characters stored on each visual point’s `page_content` |

### 2. YAML

Edit **`../rag_config.yaml`** for collection names, chunk sizes, CLIP model id, `top_k`, and LLM defaults.

---

## Setup (from scratch)

### macOS / Linux

```bash
cd python_port
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Copy or create `.env` with at least `QDRANT_URL`, `QDRANT_API_KEY`, and `GEMINI_API_KEY`.

### Windows (PowerShell)

```powershell
cd python_port
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

---

## Run the API

From **`python_port`** (so imports and `.env` resolve correctly):

```bash
python -m uvicorn src.main:app --reload --port 8010
```

- Health: [http://127.0.0.1:8010/](http://127.0.0.1:8010/)
- OpenAPI: [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs/)

Routers are mounted under **`/api`**, and the ingest router uses prefix **`/api/ingest`**, so effective ingest/chat paths are prefixed twice with **`api`** (e.g. `POST /api/api/ingest/pdf`). The bundled Streamlit app already uses these URLs.

---

## Run the Streamlit UI (optional)

In a second terminal, still inside **`python_port`**:

```bash
streamlit run streamlit_app.py
```

Set **Backend URL** to `http://127.0.0.1:8010` unless you proxy elsewhere. Use the same **Session ID** for ingest and chat. After ingest, the UI can set **Document ID** from the response to scope chat to one PDF.

---

## Typical workflow

1. **Ingest** — `POST /api/api/ingest/pdf` with `multipart/form-data`: `file`, optional `session_id`, optional `doc_id`.  
   - **Image-heavy PDFs** may show `chunks_created: 0` for native text but **`visual_chunks_created` > 0** and/or **`from_image`** chunks after vision runs.  
   - Re-ingest after changing vision or chunk settings so Qdrant matches your new pipeline.

2. **List** — `GET /api/api/ingest/documents?session_id=...`

3. **Chat** — `POST /api/api/chat` with JSON: `messages`, `session_id`, optional `doc_id`, optional `top_k`.

4. **Clear** — `DELETE /api/api/ingest/documents?session_id=...` (optional `doc_id`) to remove points before a clean re-index.

---

## Local PDF cache

Ingestion persists PDF copies and rendered page PNGs under the repo’s **`pdf_cache/`** directory (session-scoped). This folder should stay **out of git** (see root `.gitignore`).

---

## Key HTTP endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/api/ingest/pdf` | Upload one PDF |
| `POST` | `/api/api/ingest/folder` | Index PDFs from a configured folder |
| `GET` | `/api/api/ingest/documents` | List documents for a session |
| `DELETE` | `/api/api/ingest/documents` | Delete vectors for session / doc |
| `POST` | `/api/api/chat` | RAG chat |

---

## Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| Ingest returns “No content could be extracted” | Visual indexing off and no text layer; or every page failed CLIP/vision silently. Enable `RAG_VISUAL_EMBEDDINGS_ENABLED`, set `GEMINI_API_KEY`, check logs. |
| Chat says no chunks / empty context | Same `session_id` as ingest; `doc_id` filter matches ingested `doc_id`; document not duplicate-skipped with empty prior run. |
| Weak answers on scans before vision | Re-ingest after enabling Gemini; confirm `RAG_PAGE_VISION_ENABLED` is not `false`. |
| Slow first request | OpenCLIP/torch loading and cold Qdrant; normal. |
| 500 on chat | Missing LLM keys or provider errors; see API `detail` JSON. |

---

## Project layout (this port)

```
python_port/
  src/
    main.py                 # FastAPI app
    app/api/ingest/         # Ingest routes
    app/api/chat/           # Chat route
    lib/
      rag_service.py        # Ingest + retrieval + LLM graph
      settings.py           # Env + YAML settings
      clip_backend.py       # OpenCLIP helpers
  streamlit_app.py          # Optional UI
  requirements.txt
```

Parent repo provides **`rag_config.yaml`** at the workspace root.
