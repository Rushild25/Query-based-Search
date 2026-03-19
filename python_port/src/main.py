from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.app.api.auto_ingest.route import router as auto_ingest_router
from src.app.api.chat.route import router as chat_router
from src.app.api.extract.route import router as extract_router
from src.app.api.vespa.load.route import router as vespa_load_router
from src.app.api.vespa.search.route import router as vespa_search_router
from src.app.chat.page import render_chat_page
from src.app.page import render_research_page

app = FastAPI(title="TypeScript to Python Port")

app.include_router(extract_router)
app.include_router(auto_ingest_router)
app.include_router(chat_router)
app.include_router(vespa_load_router)
app.include_router(vespa_search_router)


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return render_research_page()


@app.get("/chat", response_class=HTMLResponse)
async def chat_page() -> str:
    return render_chat_page()
