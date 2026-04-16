from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.api.chat.route import router as chat_router
from src.app.api.ingest.route import router as ingest_router

app = FastAPI(
    title="PDF RAG System",
    description="Query-based Search with LangChain + LangGraph + Qdrant",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include live API routers
app.include_router(ingest_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/", tags=["Health"])
async def health() -> dict:
    """Health check endpoint"""
    return {"status": "ok", "service": "PDF RAG System"}
