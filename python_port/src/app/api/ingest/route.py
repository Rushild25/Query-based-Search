from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.lib.rag_service import ingest_pdf_bytes_for_session, ingest_pdf_path, list_documents


router = APIRouter(prefix="/api/ingest", tags=["ingest"])


class FolderIngestRequest(BaseModel):
    folder: str = Field(default="pdf_inputs")
    session_id: str = Field(default="default-session")


@router.post("/pdf")
async def ingest_pdf(file: UploadFile = File(...), session_id: str = Form(default="default-session")) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return ingest_pdf_bytes_for_session(pdf_bytes, file.filename, session_id)


@router.post("/folder")
async def ingest_folder(payload: FolderIngestRequest) -> dict:
    root = Path(__file__).resolve().parents[5]
    folder = Path(payload.folder)
    if not folder.is_absolute():
        folder = root / folder
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    results: list[dict] = []
    for pdf_path in sorted(folder.rglob("*.pdf")):
        results.append(ingest_pdf_path(pdf_path, payload.session_id))

    return {
        "success": True,
        "folder": str(folder),
        "results": results,
        "documents_ingested": sum(1 for result in results if not result.get("duplicate")),
    }


@router.get("/documents")
async def documents() -> dict:
    docs = list_documents()
    return {"documents": docs, "count": len(docs)}
