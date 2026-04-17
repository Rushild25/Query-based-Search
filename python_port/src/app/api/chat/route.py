from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.lib.rag_service import answer_question

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    messages: list[dict]
    session_id: str
    doc_id: str | None = None
    top_k: int = 5


@router.post("")
async def chat(payload: ChatRequest) -> dict:
    user_messages = [m for m in payload.messages if m.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    if not str(payload.session_id).strip():
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        last = user_messages[-1]
        result = answer_question(
            str(last.get("content", "")),
            session_id=payload.session_id,
            doc_id=payload.doc_id,
            limit=payload.top_k,
            history=payload.messages,
        )
        return {"reply": result["answer"], **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing your request: {exc}") from exc
