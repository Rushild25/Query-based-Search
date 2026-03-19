from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.lib.vespa.search import search_vespa

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    messages: list[dict]


@router.post("")
async def chat(payload: ChatRequest, authorization: str | None = Header(default=None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_messages = [m for m in payload.messages if m.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    try:
        last = user_messages[-1]
        relevant = search_vespa(str(last.get("content", "")))
        context = "\n\n".join([f"Source: {x.title} ({x.type})\nContent: {x.content}" for x in relevant])
        return {
            "reply": "I used the indexed context.",
            "system_context": context or "No relevant content found in your Google Workspace.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing your request: {exc}") from exc
