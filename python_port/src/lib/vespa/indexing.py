from __future__ import annotations

from src.lib.vespa.client import get_vespa_client


def index_document_in_vespa(document: dict) -> dict:
    client = get_vespa_client()
    client.index_document(
        document.get("url", ""),
        {
            "title": document.get("title", ""),
            "abstract": document.get("abstract", ""),
            "introduction": document.get("introduction", ""),
            "conclusion": document.get("conclusion", ""),
            "url": document.get("url", ""),
        },
    )
    return {"success": True, "document_id": document.get("url", "")}
