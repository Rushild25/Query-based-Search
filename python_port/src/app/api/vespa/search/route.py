from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
import requests
from pydantic import BaseModel

from src.lib.gemini import generate_gemini_answer

router = APIRouter(prefix="/api/vespa/search", tags=["vespa"])

VESPA_ENDPOINT = os.getenv("VESPA_SEARCH_ENDPOINT", "https://f6f0971d.a7339ade.z.vespa-app.cloud/search")


class VespaSearchRequest(BaseModel):
    query: str | None = None
    topK: int = 10
    checkUrl: str | None = None
    yql: str | None = None
    embedding: list[float] | None = None
    options: dict | None = None


def _cert_paths() -> tuple[str | None, str | None]:
    cert = os.getenv("VESPA_SERVE_CERT")
    key = os.getenv("VESPA_SERVE_KEY")
    if cert and key:
        return cert, key
    root = Path(__file__).resolve().parents[6]
    fallback_cert = root / "vespa_prime" / "security" / "serve.pem"
    fallback_key = root / "vespa_prime" / "serve_key.pem"
    if fallback_cert.exists() and fallback_key.exists():
        return str(fallback_cert), str(fallback_key)
    return None, None


def _post_to_vespa(payload: dict) -> dict:
    cert, key = _cert_paths()
    kwargs = {"json": payload, "timeout": 60}
    if cert and key:
        kwargs["cert"] = (cert, key)
    response = requests.post(VESPA_ENDPOINT, **kwargs)
    response.raise_for_status()
    return response.json()


@router.post("")
async def vespa_search(body: VespaSearchRequest) -> dict:
    try:
        if body.checkUrl:
            query = {
                "yql": f"select * from sources * where document_id contains \"{body.checkUrl}\";",
                "hits": 5,
            }
            vespa = _post_to_vespa(query)
            hits = ((vespa.get("root") or {}).get("children")) or []
            exact = [h for h in hits if (h.get("fields") or {}).get("document_id") == body.checkUrl]
            return {"exists": bool(exact), "matchCount": len(hits)}

        if body.yql:
            payload = {"yql": body.yql, "hits": ((body.options or {}).get("hits", body.topK))}
            vespa = _post_to_vespa(payload)
            return vespa

        if not body.query:
            raise HTTPException(status_code=400, detail="Missing query")

        words = body.query.strip().split()
        parts = [
            f'(title contains "{w}" OR abstract contains "{w}" OR introduction contains "{w}" OR conclusion contains "{w}")'
            for w in words
        ]
        combined = " OR ".join(parts)
        payload = {
            "yql": f"select * from sources * where {combined};",
            "ranking": "default",
            "hits": body.topK,
        }
        vespa = _post_to_vespa(payload)
        hits = ((vespa.get("root") or {}).get("children")) or []
        if not hits:
            raise HTTPException(status_code=404, detail="No results found")

        context_chunks = []
        for hit in hits:
            fields = hit.get("fields") or {}
            context_chunks.append(
                "\n".join(
                    [
                        f"Title: {fields.get('title', '')}",
                        f"Abstract: {fields.get('abstract', '')}",
                        f"Introduction: {fields.get('introduction', '')}",
                        f"Conclusion: {fields.get('conclusion', '')}",
                    ]
                )
            )
        prompt = (
            "Answer the following question based on the context below:\n\n"
            f"Question: {body.query}\n\n"
            "Context:\n"
            + "\n\n".join(context_chunks)
        )
        answer = generate_gemini_answer(prompt)
        return {"answer": answer, "query": body.query, "vespaHits": hits}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
