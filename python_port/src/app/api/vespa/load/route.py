from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
import requests
from pydantic import BaseModel

router = APIRouter(prefix="/api/vespa/load", tags=["vespa"])


class DocumentInput(BaseModel):
    url: str
    title: str
    introduction: str
    abstract: str
    conclusion: str


class VespaLoadRequest(BaseModel):
    document: DocumentInput


def _cert_paths() -> tuple[str | None, str | None]:
    cert = os.getenv("VESPA_INGEST_CERT")
    key = os.getenv("VESPA_INGEST_KEY")
    if cert and key:
        return cert, key
    root = Path(__file__).resolve().parents[6]
    fallback_cert = root / "vespa_prime" / "security" / "ingest.pem"
    fallback_key = root / "vespa_prime" / "ingest_key.pem"
    if fallback_cert.exists() and fallback_key.exists():
        return str(fallback_cert), str(fallback_key)
    return None, None


@router.post("")
async def vespa_load(payload: VespaLoadRequest) -> dict:
    doc = payload.document
    if not doc.title or not doc.abstract:
        raise HTTPException(status_code=400, detail="Missing required document fields")

    cert, key = _cert_paths()
    base_url = os.getenv("VESPA_BASE_URL", "https://f6f0971d.a7339ade.z.vespa-app.cloud/")
    endpoint = f"{base_url.rstrip('/')}/document/v1/msmarco/passage/docid/{doc.url or 0}"
    req_payload = {
        "fields": {
            "document_id": doc.url,
            "title": doc.title,
            "abstract": doc.abstract,
            "introduction": doc.introduction,
            "conclusion": doc.conclusion,
        }
    }
    try:
        kwargs = {"json": req_payload, "timeout": 60}
        if cert and key:
            kwargs["cert"] = (cert, key)
        response = requests.post(endpoint, **kwargs)
        response.raise_for_status()
        return {"message": "Document indexed", "data": response.json()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
