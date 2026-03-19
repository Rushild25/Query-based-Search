from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class VespaClient:
    base_url: str

    @classmethod
    def from_env(cls) -> "VespaClient":
        return cls(base_url=os.getenv("PY_PORT_BASE_URL", "http://127.0.0.1:8010"))

    def index_document(self, document_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "document": {
                "title": fields.get("title", ""),
                "abstract": fields.get("abstract", ""),
                "introduction": fields.get("introduction", ""),
                "conclusion": fields.get("conclusion", ""),
                "url": fields.get("url", document_id),
            }
        }
        resp = requests.post(f"{self.base_url}/api/vespa/load", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def search(self, yql: str, embedding: list[float], options: dict[str, Any] | None = None) -> dict[str, Any]:
        options = options or {}
        payload = {
            "yql": yql,
            "embedding": embedding,
            "options": options,
        }
        resp = requests.post(f"{self.base_url}/api/vespa/search", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()


_client: VespaClient | None = None


def get_vespa_client() -> VespaClient:
    global _client
    if _client is None:
        _client = VespaClient.from_env()
    return _client
