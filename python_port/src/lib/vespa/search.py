from __future__ import annotations

import json
from dataclasses import dataclass

from src.lib.embeddings import generate_embedding
from src.lib.vespa.client import get_vespa_client


@dataclass
class SearchResult:
    id: str
    title: str
    content: str
    score: float
    type: str
    metadata: dict


def search_vespa(query: str, limit: int = 5) -> list[SearchResult]:
    try:
        client = get_vespa_client()
        embedding = generate_embedding(query)
        yql = (
            "select * from content_chunk "
            f"where ({{targetHits:{limit}}}nearestNeighbor(embedding, query_embedding)) or "
            f"({{grammar: \"loose\"}}userQuery(\"content:{query}\")) "
            "order by (0.7 * closeness(embedding) + 0.3 * bm25(content)) "
            f"limit {limit}"
        )
        result = client.search(yql, embedding, {"ranking": "hybrid", "hits": limit})
        children = (result.get("root") or {}).get("children") or []
        out: list[SearchResult] = []
        for hit in children:
            fields = hit.get("fields", {})
            raw_meta = fields.get("metadata") or "{}"
            try:
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except Exception:
                meta = {}
            out.append(
                SearchResult(
                    id=str(fields.get("chunk_id", "")),
                    title=str(fields.get("title", "")),
                    content=str(fields.get("content", "")),
                    score=float(hit.get("relevance", 0.0)),
                    type=str(fields.get("document_type", "doc")),
                    metadata=meta,
                )
            )
        return out
    except Exception:
        return []
