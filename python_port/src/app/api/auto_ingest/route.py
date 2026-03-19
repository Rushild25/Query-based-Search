from __future__ import annotations

import asyncio
import os
import re
import urllib.parse

from fastapi import APIRouter, HTTPException
import httpx
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/auto-ingest", tags=["auto-ingest"])


class AutoIngestRequest(BaseModel):
    query: str
    limit: int = 20
    existingUrls: list[str] = Field(default_factory=list)


def parse_arxiv_xml(xml_text: str) -> list[dict]:
    papers = []
    entry_matches = re.findall(r"<entry>(.*?)</entry>", xml_text, re.S)
    for entry in entry_matches:
        id_match = re.search(r"<id>(.*?)</id>", entry, re.S)
        full_id = (id_match.group(1).split("/")[-1] if id_match else "").strip()
        if not full_id:
            continue
        title_match = re.search(r"<title>(.*?)</title>", entry, re.S)
        summary_match = re.search(r"<summary>(.*?)</summary>", entry, re.S)
        title = re.sub(r"\s+", " ", (title_match.group(1) if title_match else "Unknown Title")).strip()
        summary = re.sub(r"\s+", " ", (summary_match.group(1) if summary_match else "")).strip()
        papers.append(
            {
                "id": full_id,
                "title": title,
                "summary": summary,
                "htmlUrl": f"https://arxiv.org/html/{full_id}",
            }
        )
    return papers


@router.post("")
async def auto_ingest(body: AutoIngestRequest) -> dict:
    if not body.query:
        raise HTTPException(status_code=400, detail="Query is required")

    encoded_query = urllib.parse.quote(body.query)
    fetch_limit = body.limit * 3
    arxiv_url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}&start=0&max_results={fetch_limit}&"
        "sortBy=submittedDate&sortOrder=descending"
    )

    base_url = os.getenv("PY_PORT_BASE_URL", "http://127.0.0.1:8010")
    async with httpx.AsyncClient(timeout=40.0) as client:
        response = await client.get(
            arxiv_url,
            headers={"User-Agent": "Research-App/1.0", "Accept": "application/atom+xml"},
        )
        response.raise_for_status()
        all_papers = parse_arxiv_xml(response.text)

        if not all_papers:
            return {"success": True, "message": "No papers found for this query on arXiv", "papers": []}

        new_papers = [p for p in all_papers if p["htmlUrl"] not in body.existingUrls]
        successful = []
        processed = 0
        failed = 0
        already_in_vespa = 0

        for paper in new_papers:
            try:
                vespa_check = await client.post(f"{base_url}/api/vespa/search", json={"checkUrl": paper["htmlUrl"]})
                if vespa_check.is_success and vespa_check.json().get("exists"):
                    already_in_vespa += 1
                    continue

                head = await client.head(paper["htmlUrl"], headers={"User-Agent": "Research-App/1.0"})
                if not head.is_success:
                    failed += 1
                    continue

                extracted = await client.get(
                    f"{base_url}/api/extract",
                    params={"url": paper["htmlUrl"]},
                )
                if not extracted.is_success:
                    failed += 1
                    continue
                data = extracted.json()
                if not data.get("abstract") or not data.get("introduction") or not data.get("conclusion"):
                    failed += 1
                    continue

                load = await client.post(
                    f"{base_url}/api/vespa/load",
                    json={
                        "document": {
                            "url": paper["htmlUrl"],
                            "title": paper["title"],
                            "abstract": data.get("abstract", ""),
                            "introduction": data.get("introduction", ""),
                            "conclusion": data.get("conclusion", ""),
                        }
                    },
                )
                if not load.is_success:
                    failed += 1
                    continue

                processed += 1
                successful.append(
                    {
                        "url": paper["htmlUrl"],
                        "title": paper["title"],
                        "abstract": data.get("abstract", ""),
                        "introduction": data.get("introduction", ""),
                        "conclusion": data.get("conclusion", ""),
                        "indexed": True,
                    }
                )
                if processed >= body.limit:
                    break
                await asyncio.sleep(0.5)
            except Exception:
                failed += 1

    return {
        "success": True,
        "message": f"Auto-ingestion completed: {processed} new papers ingested",
        "papers": successful,
        "stats": {
            "total_found": len(all_papers),
            "new_papers": len(new_papers),
            "already_in_vespa": already_in_vespa,
            "processed": processed,
            "failed": failed,
        },
    }
