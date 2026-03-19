from __future__ import annotations

from bs4 import BeautifulSoup
from fastapi import APIRouter, HTTPException, Query
import httpx

router = APIRouter(prefix="/api/extract", tags=["extract"])


def _first_text_by_selectors(soup: BeautifulSoup, selectors: list[str], min_len: int) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(" ", strip=True)
            if len(text) > min_len:
                return text
    return ""


def _section_after_heading(soup: BeautifulSoup, heading_predicate, min_len: int) -> str:
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
        title = heading.get_text(" ", strip=True).lower()
        if not heading_predicate(title):
            continue
        parts: list[str] = []
        cursor = heading.find_next_sibling()
        while cursor and cursor.name not in {"h1", "h2", "h3", "h4", "h5"}:
            if cursor.name == "p":
                txt = cursor.get_text(" ", strip=True)
                if txt:
                    parts.append(txt)
            cursor = cursor.find_next_sibling()
        text = "\n\n".join(parts).strip()
        if len(text) > min_len:
            return text
    return ""


@router.get("")
async def extract(url: str = Query(...)) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; Research-Bot/1.0)"},
            )
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        title = (soup.title.get_text(" ", strip=True) if soup.title else "").replace("arXiv.org", "").strip()

        abstract = _first_text_by_selectors(
            soup,
            [
                ".ltx_abstract p",
                ".abstract p",
                "div[class*='abstract'] p",
                "div.abstract",
                "#abstract",
            ],
            50,
        ) or _section_after_heading(soup, lambda t: t == "abstract", 50)

        intro = _first_text_by_selectors(
            soup,
            [
                "#S1 p",
                "#sec1 p",
                "#section1 p",
                "#introduction p",
            ],
            100,
        ) or _section_after_heading(soup, lambda t: "introduction" in t, 100)

        conclusion = _first_text_by_selectors(
            soup,
            [
                "#conclusion p",
                "#conclusions p",
                "#discussion p",
            ],
            100,
        ) or _section_after_heading(
            soup,
            lambda t: any(x in t for x in ["conclusion", "discussion", "summary", "final remarks"]),
            100,
        )

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        long_paragraphs = [p for p in paragraphs if len(p) > 100]

        if not intro and long_paragraphs:
            candidate = "\n\n".join(long_paragraphs[:3]).strip()
            if len(candidate) > 200:
                intro = candidate
        if not conclusion and long_paragraphs:
            candidate = "\n\n".join(long_paragraphs[-3:]).strip()
            if len(candidate) > 200:
                conclusion = candidate

        return {
            "url": url,
            "title": title,
            "abstract": abstract or "No abstract could be extracted from this paper.",
            "introduction": intro or "No introduction could be extracted from this paper.",
            "conclusion": conclusion or "No conclusion could be extracted from this paper.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to extract paper data: {exc}") from exc
