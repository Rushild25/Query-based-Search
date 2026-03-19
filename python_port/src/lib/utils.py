from __future__ import annotations

import math
import random
import string
from datetime import datetime


def cn(*inputs: str) -> str:
    return " ".join(x for x in inputs if x).strip()


def nanoid(size: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(size))


def format_date(date_string: str) -> str:
    date = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
    return date.strftime("%b %d, %Y")


def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_bytes(value: int, decimals: int = 2) -> str:
    if value == 0:
        return "0 Bytes"
    k = 1024
    dm = max(decimals, 0)
    sizes = ["Bytes", "KB", "MB", "GB", "TB"]
    idx = int(math.floor(math.log(value, k)))
    out = round(value / (k ** idx), dm)
    return f"{out} {sizes[idx]}"


def calculate_reading_time(text: str) -> int:
    words = len(text.strip().split())
    return max(1, math.ceil(words / 200))


def clean_string(value: str) -> str:
    return "".join(ch for ch in value if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks: list[str] = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_chunk_size, len(text))
        if end_idx < len(text):
            breakpoints = [
                text.rfind(". ", start_idx, end_idx),
                text.rfind("! ", start_idx, end_idx),
                text.rfind("? ", start_idx, end_idx),
                text.rfind("\n\n", start_idx, end_idx),
            ]
            breakpoints = [bp for bp in breakpoints if bp > start_idx]
            if breakpoints:
                end_idx = max(breakpoints) + 1
        chunk = text[start_idx:end_idx].strip()
        if not chunk:
            break
        chunks.append(clean_string(chunk))
        next_start = end_idx - overlap
        start_idx = next_start if next_start > start_idx else end_idx
    return chunks
