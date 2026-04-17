from __future__ import annotations


def clean_string(value: str) -> str:
    return "".join(ch for ch in value if ch == "\n" or ch == "\t" or ord(ch) >= 32)
