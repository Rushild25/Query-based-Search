from __future__ import annotations


def label(text: str, for_id: str) -> str:
    return f'<label for="{for_id}" class="label">{text}</label>'
