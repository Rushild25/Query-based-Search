from __future__ import annotations


def badge(text: str, variant: str = "default") -> str:
    return f'<span data-variant="{variant}" class="badge">{text}</span>'
