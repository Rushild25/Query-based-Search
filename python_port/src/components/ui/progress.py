from __future__ import annotations


def progress(value: int) -> str:
    value = max(0, min(100, value))
    return f'<progress max="100" value="{value}" class="progress"></progress>'
