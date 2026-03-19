from __future__ import annotations


def input_field(name: str, placeholder: str = "", value: str = "") -> str:
    return f'<input name="{name}" placeholder="{placeholder}" value="{value}" class="input" />'
