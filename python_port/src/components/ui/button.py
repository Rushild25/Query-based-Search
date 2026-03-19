from __future__ import annotations


def button(text: str, button_type: str = "button", variant: str = "default") -> str:
    return f'<button type="{button_type}" data-variant="{variant}" class="button">{text}</button>'
