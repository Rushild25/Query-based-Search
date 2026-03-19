from __future__ import annotations

import hashlib


def generate_embedding(text: str, dim: int = 384) -> list[float]:
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    while len(values) < dim:
        for b in seed:
            values.append((b / 255.0) * 2 - 1)
            if len(values) == dim:
                break
        seed = hashlib.sha256(seed).digest()
    return values
