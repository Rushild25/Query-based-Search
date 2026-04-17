from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings


@dataclass(frozen=True)
class ClipBackend:
    model: Any
    preprocess: Any
    tokenizer: Any
    device: Any
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch

        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []

    def embed_image(self, image) -> list[float]:
        import torch

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().tolist()[0]


class ClipTextEmbeddings(Embeddings):
    def __init__(self, backend: ClipBackend) -> None:
        self._backend = backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._backend.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._backend.embed_text(text)


@lru_cache(maxsize=1)
def get_clip_backend(model_name: str, pretrained: str) -> ClipBackend:
    try:
        import open_clip
        import torch
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError(
            "CLIP support requires open_clip_torch and torch. Install the visual embedding dependencies first."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        sample = tokenizer(["dimension probe"]).to(device)
        dimension = int(model.encode_text(sample).shape[-1])

    return ClipBackend(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        dimension=dimension,
    )
