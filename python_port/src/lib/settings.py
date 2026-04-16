from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "rag_config.yaml"


@dataclass(frozen=True)
class VectorDbSettings:
    provider: str
    collection_name: str
    distance: str
    prefer_grpc: bool


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    huggingface_model: str
    openai_model: str
    batch_size: int


@dataclass(frozen=True)
class LlmSettings:
    provider: str
    google_model: str
    temperature: float


@dataclass(frozen=True)
class IngestionSettings:
    chunk_size: int
    chunk_overlap: int
    analyse_visuals: bool
    max_visual_pages_per_document: int


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int


@dataclass(frozen=True)
class RagSettings:
    vector_db: VectorDbSettings
    embeddings: EmbeddingSettings
    llm: LlmSettings
    ingestion: IngestionSettings
    retrieval: RetrievalSettings


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@lru_cache(maxsize=1)
def get_settings() -> RagSettings:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = _load_yaml(DEFAULT_CONFIG_PATH)
    rag = cfg.get("rag", {}) if isinstance(cfg.get("rag", {}), dict) else {}

    vector_cfg = rag.get("vector_db", {}) if isinstance(rag.get("vector_db", {}), dict) else {}
    emb_cfg = rag.get("embeddings", {}) if isinstance(rag.get("embeddings", {}), dict) else {}
    llm_cfg = rag.get("llm", {}) if isinstance(rag.get("llm", {}), dict) else {}
    ingestion_cfg = rag.get("ingestion", {}) if isinstance(rag.get("ingestion", {}), dict) else {}
    retrieval_cfg = rag.get("retrieval", {}) if isinstance(rag.get("retrieval", {}), dict) else {}

    vector_db = VectorDbSettings(
        provider=os.getenv("VECTOR_DB_PROVIDER", str(vector_cfg.get("provider", "qdrant"))),
        collection_name=os.getenv("QDRANT_COLLECTION", str(vector_cfg.get("collection_name", "research_chunks"))),
        distance=str(vector_cfg.get("distance", "cosine")),
        prefer_grpc=_as_bool(os.getenv("QDRANT_PREFER_GRPC"), bool(vector_cfg.get("prefer_grpc", False))),
    )

    embeddings = EmbeddingSettings(
        provider=os.getenv("EMBEDDING_PROVIDER", str(emb_cfg.get("provider", "huggingface"))),
        huggingface_model=os.getenv("HUGGINGFACE_EMBEDDING_MODEL", str(emb_cfg.get("huggingface_model", "sentence-transformers/all-MiniLM-L6-v2"))),
        openai_model=os.getenv("OPENAI_EMBEDDING_MODEL", str(emb_cfg.get("openai_model", "text-embedding-3-small"))),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", str(emb_cfg.get("batch_size", 64)))),
    )

    llm = LlmSettings(
        provider=os.getenv("LLM_PROVIDER", str(llm_cfg.get("provider", "google_genai"))),
        google_model=os.getenv("GOOGLE_GENAI_MODEL", str(llm_cfg.get("google_model", "gemini-2.0-flash"))),
        temperature=float(os.getenv("LLM_TEMPERATURE", str(llm_cfg.get("temperature", 0.1)))),
    )

    ingestion = IngestionSettings(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", str(ingestion_cfg.get("chunk_size", 1400)))),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", str(ingestion_cfg.get("chunk_overlap", 220)))),
        analyse_visuals=_as_bool(os.getenv("RAG_ANALYSE_VISUALS"), bool(ingestion_cfg.get("analyse_visuals", True))),
        max_visual_pages_per_document=int(
            os.getenv("RAG_MAX_VISUAL_PAGES_PER_DOC", str(ingestion_cfg.get("max_visual_pages_per_document", 20)))
        ),
    )

    retrieval = RetrievalSettings(
        top_k=int(os.getenv("RAG_TOP_K", str(retrieval_cfg.get("top_k", 6)))),
    )

    return RagSettings(
        vector_db=vector_db,
        embeddings=embeddings,
        llm=llm,
        ingestion=ingestion,
        retrieval=retrieval,
    )
