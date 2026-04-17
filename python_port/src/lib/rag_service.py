from __future__ import annotations

import base64
import hashlib
import importlib
import io
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from typing import Any, TypedDict

import fitz
from PIL import Image
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient, models

from src.lib.clip_backend import ClipTextEmbeddings, get_clip_backend
from src.lib.settings import get_settings
from src.lib.utils import clean_string
from dotenv import load_dotenv
load_dotenv()


_SETTINGS = get_settings()


def _normalize_session_id(session_id: str | None) -> str:
    value = str(session_id or "").strip()
    if not value:
        return "default-session"
    if value.lower() in {"string", "none", "null", "undefined"}:
        return "default-session"
    return value


def _normalize_doc_id(doc_id: str | None) -> str | None:
    value = str(doc_id or "").strip()
    if not value:
        return None
    if value.lower() in {"string", "none", "null", "undefined"}:
        return None
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.lower())
    return sanitized.strip("-_.") or None


def _derive_doc_id_from_name(original_name: str, checksum: str) -> str:
    stem = Path(original_name).stem
    normalized = _normalize_doc_id(stem)
    if normalized:
        return normalized
    return checksum[:16]


def _qdrant_client_timeout_seconds() -> int:
    raw = os.getenv("QDRANT_TIMEOUT_SECONDS", "120").strip()
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return 120


def _qdrant_upsert_timeout_seconds() -> int:
    raw = os.getenv("QDRANT_UPSERT_TIMEOUT_SECONDS", "180").strip()
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return 180


def _qdrant_upsert_batch_size() -> int:
    raw = os.getenv("QDRANT_UPSERT_BATCH_SIZE", "24").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 24


def _visual_embeddings_enabled() -> bool:
    return bool(_SETTINGS.visual_embeddings.enabled)


def _visual_collection_name() -> str:
    return _SETTINGS.visual_embeddings.collection_name


@lru_cache(maxsize=1)
def _get_clip_backend():
    if _SETTINGS.embeddings.provider.strip().lower() != "clip" and _SETTINGS.visual_embeddings.provider.strip().lower() != "clip":
        raise RuntimeError("CLIP backend requested but the active embedding provider is not set to clip")
    return get_clip_backend(_SETTINGS.visual_embeddings.model_name, _SETTINGS.visual_embeddings.pretrained)


@lru_cache(maxsize=1)
def _get_clip_text_embeddings() -> ClipTextEmbeddings:
    return ClipTextEmbeddings(_get_clip_backend())


def _clip_text_embedding(text: str) -> list[float]:
    return _get_clip_backend().embed_text(text)


def _clip_image_embedding(image: Image.Image) -> list[float]:
    return _get_clip_backend().embed_image(image.convert("RGB"))


def _render_page_image(page: fitz.Page, scale: float = 1.5) -> Image.Image:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")


def _page_vision_enabled() -> bool:
    return os.getenv("RAG_PAGE_VISION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}


def _page_vision_text_max_chars() -> int:
    raw = os.getenv("RAG_PAGE_VISION_TEXT_MAX_CHARS", "64").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 64


def _visual_page_content_max_chars() -> int:
    raw = os.getenv("RAG_VISUAL_PAGE_CONTENT_MAX_CHARS", "16000").strip()
    try:
        return max(2000, int(raw))
    except ValueError:
        return 16000


def _message_content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _should_run_page_vision(page_text: str) -> bool:
    if not _page_vision_enabled():
        return False
    if not os.getenv("GEMINI_API_KEY", "").strip():
        return False
    return len(page_text.strip()) <= _page_vision_text_max_chars()


def _transcribe_page_image_with_gemini(page_image: Image.Image) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return ""
    model_name = os.getenv("RAG_PAGE_VISION_MODEL", "").strip() or _SETTINGS.llm.google_model
    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=api_key,
    )
    buf = io.BytesIO()
    page_image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    prompt = (
        "You are preparing content for a search index (RAG). Read this PDF page image.\n"
        "Return plain UTF-8 text only, no markdown fences:\n"
        "1) Transcribe all readable text in natural reading order; use line breaks between blocks.\n"
        "2) Briefly describe non-text visuals (charts, diagrams, photos, logos) so they can be searched.\n"
        "If the page is blank, respond exactly: NO_VISIBLE_CONTENT"
    )
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    )
    response = model.invoke([msg])
    raw = clean_string(_message_content_to_str(getattr(response, "content", response))).strip()
    if raw.upper() in {"NO_VISIBLE_CONTENT", "NO VISIBLE CONTENT."}:
        return ""
    return raw


def _persist_rendered_page_image(
    image: Image.Image,
    session_id: str,
    doc_id: str,
    checksum: str,
    page_number: int,
) -> str:
    visual_root = _pdf_cache_root() / _safe_session_folder_name(session_id) / doc_id / "visual"
    visual_root.mkdir(parents=True, exist_ok=True)
    target = visual_root / f"{checksum}_page_{page_number}.png"
    if not target.exists():
        image.save(target, format="PNG")
    return str(target.resolve())


def _build_google_genai_model():
    import os

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=_SETTINGS.llm.google_model,
        temperature=_SETTINGS.llm.temperature,
        google_api_key=api_key,
    )


def _build_groq_model(model_name: str | None = None):
    import os

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    chat_groq_module = importlib.import_module("langchain_groq")
    ChatGroq = getattr(chat_groq_module, "ChatGroq")

    return ChatGroq(
        model=model_name or _SETTINGS.llm.groq_model,
        temperature=_SETTINGS.llm.temperature,
        api_key=api_key,
    )


class IngestState(TypedDict, total=False):
    pdf_bytes: bytes
    original_name: str
    source_path: str
    doc_id: str
    session_id: str
    checksum: str
    duplicate: bool
    page_count: int
    text_chunks: int
    visual_chunks: int
    visual_pages_processed: int
    docs: list[Document]
    visual_points: list[VisualPoint]
    result: dict[str, Any]


class QaState(TypedDict, total=False):
    question: str
    session_id: str
    doc_id: str
    history: list[dict[str, Any]]
    top_k: int
    matches: list[dict[str, Any]]
    answer: str


class VisualPoint(TypedDict, total=False):
    point_id: str
    vector: list[float]
    metadata: dict[str, Any]
    page_content: str


def _safe_session_folder_name(session_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in session_id) or "default-session"


@lru_cache(maxsize=1)
def _pdf_cache_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "pdf_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _persist_pdf_copy(
    pdf_bytes: bytes,
    session_id: str,
    checksum: str,
    original_name: str,
    source_path: str | None,
) -> str:
    if source_path:
        candidate = Path(source_path)
        if candidate.exists():
            return str(candidate.resolve())

    session_dir = _pdf_cache_root() / _safe_session_folder_name(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    target = session_dir / f"{checksum}.pdf"
    if not target.exists():
        target.write_bytes(pdf_bytes)
    return str(target.resolve())


def _distance_from_config() -> models.Distance:
    distance = _SETTINGS.vector_db.distance.strip().lower()
    if distance == "dot":
        return models.Distance.DOT
    if distance == "euclid":
        return models.Distance.EUCLID
    return models.Distance.COSINE


@lru_cache(maxsize=1)
def _get_embeddings():
    provider = _SETTINGS.embeddings.provider.strip().lower()
    if provider == "clip":
        return _get_clip_text_embeddings()
    if provider == "openai":
        return OpenAIEmbeddings(model=_SETTINGS.embeddings.openai_model)
    return HuggingFaceEmbeddings(
        model_name=_SETTINGS.embeddings.huggingface_model,
        encode_kwargs={"batch_size": _SETTINGS.embeddings.batch_size},
    )


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url:
        raise RuntimeError("QDRANT_URL is required for cloud vector retrieval")
    return QdrantClient(
        url=url,
        api_key=api_key or None,
        prefer_grpc=_SETTINGS.vector_db.prefer_grpc,
        timeout=_qdrant_client_timeout_seconds(),
    )


def _collection_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.checksum",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.session_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.doc_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="metadata.chunk_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def _ensure_collection(client: QdrantClient, embeddings, collection_name: str | None = None) -> None:
    collection = collection_name or _SETTINGS.vector_db.collection_name
    existing = {item.name for item in client.get_collections().collections}
    if collection in existing:
        _collection_payload_indexes(client, collection)
        return
    probe_vector = embeddings.embed_query("dimension probe")
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=len(probe_vector), distance=_distance_from_config()),
    )
    _collection_payload_indexes(client, collection)


def _collection_exists(client: QdrantClient, collection_name: str) -> bool:
    return collection_name in {item.name for item in client.get_collections().collections}


@lru_cache(maxsize=1)
def _bootstrap_vector_collection() -> None:
    client = _get_qdrant_client()
    embeddings = _get_embeddings()
    _ensure_collection(client, embeddings)


@lru_cache(maxsize=1)
def _bootstrap_visual_collection() -> None:
    client = _get_qdrant_client()
    embeddings = _get_clip_text_embeddings()
    _ensure_collection(client, embeddings, _visual_collection_name())


@lru_cache(maxsize=1)
def _get_vectorstore() -> QdrantVectorStore:
    _bootstrap_vector_collection()
    embeddings = _get_embeddings()
    client = _get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=_SETTINGS.vector_db.collection_name,
        embedding=embeddings,
    )


def _ensure_vector_collection(client: QdrantClient | None = None) -> QdrantClient:
    resolved_client = client or _get_qdrant_client()
    if client is None:
        _bootstrap_vector_collection()
    else:
        _ensure_collection(resolved_client, _get_embeddings())
    return resolved_client


def _ensure_visual_vector_collection(client: QdrantClient | None = None) -> QdrantClient:
    resolved_client = client or _get_qdrant_client()
    if not _visual_embeddings_enabled():
        return resolved_client
    if client is None:
        _bootstrap_visual_collection()
    else:
        _ensure_collection(resolved_client, _get_clip_text_embeddings(), _visual_collection_name())
    return resolved_client


def _get_chat_models() -> list[Any]:
    primary = _SETTINGS.llm.provider.strip().lower()
    fallback = _SETTINGS.llm.fallback_provider.strip().lower()
    models_list: list[Any] = []

    def build(provider_name: str):
        if provider_name == "google_genai":
            return _build_google_genai_model()
        if provider_name == "groq":
            return _build_groq_model()
        return None

    primary_model = build(primary)
    if primary_model is not None:
        models_list.append(primary_model)

    if fallback != primary:
        fallback_model = build(fallback)
        if fallback_model is not None:
            models_list.append(fallback_model)

    return models_list


def _document_exists(client: QdrantClient, checksum: str, session_id: str) -> bool:
    _ensure_vector_collection(client)
    count = client.count(
        collection_name=_SETTINGS.vector_db.collection_name,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(key="metadata.checksum", match=models.MatchValue(value=checksum)),
                models.FieldCondition(key="metadata.session_id", match=models.MatchValue(value=session_id)),
            ]
        ),
        exact=False,
    )
    return (count.count or 0) > 0


def _build_ingest_graph():
    def extract_chunks(state: IngestState) -> IngestState:
        pdf_bytes = state.get("pdf_bytes", b"")
        original_name = state.get("original_name", "document.pdf")
        source_path = str(state.get("source_path", "") or "").strip()
        session_id = _normalize_session_id(str(state.get("session_id", "")))
        if not pdf_bytes:
            raise ValueError("Empty PDF payload")

        checksum = hashlib.sha256(pdf_bytes).hexdigest()
        doc_id = _normalize_doc_id(str(state.get("doc_id", ""))) or _derive_doc_id_from_name(original_name, checksum)
        client = _ensure_vector_collection(_get_qdrant_client())
        if _document_exists(client, checksum, session_id):
            return {
                **state,
                "checksum": checksum,
                "duplicate": True,
                "docs": [],
                "result": {
                    "success": True,
                    "duplicate": True,
                    "message": "Document already exists in cloud vector DB",
                    "document": {
                        "doc_id": doc_id,
                        "checksum": checksum,
                        "original_name": original_name,
                        "session_id": session_id,
                    },
                    "chunks_created": 0,
                    "visual_chunks_created": 0,
                },
            }

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=_SETTINGS.ingestion.chunk_size,
            chunk_overlap=_SETTINGS.ingestion.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        document_id = uuid.uuid4().hex
        source_name = Path(original_name).stem or original_name
        ingested_at = datetime.now(timezone.utc).isoformat()
        stored_pdf_path = _persist_pdf_copy(pdf_bytes, session_id, checksum, original_name, source_path)
        docs: list[Document] = []
        visual_points: list[VisualPoint] = []

        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_chunks = 0
        visual_chunks = 0
        visual_pages_processed = 0

        try:
            for page_index, page in enumerate(pdf, start=1):
                page_text = clean_string(page.get_text("text") or "").strip()
                run_page_vision = _should_run_page_vision(page_text)
                render_scale = 2.0 if run_page_vision else 1.5

                if page_text:
                    for chunk in splitter.split_text(page_text):
                        docs.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "document_id": document_id,
                                    "doc_id": doc_id,
                                    "session_id": session_id,
                                    "source_name": source_name,
                                    "original_name": original_name,
                                    "checksum": checksum,
                                    "pdf_path": stored_pdf_path,
                                    "page_number": page_index,
                                    "chunk_type": "text",
                                    "ingested_at": ingested_at,
                                },
                            )
                        )
                        text_chunks += 1

                max_visual_pages = _SETTINGS.visual_embeddings.max_visual_pages_per_document
                want_clip_visual = _visual_embeddings_enabled() and visual_pages_processed < max_visual_pages
                want_gemini_page = run_page_vision
                if want_clip_visual or want_gemini_page:
                    try:
                        page_image = _render_page_image(page, scale=render_scale)

                        vision_transcript = ""
                        if want_gemini_page:
                            try:
                                vision_transcript = clean_string(_transcribe_page_image_with_gemini(page_image)).strip()
                            except Exception:
                                vision_transcript = ""

                        if want_gemini_page and vision_transcript:
                            for chunk in splitter.split_text(vision_transcript):
                                docs.append(
                                    Document(
                                        page_content=chunk,
                                        metadata={
                                            "document_id": document_id,
                                            "doc_id": doc_id,
                                            "session_id": session_id,
                                            "source_name": source_name,
                                            "original_name": original_name,
                                            "checksum": checksum,
                                            "pdf_path": stored_pdf_path,
                                            "page_number": page_index,
                                            "chunk_type": "from_image",
                                            "ingested_at": ingested_at,
                                        },
                                    )
                                )
                                text_chunks += 1

                        if want_clip_visual:
                            image_path = _persist_rendered_page_image(page_image, session_id, doc_id, checksum, page_index)
                            merged = "\n\n".join(part for part in (page_text, vision_transcript) if part).strip()
                            visual_text = clean_string(
                                merged or f"Visual page {page_index} from {source_name}"
                            ).strip()
                            cap = _visual_page_content_max_chars()
                            if len(visual_text) > cap:
                                visual_text = visual_text[:cap].rstrip() + "..."
                            visual_points.append(
                                {
                                    "point_id": uuid.uuid4().hex,
                                    "vector": _clip_image_embedding(page_image),
                                    "page_content": visual_text,
                                    "metadata": {
                                        "document_id": document_id,
                                        "doc_id": doc_id,
                                        "session_id": session_id,
                                        "source_name": source_name,
                                        "original_name": original_name,
                                        "checksum": checksum,
                                        "pdf_path": stored_pdf_path,
                                        "image_path": image_path,
                                        "page_number": page_index,
                                        "chunk_type": "visual",
                                        "ingested_at": ingested_at,
                                    },
                                }
                            )
                            visual_chunks += 1
                            visual_pages_processed += 1
                    except Exception:
                        continue
        finally:
            page_count = pdf.page_count
            pdf.close()

        return {
            **state,
            "checksum": checksum,
            "duplicate": False,
            "page_count": page_count,
            "text_chunks": text_chunks,
            "visual_chunks": visual_chunks,
            "visual_pages_processed": visual_pages_processed,
            "docs": docs,
            "visual_points": visual_points,
        }

    def upsert_vectors(state: IngestState) -> IngestState:
        if state.get("duplicate"):
            return state

        docs = state.get("docs", [])
        visual_points = state.get("visual_points", [])
        if not docs and not visual_points:
            return {
                **state,
                "result": {
                    "success": False,
                    "duplicate": False,
                    "message": "No content could be extracted from the PDF",
                    "chunks_created": 0,
                    "visual_chunks_created": 0,
                },
            }

        vectorstore = _get_vectorstore()
        ids = [uuid.uuid4().hex for _ in docs]
        batch_size = _qdrant_upsert_batch_size()
        upsert_timeout = _qdrant_upsert_timeout_seconds()

        for start in range(0, len(docs), batch_size):
            batch_docs = docs[start : start + batch_size]
            batch_ids = ids[start : start + batch_size]
            attempts = 0
            while True:
                attempts += 1
                try:
                    vectorstore.add_documents(batch_docs, ids=batch_ids, timeout=upsert_timeout)
                    break
                except Exception:
                    if attempts >= 2:
                        raise

        if visual_points and _visual_embeddings_enabled():
            client = _ensure_visual_vector_collection(_get_qdrant_client())
            visual_collection = _visual_collection_name()
            batch_size = _qdrant_upsert_batch_size()
            for start in range(0, len(visual_points), batch_size):
                batch = visual_points[start : start + batch_size]
                client.upsert(
                    collection_name=visual_collection,
                    points=[
                        models.PointStruct(
                            id=point["point_id"],
                            vector=point["vector"],
                            payload={
                                "metadata": point["metadata"],
                                "page_content": point["page_content"],
                            },
                        )
                        for point in batch
                    ],
                    wait=True,
                )

        first_meta: dict[str, Any] = {}
        if docs:
            first_meta = dict(docs[0].metadata)
        elif visual_points:
            first_meta = dict(visual_points[0].get("metadata") or {})

        result = {
            "success": True,
            "duplicate": False,
            "message": f"Ingested {state.get('original_name', 'document.pdf')} into Qdrant",
            "document": {
                "document_id": first_meta.get("document_id", ""),
                "doc_id": first_meta.get("doc_id", ""),
                "session_id": first_meta.get("session_id", ""),
                "source_name": first_meta.get("source_name", ""),
                "original_name": first_meta.get("original_name", ""),
                "checksum": state.get("checksum", ""),
                "page_count": state.get("page_count", 0),
                "chunk_count": state.get("text_chunks", 0),
                "visual_chunk_count": state.get("visual_chunks", 0),
                "ingested_at": first_meta.get("ingested_at", ""),
            },
            "chunks_created": len(docs),
            "visual_chunks_created": len(visual_points),
            "vector_backend": _SETTINGS.vector_db.provider,
            "vector_collection": _SETTINGS.vector_db.collection_name,
            "visual_vector_collection": _visual_collection_name() if _visual_embeddings_enabled() else None,
        }
        return {**state, "result": result}

    builder = StateGraph(IngestState)
    builder.add_node("extract_chunks", extract_chunks)
    builder.add_node("upsert_vectors", upsert_vectors)
    builder.add_edge(START, "extract_chunks")
    builder.add_edge("extract_chunks", "upsert_vectors")
    builder.add_edge("upsert_vectors", END)
    return builder.compile()


def _build_qa_graph():
    def retrieve(state: QaState) -> QaState:
        question = state.get("question", "")
        session_id = _normalize_session_id(str(state.get("session_id", "")))
        doc_id = _normalize_doc_id(str(state.get("doc_id", "")))
        if not question.strip():
            return {**state, "matches": []}

        top_k = int(state.get("top_k") or _SETTINGS.retrieval.top_k)
        text_collection = _SETTINGS.vector_db.collection_name
        visual_collection = _visual_collection_name()
        client = _get_qdrant_client()
        query_vector = _clip_text_embedding(question) if _SETTINGS.embeddings.provider.strip().lower() == "clip" else _get_embeddings().embed_query(question)
        must_conditions: list[models.FieldCondition] = [
            models.FieldCondition(key="metadata.session_id", match=models.MatchValue(value=session_id))
        ]
        if doc_id:
            must_conditions.append(models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id)))

        matches: list[dict[str, Any]] = []
        search_limit = max(top_k * 2, 8)
        text_hits = client.query_points(
            collection_name=text_collection,
            query=query_vector,
            query_filter=models.Filter(must=must_conditions),
            limit=search_limit,
            with_payload=True,
            with_vectors=False,
        ).points
        visual_hits: list[Any] = []
        if _visual_embeddings_enabled() and _collection_exists(client, visual_collection):
            visual_hits = client.query_points(
                collection_name=visual_collection,
                query=query_vector,
                query_filter=models.Filter(must=must_conditions),
                limit=search_limit,
                with_payload=True,
                with_vectors=False,
            ).points

        def to_match(hit: Any, modality: str) -> dict[str, Any]:
            payload = hit.payload or {}
            metadata = payload.get("metadata") or {}
            return {
                "document_id": metadata.get("document_id", ""),
                "doc_id": metadata.get("doc_id", ""),
                "source_name": metadata.get("source_name", ""),
                "original_name": metadata.get("original_name", ""),
                "pdf_path": metadata.get("pdf_path", ""),
                "image_path": metadata.get("image_path", ""),
                "page_number": metadata.get("page_number"),
                "chunk_type": metadata.get("chunk_type", modality),
                "content": payload.get("page_content", ""),
                "score": float(hit.score or 0.0),
                "modality": modality,
            }

        text_matches = [to_match(hit, "text") for hit in text_hits]
        visual_matches = [to_match(hit, "visual") for hit in visual_hits]

        def _dedupe_key(match: dict[str, Any]) -> tuple[Any, ...]:
            return (
                _normalize_doc_id(str(match.get("doc_id", ""))),
                match.get("page_number"),
                match.get("chunk_type"),
                str(match.get("content", ""))[:250],
            )

        merged: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        text_quota = max(1, top_k // 2)
        visual_quota = max(1, top_k - text_quota)

        for candidate in text_matches[:text_quota] + visual_matches[:visual_quota]:
            key = _dedupe_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)

        if len(merged) < top_k:
            for candidate in sorted(text_matches + visual_matches, key=lambda item: item.get("score", 0.0), reverse=True):
                key = _dedupe_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)
                if len(merged) >= top_k:
                    break

        matches = merged[:top_k]
        return {**state, "matches": matches}

    def generate_answer(state: QaState) -> QaState:
        question = state.get("question", "")
        matches = state.get("matches", [])
        history = state.get("history", [])
        text_matches = [match for match in matches if match.get("modality") != "visual"]
        visual_matches = [match for match in matches if match.get("modality") == "visual"]

        text_context = "\n\n".join(
            "\n".join(
                [
                    f"Source: {match.get('source_name', '')}",
                    f"Original file: {match.get('original_name', '')}",
                    f"Page: {match.get('page_number', 'unknown')}",
                    f"Type: {match.get('chunk_type', 'text')}",
                    f"Content: {match.get('content', '')}",
                ]
            )
            for match in text_matches
        )
        visual_context = "\n\n".join(
            "\n".join(
                [
                    f"Source: {match.get('source_name', '')}",
                    f"Original file: {match.get('original_name', '')}",
                    f"Page: {match.get('page_number', 'unknown')}",
                    f"Type: {match.get('chunk_type', 'visual')}",
                    f"Content: {match.get('content', '')}",
                ]
            )
            for match in visual_matches
        )

        combined_context_parts = []
        if text_context:
            combined_context_parts.append(f"Retrieved Text Chunks:\n{text_context}")
        if visual_context:
            combined_context_parts.append(f"Retrieved Visual Pages:\n{visual_context}")
        combined_context = "\n\n".join(combined_context_parts)

        history_text = "\n".join(
            f"{item.get('role', 'user').title()}: {item.get('content', '')}" for item in history[-6:] if item.get("content")
        )

        chat_models = _get_chat_models()
        if not chat_models:
            fallback = (
                "LLM is not configured. Retrieved context is provided in sources; set GEMINI_API_KEY (primary) or GROQ_API_KEY (fallback) to enable final answer synthesis."
                if matches
                else "No relevant chunks found."
            )
            return {**state, "answer": fallback}

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research RAG assistant. Answer only from retrieved context. "
                    "If information is missing, say so. Synthesize across text chunks, transcribed page images (chunk_type from_image), "
                    "and visual page notes; mention disagreements when sources conflict.",
                ),
                (
                    "human",
                    "Conversation History:\n{history}\n\nQuestion:\n{question}\n\nRetrieved Context:\n{context}\n\n"
                    "Provide: (1) concise answer, (2) bullet list of evidence used.",
                ),
            ]
        )
        payload = {
            "history": history_text or "No previous conversation.",
            "question": question,
            "context": combined_context or "No relevant context retrieved.",
        }

        answer = ""
        last_error: Exception | None = None
        for chat_model in chat_models:
            try:
                chain = prompt | chat_model | StrOutputParser()
                candidate = chain.invoke(payload)
                answer = str(candidate).strip()
                if answer:
                    break
            except Exception as exc:
                last_error = exc
                continue

        if not answer:
            if last_error is not None:
                raise RuntimeError(f"All configured LLM providers failed: {last_error}") from last_error
            raise RuntimeError("All configured LLM providers failed to generate an answer")

        return {**state, "answer": answer.strip()}

    builder = StateGraph(QaState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate_answer", generate_answer)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile()


_INGEST_GRAPH = _build_ingest_graph()
_QA_GRAPH = _build_qa_graph()

def ingest_pdf_bytes_for_session(
    pdf_bytes: bytes,
    original_name: str,
    session_id: str,
    source_path: str | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    resolved_session = _normalize_session_id(session_id)
    resolved_doc_id = _normalize_doc_id(doc_id)
    state = _INGEST_GRAPH.invoke(
        {
            "pdf_bytes": pdf_bytes,
            "original_name": original_name,
            "source_path": source_path or "",
            "doc_id": resolved_doc_id or "",
            "session_id": resolved_session,
        }
    )
    return state.get("result", {"success": False, "message": "Ingestion failed"})


def ingest_pdf_path(pdf_path: Path, session_id: str = "default-session") -> dict[str, Any]:
    return ingest_pdf_bytes_for_session(
        pdf_path.read_bytes(),
        pdf_path.name,
        session_id,
        source_path=str(pdf_path.resolve()),
        doc_id=Path(pdf_path.name).stem,
    )


def _merge_document_rows(target: dict[str, dict[str, Any]], point: Any, is_visual_collection: bool) -> None:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    document_id = str(metadata.get("document_id", "")).strip()
    if not document_id:
        return

    existing = target.get(document_id)
    if existing is None:
        existing = {
            "document_id": document_id,
            "doc_id": metadata.get("doc_id", ""),
            "session_id": metadata.get("session_id", ""),
            "source_name": metadata.get("source_name", ""),
            "original_name": metadata.get("original_name", ""),
            "checksum": metadata.get("checksum", ""),
            "ingested_at": metadata.get("ingested_at", ""),
            "chunk_count": 0,
            "visual_chunk_count": 0,
        }
        target[document_id] = existing

    existing["chunk_count"] += 1
    if is_visual_collection or metadata.get("chunk_type") == "visual":
        existing["visual_chunk_count"] += 1


def list_documents(session_id: str | None = None) -> list[dict[str, Any]]:
    client = _ensure_vector_collection(_get_qdrant_client())
    docs_by_id: dict[str, dict[str, Any]] = {}
    resolved_session = _normalize_session_id(session_id) if session_id is not None else None

    scroll_filter = None
    if resolved_session:
        scroll_filter = models.Filter(
            must=[models.FieldCondition(key="metadata.session_id", match=models.MatchValue(value=resolved_session))]
        )

    for collection_name, is_visual_collection in [
        (_SETTINGS.vector_db.collection_name, False),
        (_visual_collection_name(), True),
    ]:
        if is_visual_collection and not _collection_exists(client, collection_name):
            continue
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                scroll_filter=scroll_filter,
                offset=offset,
                limit=256,
            )
            for point in points:
                _merge_document_rows(docs_by_id, point, is_visual_collection)

            if offset is None:
                break

    return list(docs_by_id.values())


def clear_documents(session_id: str, doc_id: str | None = None) -> dict[str, Any]:
    resolved_session = _normalize_session_id(session_id)
    resolved_doc_id = _normalize_doc_id(doc_id)

    must_conditions: list[models.FieldCondition] = [
        models.FieldCondition(key="metadata.session_id", match=models.MatchValue(value=resolved_session))
    ]
    if resolved_doc_id:
        must_conditions.append(models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=resolved_doc_id)))

    client = _get_qdrant_client()
    target_filter = models.Filter(must=must_conditions)
    before = 0
    after = 0
    for collection_name in [_SETTINGS.vector_db.collection_name, _visual_collection_name()]:
        if collection_name != _SETTINGS.vector_db.collection_name and not _collection_exists(client, collection_name):
            continue
        try:
            before += client.count(
                collection_name=collection_name,
                count_filter=target_filter,
                exact=False,
            ).count or 0
            client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=target_filter),
                wait=True,
            )
            after += client.count(
                collection_name=collection_name,
                count_filter=target_filter,
                exact=False,
            ).count or 0
        except Exception:
            continue

    return {
        "success": True,
        "session_id": resolved_session,
        "doc_id": resolved_doc_id,
        "points_before": before,
        "points_after": after,
        "points_deleted": max(0, before - after),
    }


def answer_question(
    question: str,
    session_id: str,
    doc_id: str | None = None,
    limit: int = 5,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_session = _normalize_session_id(session_id)
    resolved_doc_id = _normalize_doc_id(doc_id)
    qa_state = _QA_GRAPH.invoke(
        {
            "question": question,
            "session_id": resolved_session,
            "doc_id": resolved_doc_id or "",
            "top_k": limit,
            "history": history or [],
        }
    )
    matches = qa_state.get("matches", [])
    text_match_count = sum(1 for match in matches if match.get("modality") != "visual")
    visual_match_count = sum(1 for match in matches if match.get("modality") == "visual")
    context_lines = [
        "\n".join(
            [
                f"[{idx}] Source: {match.get('source_name', '')}",
                f"Original file: {match.get('original_name', '')}",
                f"Page: {match.get('page_number', 'unknown')}",
                f"Type: {match.get('chunk_type', 'text')}",
                f"Content: {match.get('content', '')}",
            ]
        )
        for idx, match in enumerate(matches, start=1)
    ]

    source_overview: dict[str, int] = {}
    for match in matches:
        label = str(match.get("original_name") or match.get("source_name") or "unknown")
        source_overview[label] = source_overview.get(label, 0) + 1

    return {
        "answer": qa_state.get("answer", "No answer generated."),
        "session_id": resolved_session,
        "doc_id": resolved_doc_id,
        "sources": matches,
        "source_overview": [{"document": key, "chunks": value} for key, value in source_overview.items()],
        "context": context_lines,
        "matched_chunks": len(matches),
        "text_matches": text_match_count,
        "visual_matches": visual_match_count,
        "retrieval_backend": f"{_SETTINGS.vector_db.provider}:{_SETTINGS.vector_db.collection_name}",
        "visual_retrieval_backend": _visual_collection_name() if _visual_embeddings_enabled() else None,
    }


