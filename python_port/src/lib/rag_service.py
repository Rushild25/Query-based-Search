from __future__ import annotations

import base64
import hashlib
import io
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
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient, models

from src.lib.settings import get_settings
from src.lib.utils import clean_string


_SETTINGS = get_settings()


class IngestState(TypedDict, total=False):
    pdf_bytes: bytes
    original_name: str
    session_id: str
    checksum: str
    duplicate: bool
    page_count: int
    text_chunks: int
    visual_chunks: int
    visual_pages_processed: int
    docs: list[Document]
    result: dict[str, Any]


class QaState(TypedDict, total=False):
    question: str
    session_id: str
    history: list[dict[str, Any]]
    top_k: int
    matches: list[dict[str, Any]]
    answer: str


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
    if provider == "openai":
        return OpenAIEmbeddings(model=_SETTINGS.embeddings.openai_model)
    return HuggingFaceEmbeddings(
        model_name=_SETTINGS.embeddings.huggingface_model,
        encode_kwargs={"batch_size": _SETTINGS.embeddings.batch_size},
    )


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    import os

    url = os.getenv("QDRANT_URL", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    if not url:
        raise RuntimeError("QDRANT_URL is required for cloud vector retrieval")
    return QdrantClient(url=url, api_key=api_key or None, prefer_grpc=_SETTINGS.vector_db.prefer_grpc)


def _ensure_collection(client: QdrantClient, embeddings) -> None:
    collection = _SETTINGS.vector_db.collection_name
    existing = {item.name for item in client.get_collections().collections}
    if collection in existing:
        client.create_payload_index(
            collection_name=collection,
            field_name="metadata.checksum",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection,
            field_name="metadata.session_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        return
    probe_vector = embeddings.embed_query("dimension probe")
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=len(probe_vector), distance=_distance_from_config()),
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="metadata.checksum",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="metadata.session_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


@lru_cache(maxsize=1)
def _bootstrap_vector_collection() -> None:
    client = _get_qdrant_client()
    embeddings = _get_embeddings()
    _ensure_collection(client, embeddings)


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


def _get_chat_model():
    import os

    provider = _SETTINGS.llm.provider.strip().lower()
    if provider == "google_genai":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        return ChatGoogleGenerativeAI(
            model=_SETTINGS.llm.google_model,
            temperature=_SETTINGS.llm.temperature,
            google_api_key=api_key,
        )
    return None


def _is_visual_page(page_text: str, image_count: int) -> bool:
    if image_count > 0:
        return True
    keywords = ["figure", "fig.", "table", "diagram", "plot", "chart", "graph", "pipeline", "architecture"]
    lowered = page_text.lower()
    return any(keyword in lowered for keyword in keywords)


def _page_image_to_base64(page: fitz.Page) -> str:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("utf-8")


def _describe_visual_page(page: fitz.Page, source_name: str, page_number: int) -> str:
    chat_model = _get_chat_model()
    if chat_model is None:
        return f"Visual content detected on page {page_number} of {source_name}. Configure GEMINI_API_KEY to enable detailed figure analysis."

    image_data = _page_image_to_base64(page)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Analyze this research-paper page and summarize all graphical, table, or diagram content. "
                    "Extract labels, trends, key comparisons, and the main takeaway in concise bullet points."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
    )
    try:
        response = chat_model.invoke([message])
        return str(getattr(response, "content", "")).strip() or f"Visual content detected on page {page_number} of {source_name}."
    except Exception:
        return f"Visual content detected on page {page_number} of {source_name}, but detailed visual analysis failed."


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
        session_id = str(state.get("session_id", "")).strip() or "default-session"
        if not pdf_bytes:
            raise ValueError("Empty PDF payload")

        checksum = hashlib.sha256(pdf_bytes).hexdigest()
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
        docs: list[Document] = []

        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_chunks = 0
        visual_chunks = 0
        visual_pages_processed = 0

        try:
            for page_index, page in enumerate(pdf, start=1):
                page_text = clean_string(page.get_text("text") or "").strip()
                if page_text:
                    for chunk in splitter.split_text(page_text):
                        docs.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "document_id": document_id,
                                    "session_id": session_id,
                                    "source_name": source_name,
                                    "original_name": original_name,
                                    "checksum": checksum,
                                    "page_number": page_index,
                                    "chunk_type": "text",
                                    "ingested_at": ingested_at,
                                },
                            )
                        )
                        text_chunks += 1

                if not _SETTINGS.ingestion.analyse_visuals:
                    continue

                if visual_pages_processed >= _SETTINGS.ingestion.max_visual_pages_per_document:
                    continue

                image_count = len(page.get_images(full=True))
                if _is_visual_page(page_text, image_count):
                    visual_summary = _describe_visual_page(page, source_name, page_index)
                    docs.append(
                        Document(
                            page_content=visual_summary,
                            metadata={
                                "document_id": document_id,
                                "session_id": session_id,
                                "source_name": source_name,
                                "original_name": original_name,
                                "checksum": checksum,
                                "page_number": page_index,
                                "chunk_type": "visual",
                                "ingested_at": ingested_at,
                            },
                        )
                    )
                    visual_chunks += 1
                    visual_pages_processed += 1
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
        }

    def upsert_vectors(state: IngestState) -> IngestState:
        if state.get("duplicate"):
            return state

        docs = state.get("docs", [])
        if not docs:
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
        vectorstore.add_documents(docs, ids=ids)

        first_meta = docs[0].metadata if docs else {}
        result = {
            "success": True,
            "duplicate": False,
            "message": f"Ingested {state.get('original_name', 'document.pdf')} into Qdrant",
            "document": {
                "document_id": first_meta.get("document_id", ""),
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
            "visual_chunks_created": state.get("visual_chunks", 0),
            "vector_backend": _SETTINGS.vector_db.provider,
            "vector_collection": _SETTINGS.vector_db.collection_name,
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
        session_id = str(state.get("session_id", "")).strip() or "default-session"
        if not question.strip():
            return {**state, "matches": []}

        top_k = int(state.get("top_k") or _SETTINGS.retrieval.top_k)
        vectorstore = _get_vectorstore()
        retrieved = vectorstore.similarity_search_with_score(
            question,
            k=top_k,
            filter=models.Filter(
                must=[models.FieldCondition(key="metadata.session_id", match=models.MatchValue(value=session_id))]
            ),
        )

        matches: list[dict[str, Any]] = []
        for document, score in retrieved:
            meta = document.metadata or {}
            matches.append(
                {
                    "document_id": meta.get("document_id", ""),
                    "source_name": meta.get("source_name", ""),
                    "original_name": meta.get("original_name", ""),
                    "page_number": meta.get("page_number"),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "content": document.page_content,
                    "score": float(score),
                }
            )
        return {**state, "matches": matches}

    def generate_answer(state: QaState) -> QaState:
        question = state.get("question", "")
        matches = state.get("matches", [])
        history = state.get("history", [])

        context = "\n\n".join(
            "\n".join(
                [
                    f"Source: {match.get('source_name', '')}",
                    f"Original file: {match.get('original_name', '')}",
                    f"Page: {match.get('page_number', 'unknown')}",
                    f"Type: {match.get('chunk_type', 'text')}",
                    f"Content: {match.get('content', '')}",
                ]
            )
            for match in matches
        )

        history_text = "\n".join(
            f"{item.get('role', 'user').title()}: {item.get('content', '')}" for item in history[-6:] if item.get("content")
        )

        chat_model = _get_chat_model()
        if chat_model is None:
            fallback = (
                "LLM is not configured. Retrieved context is provided in sources; set GEMINI_API_KEY (or another LLM provider) to enable final answer synthesis."
                if matches
                else "No relevant chunks found."
            )
            return {**state, "answer": fallback}

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a research RAG assistant. Answer only from retrieved context. "
                    "If information is missing, say so. Mention when evidence came from visual/diagram chunks.",
                ),
                (
                    "human",
                    "Conversation History:\n{history}\n\nQuestion:\n{question}\n\nRetrieved Context:\n{context}\n\n"
                    "Provide: (1) concise answer, (2) bullet list of evidence used.",
                ),
            ]
        )
        chain = prompt | chat_model | StrOutputParser()
        answer = chain.invoke(
            {
                "history": history_text or "No previous conversation.",
                "question": question,
                "context": context or "No relevant context retrieved.",
            }
        )
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

def ingest_pdf_bytes_for_session(pdf_bytes: bytes, original_name: str, session_id: str) -> dict[str, Any]:
    resolved_session = str(session_id).strip() or "default-session"
    state = _INGEST_GRAPH.invoke(
        {"pdf_bytes": pdf_bytes, "original_name": original_name, "session_id": resolved_session}
    )
    return state.get("result", {"success": False, "message": "Ingestion failed"})


def ingest_pdf_path(pdf_path: Path, session_id: str = "default-session") -> dict[str, Any]:
    return ingest_pdf_bytes_for_session(pdf_path.read_bytes(), pdf_path.name, session_id)


def list_documents() -> list[dict[str, Any]]:
    client = _ensure_vector_collection(_get_qdrant_client())
    docs_by_id: dict[str, dict[str, Any]] = {}
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=_SETTINGS.vector_db.collection_name,
            with_payload=True,
            with_vectors=False,
            offset=offset,
            limit=256,
        )
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            document_id = str(metadata.get("document_id", "")).strip()
            if not document_id:
                continue
            existing = docs_by_id.get(document_id)
            if existing is None:
                docs_by_id[document_id] = {
                    "document_id": document_id,
                    "source_name": metadata.get("source_name", ""),
                    "original_name": metadata.get("original_name", ""),
                    "checksum": metadata.get("checksum", ""),
                    "ingested_at": metadata.get("ingested_at", ""),
                    "chunk_count": 0,
                    "visual_chunk_count": 0,
                }
                existing = docs_by_id[document_id]
            existing["chunk_count"] += 1
            if metadata.get("chunk_type") == "visual":
                existing["visual_chunk_count"] += 1

        if offset is None:
            break

    return list(docs_by_id.values())


def answer_question(
    question: str,
    session_id: str,
    limit: int = 5,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_session = str(session_id).strip() or "default-session"
    qa_state = _QA_GRAPH.invoke(
        {
            "question": question,
            "session_id": resolved_session,
            "top_k": limit,
            "history": history or [],
        }
    )
    matches = qa_state.get("matches", [])
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
        "sources": matches,
        "source_overview": [{"document": key, "chunks": value} for key, value in source_overview.items()],
        "context": context_lines,
        "matched_chunks": len(matches),
        "retrieval_backend": f"{_SETTINGS.vector_db.provider}:{_SETTINGS.vector_db.collection_name}",
    }


