from __future__ import annotations

import json
from typing import Any

import requests
import streamlit as st


st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")


DEFAULT_BASE_URL = "http://127.0.0.1:8010"


def _normalize_base_url(raw: str) -> str:
    return raw.strip().rstrip("/")


def _api_url(base_url: str, path: str) -> str:
    return f"{_normalize_base_url(base_url)}{path}"


def _safe_error_text(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict) and "detail" in payload:
            return str(payload["detail"])
        return json.dumps(payload)
    except Exception:
        return response.text or f"HTTP {response.status_code}"


def _ingest_pdf(
    base_url: str,
    session_id: str,
    file_name: str,
    file_bytes: bytes,
    doc_id: str | None = None,
) -> dict[str, Any]:
    url = _api_url(base_url, "/api/api/ingest/pdf")
    files = {"file": (file_name, file_bytes, "application/pdf")}
    data = {"session_id": session_id}
    if doc_id:
        data["doc_id"] = doc_id
    response = requests.post(url, files=files, data=data, timeout=300)
    if response.status_code >= 400:
        raise RuntimeError(_safe_error_text(response))
    return response.json()


def _fetch_documents(base_url: str, session_id: str) -> dict[str, Any]:
    url = _api_url(base_url, "/api/api/ingest/documents")
    params = {"session_id": session_id}
    response = requests.get(url, params=params, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(_safe_error_text(response))
    return response.json()


def _chat(
    base_url: str,
    session_id: str,
    top_k: int,
    messages: list[dict[str, str]],
    doc_id: str | None,
) -> dict[str, Any]:
    url = _api_url(base_url, "/api/api/chat")
    payload = {
        "messages": messages,
        "session_id": session_id,
        "doc_id": doc_id,
        "top_k": top_k,
    }
    response = requests.post(url, json=payload, timeout=300)
    if response.status_code >= 400:
        raise RuntimeError(_safe_error_text(response))
    return response.json()


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = ""


st.title("PDF RAG Frontend")
st.caption("Upload PDFs, ingest into the current session, and chat against hybrid text + visual retrieval.")

with st.sidebar:
    st.header("Settings")
    base_url = st.text_input("Backend URL", value=DEFAULT_BASE_URL, help="FastAPI base URL")
    session_id = st.text_input("Session ID", value="default-session")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
    doc_id_input = st.text_input(
        "Document ID (doc_id)",
        value=st.session_state.active_doc_id,
        help="Restrict retrieval to one document.",
    )
    st.session_state.active_doc_id = doc_id_input.strip()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []

    st.divider()
    st.subheader("Indexed Documents")
    if st.button("Refresh Documents", use_container_width=True):
        try:
            docs_payload = _fetch_documents(base_url, session_id)
            docs = docs_payload.get("documents", []) if isinstance(docs_payload, dict) else []
            st.success(f"Found {len(docs)} documents")
            st.session_state.docs_preview = docs
        except Exception as exc:
            st.error(str(exc))

    docs_preview = st.session_state.get("docs_preview", [])
    if docs_preview:
        for item in docs_preview[:10]:
            name = item.get("doc_id") or item.get("original_name") or item.get("source_name") or "unknown"
            chunks = item.get("chunk_count", 0)
            visual_chunks = item.get("visual_chunk_count", 0)
            st.caption(f"- {name} ({chunks} chunks, {visual_chunks} visual)")

upload_col, ingest_col = st.columns([2, 1])
with upload_col:
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
with ingest_col:
    st.write("")
    st.write("")
    ingest_clicked = st.button("Ingest PDF", use_container_width=True)

if ingest_clicked:
    if uploaded_pdf is None:
        st.warning("Choose a PDF file first.")
    else:
        with st.spinner("Ingesting PDF..."):
            try:
                result = _ingest_pdf(
                    base_url=base_url,
                    session_id=session_id,
                    file_name=uploaded_pdf.name,
                    file_bytes=uploaded_pdf.getvalue(),
                    doc_id=st.session_state.active_doc_id or None,
                )
                created = result.get("chunks_created", 0)
                created_doc_id = str(result.get("document", {}).get("doc_id", "")).strip()
                if created_doc_id:
                    st.session_state.active_doc_id = created_doc_id
                st.success(f"Ingest complete. Chunks created: {created}")
                st.json(result)
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")

st.divider()
st.subheader("Chat")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask about the ingested PDF...")
if user_prompt:
    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = _chat(
                    base_url=base_url,
                    session_id=session_id,
                    top_k=top_k,
                    messages=st.session_state.chat_messages,
                    doc_id=st.session_state.active_doc_id or None,
                )
                reply = str(result.get("reply", "No reply generated."))
                st.markdown(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

                with st.expander("Debug: Retrieval details", expanded=False):
                    st.json(
                        {
                            "matched_chunks": result.get("matched_chunks", 0),
                            "text_matches": result.get("text_matches", 0),
                            "visual_matches": result.get("visual_matches", 0),
                            "doc_id": result.get("doc_id"),
                            "source_overview": result.get("source_overview", []),
                            "retrieval_backend": result.get("retrieval_backend", ""),
                            "visual_retrieval_backend": result.get("visual_retrieval_backend", ""),
                        }
                    )
            except Exception as exc:
                error_text = f"Chat failed: {exc}"
                st.error(error_text)
                st.session_state.chat_messages.append({"role": "assistant", "content": error_text})
