from __future__ import annotations

from googleapiclient.discovery import build

from src.lib.google.auth_server import get_google_auth_client


def list_google_docs(access_token: str) -> list[dict]:
    auth = get_google_auth_client(access_token)
    drive = build("drive", "v3", credentials=auth)
    response = drive.files().list(
        q="mimeType='application/vnd.google-apps.document'",
        fields="files(id, name, mimeType, modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=50,
    ).execute()
    return response.get("files", [])


def get_google_doc_content(access_token: str, doc_id: str) -> str:
    auth = get_google_auth_client(access_token)
    docs = build("docs", "v1", credentials=auth)
    response = docs.documents().get(documentId=doc_id).execute()
    body = (response.get("body") or {}).get("content") or []
    chunks: list[str] = []
    for element in body:
        paragraph = element.get("paragraph") or {}
        for para_element in paragraph.get("elements") or []:
            text_run = para_element.get("textRun") or {}
            if text_run.get("content"):
                chunks.append(text_run["content"])
    return "".join(chunks)
