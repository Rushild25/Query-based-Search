from __future__ import annotations

from googleapiclient.discovery import build

from src.lib.google.auth_server import get_google_auth_client


def list_google_sheets(access_token: str) -> list[dict]:
    auth = get_google_auth_client(access_token)
    drive = build("drive", "v3", credentials=auth)
    response = drive.files().list(
        q="mimeType='application/vnd.google-apps.spreadsheet'",
        fields="files(id, name, mimeType, modifiedTime)",
        orderBy="modifiedTime desc",
        pageSize=50,
    ).execute()
    return response.get("files", [])


def get_google_sheet_data(access_token: str, sheet_id: str) -> list[list[str]]:
    auth = get_google_auth_client(access_token)
    sheets = build("sheets", "v4", credentials=auth)
    meta = sheets.spreadsheets().get(spreadsheetId=sheet_id, fields="sheets.properties").execute()
    titles = [
        (sheet.get("properties") or {}).get("title")
        for sheet in (meta.get("sheets") or [])
        if (sheet.get("properties") or {}).get("title")
    ]
    data: list[list[str]] = []
    for title in titles:
        response = sheets.spreadsheets().values().get(spreadsheetId=sheet_id, range=title).execute()
        values = response.get("values") or []
        data.extend(values)
    return data


def sheet_data_to_text(data: list[list[str]]) -> str:
    if not data:
        return ""
    headers = data[0]
    rows = data[1:]
    lines: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        parts = []
        for col_index, header in enumerate(headers):
            if col_index < len(row) and row[col_index]:
                parts.append(f"{header}: {row[col_index]}")
        lines.append(f"Row {row_index}: " + ". ".join(parts))
    return "\n".join(lines)
