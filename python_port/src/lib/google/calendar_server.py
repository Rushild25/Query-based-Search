from __future__ import annotations

from datetime import datetime, timedelta

from googleapiclient.discovery import build

from src.lib.google.auth_server import get_google_auth_client


def list_google_calendar_events(
    access_token: str,
    calendar_id: str = "primary",
    time_min: str | None = None,
    time_max: str | None = None,
) -> list[dict]:
    time_min = time_min or datetime.utcnow().isoformat() + "Z"
    time_max = time_max or (datetime.utcnow() + timedelta(days=90)).isoformat() + "Z"
    auth = get_google_auth_client(access_token)
    calendar = build("calendar", "v3", credentials=auth)
    response = calendar.events().list(
        calendarId=calendar_id,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
        maxResults=100,
    ).execute()
    return response.get("items", [])


def list_google_calendars(access_token: str) -> list[dict]:
    auth = get_google_auth_client(access_token)
    calendar = build("calendar", "v3", credentials=auth)
    response = calendar.calendarList().list().execute()
    return response.get("items", [])


def calendar_events_to_text(events: list[dict]) -> str:
    if not events:
        return ""
    lines: list[str] = []
    for event in events:
        lines.append(f"Event: {event.get('summary', '')}")
        if event.get("description"):
            lines.append(f"Description: {event['description']}")
        start = ((event.get("start") or {}).get("dateTime")) or ""
        end = ((event.get("end") or {}).get("dateTime")) or ""
        if start:
            lines.append(f"Start: {start}")
        if end:
            lines.append(f"End: {end}")
        attendees = event.get("attendees") or []
        if attendees:
            names = [a.get("displayName") or a.get("email", "") for a in attendees]
            lines.append("Attendees: " + ", ".join(x for x in names if x))
        lines.append("")
    return "\n".join(lines)
