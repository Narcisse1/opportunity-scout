"""Exports for the application pipeline: iCalendar deadlines and CSV."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Any

from .opportunities import parse_date

CSV_COLUMNS = [
    "status", "name", "organizer", "type", "deadline", "score", "verdict",
    "funding_or_prize", "url", "notes",
]  # fmt: skip


def _ics_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
    )


def _fold(line: str) -> str:
    """RFC 5545 line folding: max 75 octets per line, continuation lines start with a space."""
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    parts, start = [], 0
    limit = 75
    while start < len(raw):
        end = min(start + limit, len(raw))
        while end < len(raw) and (raw[end] & 0xC0) == 0x80:  # don't split a UTF-8 character
            end -= 1
        parts.append(raw[start:end].decode("utf-8"))
        start, limit = end, 74
    return "\r\n ".join(parts)


def to_ics(records: list[dict[str, Any]], now: datetime | None = None) -> str:
    """One all-day event per deadline, with a reminder a week before."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Opportunity Scout//Application deadlines//EN",
        "CALSCALE:GREGORIAN",
        "X-WR-CALNAME:Funding application deadlines",
    ]
    for rec in records:
        deadline = parse_date(rec.get("deadline"))
        if deadline is None:
            continue
        opp = rec.get("opportunity") or {}
        description = "\n".join(
            part
            for part in (
                f"Status: {rec.get('status', 'saved')}",
                f"Fit score: {rec.get('score')}/100" if rec.get("score") is not None else "",
                f"Offer: {opp.get('funding_or_prize')}" if opp.get("funding_or_prize") else "",
                rec.get("url") or "",
                f"Notes: {rec['notes']}" if rec.get("notes") else "",
            )
            if part
        )
        lines += [
            "BEGIN:VEVENT",
            f"UID:{rec.get('key', rec['name'])}@opportunity-scout",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{deadline:%Y%m%d}",
            f"DTEND;VALUE=DATE:{deadline + timedelta(days=1):%Y%m%d}",
            f"SUMMARY:{_ics_escape('Deadline: ' + rec['name'])}",
            f"DESCRIPTION:{_ics_escape(description)}",
        ]
        if rec.get("url"):
            lines.append(f"URL:{rec['url']}")
        lines += [
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            f"DESCRIPTION:{_ics_escape(rec['name'] + ' closes in 7 days')}",
            "TRIGGER:-P7D",
            "END:VALARM",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"


def to_csv(records: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        opp = rec.get("opportunity") or {}
        writer.writerow({**rec, "funding_or_prize": opp.get("funding_or_prize", "")})
    return out.getvalue()
