"""Weekly email digest: new programs that fit the company, plus deadlines coming up.

Rendered as table-based HTML with inline styles (what email clients support) and a plain-text
alternative. All scraped values are escaped: program pages are untrusted input.
"""

from __future__ import annotations

from datetime import date
from html import escape
from typing import Any

from .opportunities import days_left

INK = "#16233b"
MUTED = "#5b6577"
LINE = "#e4e7ee"
ACCENT = "#1f4e8c"


def _deadline_text(deadline: str | None, today: date) -> str:
    left = days_left(deadline, today)
    if left is None:
        return "Rolling / not stated"
    return f"{deadline} ({left} day{'s' if left != 1 else ''} left)"


def _program_block(r: dict[str, Any], today: date) -> str:
    opp = r.get("opportunity") or {}
    fit = r.get("fit") or {}
    reasons = "".join(
        f'<li style="margin:0 0 4px">{escape(x)}</li>' for x in (fit.get("reasons") or [])[:3]
    )
    blockers = fit.get("eligibility_blockers") or []
    blocker = (
        f'<p style="margin:8px 0 0;color:#9a3b1b;font-size:13px">Check first: '
        f"{escape('; '.join(blockers[:2]))}</p>"
        if blockers
        else ""
    )
    prize = opp.get("funding_or_prize")
    meta = " &middot; ".join(
        escape(x)
        for x in [
            (r.get("type") or "").replace("_", " ").capitalize(),
            opp.get("organizer") or "",
            prize or "",
        ]
        if x
    )
    url = escape(r.get("url") or r.get("source_url") or "#", quote=True)
    return f"""
<tr><td style="padding:18px 0;border-top:1px solid {LINE}">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
    <td style="vertical-align:top">
      <a href="{url}" style="font:600 16px Georgia,serif;color:{INK};text-decoration:none">{escape(r.get("name") or "")}</a>
      <div style="font-size:13px;color:{MUTED};margin-top:3px">{meta}</div>
    </td>
    <td style="vertical-align:top;text-align:right;white-space:nowrap;padding-left:12px">
      <span style="display:inline-block;background:#eef3fb;color:{ACCENT};font-weight:600;font-size:13px;padding:4px 9px;border-radius:12px">Fit {int(r.get("score") or 0)}</span>
    </td>
  </tr></table>
  <p style="margin:10px 0 6px;font-size:14px;color:{INK}">{escape(opp.get("summary") or "")}</p>
  <ul style="margin:6px 0 0 18px;padding:0;font-size:13.5px;color:{MUTED}">{reasons}</ul>
  {blocker}
  <p style="margin:10px 0 0;font-size:13px;color:{INK}"><b>Deadline:</b> {escape(_deadline_text(r.get("deadline"), today))}
    &nbsp;&middot;&nbsp; <a href="{url}" style="color:{ACCENT}">Open program page</a></p>
</td></tr>"""


def _deadline_row(r: dict[str, Any], today: date) -> str:
    left = days_left(r.get("deadline"), today)
    stage = escape((r.get("status") or "").capitalize())
    url = escape(r.get("url") or "#", quote=True)
    return f"""
<tr>
  <td style="padding:8px 0;border-top:1px solid {LINE};font-size:14px"><a href="{url}" style="color:{INK}">{escape(r.get("name") or "")}</a></td>
  <td style="padding:8px 0;border-top:1px solid {LINE};font-size:13px;color:{MUTED}">{stage}</td>
  <td style="padding:8px 0;border-top:1px solid {LINE};font-size:13px;text-align:right;color:{"#9a3b1b" if left is not None and left <= 7 else INK}">{escape(str(r.get("deadline")))} &middot; {left}d</td>
</tr>"""


def weekly_email(
    company: str,
    new: list[dict[str, Any]],
    upcoming: list[dict[str, Any]],
    today: date,
    *,
    app_url: str = "",
    notice: str = "",
) -> dict[str, str]:
    """Build subject, HTML and text bodies for the weekly digest."""
    n = len(new)
    if n:
        subject = f"{n} new funding program{'s' if n != 1 else ''} for {company}"
    elif upcoming:
        subject = f"{len(upcoming)} application deadline{'s' if len(upcoming) != 1 else ''} coming up for {company}"
    else:
        subject = f"Weekly funding scan for {company}: nothing new this week"

    programs = "".join(_program_block(r, today) for r in new) or (
        f'<tr><td style="padding:14px 0;border-top:1px solid {LINE};font-size:14px;color:{MUTED}">'
        "No new programs passed the fit threshold this week. The scan skips pages it has "
        "already reviewed, so only genuinely new programs appear here.</td></tr>"
    )
    deadlines = (
        f"""
<h2 style="font:600 18px Georgia,serif;color:{INK};margin:28px 0 6px">Deadlines in your pipeline</h2>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0">{"".join(_deadline_row(r, today) for r in upcoming)}</table>"""
        if upcoming
        else ""
    )
    notice_html = (
        f'<p style="margin:16px 0 0;padding:10px 12px;background:#fff7e6;border-radius:6px;font-size:13px;color:#7a5200">{escape(notice)}</p>'
        if notice
        else ""
    )
    cta = (
        f'<p style="margin:26px 0 0"><a href="{escape(app_url, quote=True)}" style="display:inline-block;background:{INK};color:#fff;text-decoration:none;font-weight:600;font-size:14px;padding:10px 18px;border-radius:6px">Open Opportunity Scout</a></p>'
        if app_url
        else ""
    )
    html = f"""<!doctype html>
<html><body style="margin:0;padding:0;background:#f4f5f8">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f5f8"><tr><td align="center" style="padding:28px 12px">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:10px;border:1px solid {LINE}">
<tr><td style="padding:28px 32px 8px;font-family:Arial,Helvetica,sans-serif">
  <div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:{MUTED}">Opportunity Scout &middot; Week of {today:%d %B %Y}</div>
  <h1 style="font:600 24px Georgia,serif;color:{INK};margin:8px 0 4px">{escape(subject)}</h1>
  <p style="margin:0;font-size:14px;color:{MUTED}">Programs found on the web this week, checked against {escape(company)}'s profile and ranked by fit.</p>
  {notice_html}
</td></tr>
<tr><td style="padding:8px 32px 28px;font-family:Arial,Helvetica,sans-serif">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">{programs}</table>
  {deadlines}
  {cta}
  <p style="margin:26px 0 0;font-size:12px;color:{MUTED}">Fit scores and reasons are AI-generated from each program's own page. Always confirm eligibility on the official page before applying.</p>
</td></tr></table>
</td></tr></table>
</body></html>"""

    lines = [subject, f"Week of {today:%d %B %Y}", ""]
    if notice:
        lines += [f"Note: {notice}", ""]
    for r in new:
        lines += [
            f"- {r.get('name')} (fit {r.get('score')})",
            f"  Deadline: {_deadline_text(r.get('deadline'), today)}",
            f"  {r.get('url') or r.get('source_url') or ''}",
        ]
    if not new:
        lines.append("No new programs passed the fit threshold this week.")
    if upcoming:
        lines += ["", "Deadlines in your pipeline:"]
        lines += [f"- {r.get('name')}: {r.get('deadline')} ({r.get('status')})" for r in upcoming]
    if app_url:
        lines += ["", app_url]
    return {"subject": subject, "html": html, "text": "\n".join(lines)}


def upcoming_deadlines(
    records: list[dict[str, Any]], today: date, within_days: int
) -> list[dict[str, Any]]:
    """Tracked programs (saved/applying) whose deadline falls in the next `within_days` days."""
    keep = []
    for r in records:
        left = days_left(r.get("deadline"), today)
        if left is not None and 0 <= left <= within_days:
            keep.append(r)
    return sorted(keep, key=lambda r: r["deadline"])
