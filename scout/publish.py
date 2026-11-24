"""Report rendering (Markdown, Obsidian-ready with YAML frontmatter) and Notion publishing."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import httpx
import yaml

from .http import request_json
from .intel import IntelReport
from .opportunities import ScanReport, days_left

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _frontmatter(data: dict) -> str:
    return "---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + "---\n\n"


def opportunities_markdown(report: ScanReport, min_score: int, today: date) -> str:
    shortlist = report.shortlist(min_score)
    out = [
        _frontmatter(
            {
                "type": "opportunity-scan",
                "date": today.isoformat(),
                "shortlisted": len(shortlist),
                "pages_processed": report.documents_processed,
                "cost_usd": report.usage.get("estimated_cost_usd"),
                "tags": ["scout/opportunities"],
            }
        ),
        f"# Opportunity scan — {today.isoformat()}\n",
        f"{report.documents_found} search results, {report.documents_processed} new pages "
        f"analysed, {len(shortlist)} opportunities at or above {min_score}/100.\n",
    ]
    if shortlist:
        out.append("\n| Score | Opportunity | Type | Deadline | Offer |\n|---|---|---|---|---|")
        for item in shortlist:
            o = item.opportunity
            remaining = days_left(o.deadline, today)
            when = f"{o.deadline} ({remaining}d)" if remaining is not None else "rolling / n/a"
            link = o.url or item.source_url
            out.append(
                f"| {item.score} | [{o.name}]({link}) | {o.type} | {when} | "
                f"{(o.funding_or_prize or '—').replace('|', '/')} |"
            )
    for item in shortlist:
        o, fit = item.opportunity, item.fit
        out.append(f"\n## {o.name} — {fit.score}/100 ({fit.verdict})\n")
        out.append(
            f"**Organizer:** {o.organizer or 'n/a'}  \n**Link:** {o.url or item.source_url}\n"
        )
        out.append(f"{o.summary}\n")
        out.append("**Why it fits**\n" + "".join(f"\n- {r}" for r in fit.reasons) + "\n")
        if fit.risks:
            out.append("**Risks / unclear**\n" + "".join(f"\n- {r}" for r in fit.risks) + "\n")
        if item.draft:
            out.append("### Draft application answers\n")
            for ans in item.draft.answers:
                out.append(f"**{ans.question}**\n\n{ans.answer}\n")
            if item.draft.missing_info:
                out.append(
                    "> [!todo] Gather before submitting\n"
                    + "".join(f"> - {m}\n" for m in item.draft.missing_info)
                )
    rejected = [i for i in report.opportunities if i.rejected or (i.fit and i.score < min_score)]
    if rejected:
        out.append("\n## Filtered out\n")
        for item in rejected:
            why = "; ".join(item.rejected) or f"{item.score}/100 ({item.fit.verdict})"
            out.append(f"- {item.opportunity.name} — {why}")
    if report.notice:
        out.append(f"\n> [!warning] {report.notice}")
    if report.errors:
        out.append("\n## Errors\n" + "".join(f"\n- {e}" for e in report.errors))
    return "\n".join(out) + "\n"


def intel_markdown(report: IntelReport, today: date) -> str:
    out = [
        _frontmatter(
            {
                "type": "competitor-intel",
                "date": today.isoformat(),
                "findings": len(report.findings),
                "cost_usd": report.usage.get("estimated_cost_usd"),
                "tags": ["scout/intel"],
            }
        ),
        f"# Competitor briefing — {today.isoformat()}\n",
    ]
    if report.digest:
        d = report.digest
        out.append(f"> {d.headline}\n")
        out.append("## What it means for us" + "".join(f"\n- {i}" for i in d.implications) + "\n")
        out.append(
            "## Suggested actions" + "".join(f"\n- {a}" for a in d.recommended_actions) + "\n"
        )
    else:
        out.append("No meaningful competitor changes detected this period.\n")
    for comp in report.competitors:
        out.append(f"\n## {comp.competitor}\n")
        pages = ", ".join(f"{p.label}: {p.status}" for p in comp.pages) or "no pages tracked"
        out.append(f"_Pages — {pages}; news items: {len(comp.news)}_\n")
        for f in sorted(comp.findings, key=lambda f: ["high", "medium", "low"].index(f.importance)):
            out.append(
                f"- **[{f.importance}] {f.category}:** {f.summary}  \n"
                f'  > "{f.evidence_quote}" — [source]({f.source_url})'
            )
        if comp.ungrounded_dropped:
            out.append(f"\n_{comp.ungrounded_dropped} unverifiable finding(s) discarded._")
        if comp.error:
            out.append(f"\n_Error: {comp.error}_")
    return "\n".join(out) + "\n"


def write_obsidian(vault: str | Path, folder: str, filename: str, content: str) -> Path:
    target = Path(vault) / folder
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Notion
# ---------------------------------------------------------------------------

_INLINE = re.compile(r"\*\*|__|`")


def markdown_to_blocks(markdown: str) -> list[dict]:
    """Small Markdown -> Notion block converter (headings, bullets, quotes, paragraphs)."""
    body = re.sub(r"\A---\n.*?\n---\n", "", markdown, flags=re.S)
    blocks: list[dict] = []
    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.startswith("|---"):
            continue
        kind, text = "paragraph", line
        for prefix, block_type in (
            ("### ", "heading_3"),
            ("## ", "heading_2"),
            ("# ", "heading_1"),
            ("- ", "bulleted_list_item"),
            ("> ", "quote"),
        ):
            if line.lstrip().startswith(prefix):
                kind, text = block_type, line.lstrip()[len(prefix) :]
                break
        text = _INLINE.sub("", text)
        for start in range(0, len(text), 2000):  # Notion rich-text limit per item
            blocks.append(
                {
                    "object": "block",
                    "type": kind,
                    kind: {
                        "rich_text": [
                            {"type": "text", "text": {"content": text[start : start + 2000]}}
                        ]
                    },
                }
            )
    return blocks


class NotionPublisher:
    API = "https://api.notion.com/v1"
    VERSION = "2022-06-28"

    def __init__(self, token: str, database_id: str, *, client: httpx.Client | None = None) -> None:
        self.database_id = database_id
        self.client = client or httpx.Client()
        self.headers = {"Authorization": f"Bearer {token}", "Notion-Version": self.VERSION}

    def publish(self, title: str, markdown: str, extra_properties: dict | None = None) -> str:
        blocks = markdown_to_blocks(markdown)
        properties = {"Name": {"title": [{"text": {"content": title[:2000]}}]}}
        properties.update(extra_properties or {})
        page = request_json(
            self.client,
            "POST",
            f"{self.API}/pages",
            json={
                "parent": {"database_id": self.database_id},
                "properties": properties,
                "children": blocks[:100],
            },
            headers=self.headers,
        )
        for start in range(100, len(blocks), 100):  # Notion accepts 100 children per request
            request_json(
                self.client,
                "PATCH",
                f"{self.API}/blocks/{page['id']}/children",
                json={"children": blocks[start : start + 100]},
                headers=self.headers,
            )
        return page.get("url", page["id"])
