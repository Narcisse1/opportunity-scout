"""Competitor & market-intelligence monitor.

Per competitor:
    fetch tracked pages -> hash-compare with the last snapshot (no LLM call when nothing
    changed) -> unified diff of changed text + fresh news results
    -> Claude: classify meaningful changes with a verbatim evidence quote (structured)
    -> grounding check: drop any finding whose quote is not literally in the evidence
Then one synthesis call turns verified findings into a founder briefing.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from .http import SourceError
from .llm import LLMError, StructuredLLM
from .models import ChangeFinding, Competitor, CompetitorAnalysis, IntelDigest, StartupProfile
from .prompts import INTEL_SYSTEM, digest_system
from .sources import Document, Fetcher, SearchSource
from .store import Store, content_hash

log = logging.getLogger(__name__)

MAX_DIFF_LINES = 400
MAX_NEWS_CHARS = 3000


def compute_diff(old: str, new: str, max_lines: int = MAX_DIFF_LINES) -> str:
    """Changed lines only (+/-), ignoring whitespace-only edits."""
    old_lines = [ln.strip() for ln in old.splitlines() if ln.strip()]
    new_lines = [ln.strip() for ln in new.splitlines() if ln.strip()]
    changed = [
        line
        for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    if len(changed) > max_lines:
        dropped = len(changed) - max_lines
        changed = changed[:max_lines] + [f"[... {dropped} more changed lines not shown]"]
    return "\n".join(changed)


_SPACE = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text.translate(_QUOTES)).strip().lower()


def is_grounded(quote: str, evidence: str) -> bool:
    q = _norm(quote).lstrip("+- ")
    return len(q) >= 8 and q in _norm(evidence)


@dataclass
class PageCheck:
    label: str
    url: str
    status: Literal["baseline", "unchanged", "changed", "error"]
    diff: str = ""
    error: str = ""


@dataclass
class CompetitorReport:
    competitor: str
    pages: list[PageCheck] = field(default_factory=list)
    news: list[Document] = field(default_factory=list)
    findings: list[ChangeFinding] = field(default_factory=list)
    ungrounded_dropped: int = 0
    error: str = ""


@dataclass
class IntelReport:
    started: str
    competitors: list[CompetitorReport] = field(default_factory=list)
    digest: IntelDigest | None = None
    usage: dict = field(default_factory=dict)

    @property
    def findings(self) -> list[tuple[str, ChangeFinding]]:
        return [(c.competitor, f) for c in self.competitors for f in c.findings]


class IntelMonitor:
    def __init__(
        self,
        competitors: list[Competitor],
        *,
        fetcher: Fetcher,
        llm: StructuredLLM,
        store: Store,
        news: SearchSource | None = None,
        profile: StartupProfile | None = None,
        news_days: int = 7,
        news_per_competitor: int = 5,
    ) -> None:
        self.competitors = competitors
        self.fetcher = fetcher
        self.llm = llm
        self.store = store
        self.news = news
        self.profile = profile
        self.news_days = news_days
        self.news_per_competitor = news_per_competitor

    def run(self) -> IntelReport:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        report = IntelReport(started=now)
        for competitor in self.competitors:
            report.competitors.append(self._check(competitor, now))

        verified = report.findings
        if verified:
            lines = [
                f"- [{name}] ({f.category}, {f.importance}) {f.summary} — source: {f.source_url}"
                for name, f in verified
            ]
            try:
                report.digest = self.llm.structured(
                    system=digest_system(self.profile),
                    prompt="Verified findings this period:\n" + "\n".join(lines),
                    schema=IntelDigest,
                )
            except LLMError as exc:
                log.warning("digest failed: %s", exc)
        report.usage = self.llm.usage.summary()
        return report

    def _check(self, competitor: Competitor, now: str) -> CompetitorReport:
        result = CompetitorReport(competitor=competitor.name)

        for label, url in competitor.pages.items():
            try:
                doc = self.fetcher.fetch(url)
            except SourceError as exc:
                result.pages.append(PageCheck(label, url, "error", error=str(exc)))
                continue
            previous = self.store.latest_snapshot(url)
            if previous is None:
                self.store.add_snapshot(url, doc.content, now)
                result.pages.append(PageCheck(label, url, "baseline"))
            elif previous[0] == content_hash(doc.content):
                result.pages.append(PageCheck(label, url, "unchanged"))
            else:
                diff = compute_diff(previous[1], doc.content)
                self.store.add_snapshot(url, doc.content, now)
                status = "changed" if diff else "unchanged"
                result.pages.append(PageCheck(label, url, status, diff=diff))

        if self.news and competitor.news_query:
            try:
                hits = self.news.search(
                    competitor.news_query, self.news_per_competitor, since_days=self.news_days
                )
                result.news = [d for d in hits if not self.store.is_seen(f"news:{d.url}")]
            except SourceError as exc:
                result.error = f"news search failed: {exc}"

        evidence_blocks = [
            f'<evidence url="{p.url}" kind="page diff: {p.label}">\n{p.diff}\n</evidence>'
            for p in result.pages
            if p.status == "changed"
        ] + [
            f'<evidence url="{d.url}" kind="news" title="{d.title}" '
            f'published="{d.published or "unknown"}">\n{d.content[:MAX_NEWS_CHARS]}\n</evidence>'
            for d in result.news
        ]
        if not evidence_blocks:
            return result  # nothing new: no tokens spent

        evidence = "\n\n".join(evidence_blocks)
        try:
            analysis = self.llm.structured(
                system=INTEL_SYSTEM,
                prompt=f"Competitor: {competitor.name}\n\n{evidence}",
                schema=CompetitorAnalysis,
            )
        except LLMError as exc:
            result.error = f"analysis failed: {exc}"
            return result

        for finding in analysis.findings:
            if is_grounded(finding.evidence_quote, evidence):
                result.findings.append(finding)
            else:
                result.ungrounded_dropped += 1
                log.info("dropped ungrounded finding for %s: %s", competitor.name, finding.summary)

        for doc in result.news:
            self.store.mark_seen(f"news:{doc.url}", now)
        return result
