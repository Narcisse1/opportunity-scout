"""Grant, accelerator & hackathon tracker.

Pipeline per run:
    search (Firecrawl/Exa) -> skip already-processed URLs -> scrape thin results
    -> Claude: extract opportunities (structured)
    -> deterministic rules: expired deadline, excluded keywords, already-known opportunity
    -> Claude: fit score vs. startup profile (structured)
    -> Claude: draft application answers for the shortlist (structured)
    -> persist to SQLite -> publish (Obsidian / Notion)

Cheap deterministic checks run before every paid LLM step, so tokens go to pages and
opportunities that can still matter.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from .http import SourceError
from .llm import LLMError, StructuredLLM
from .models import (
    ApplicationDraft,
    ExtractedOpportunity,
    ExtractionResult,
    FitAssessment,
    StartupProfile,
)
from .prompts import EXTRACT_SYSTEM, draft_system, fit_system
from .sources import Document, Fetcher, SearchSource
from .store import Store

log = logging.getLogger(__name__)

DEFAULT_QUESTIONS = [
    "Describe your startup in one sentence.",
    "What problem are you solving, and why now?",
    "Why is your team the right one to build this?",
    "How would this program help you reach your next milestone?",
]

MAX_DOC_CHARS = 40_000  # deliberate cap per page; long listing pages are noted as truncated


# ---------------------------------------------------------------------------
# Deterministic rules
# ---------------------------------------------------------------------------


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def days_left(deadline: str | None, today: date) -> int | None:
    parsed = parse_date(deadline)
    return None if parsed is None else (parsed - today).days


def opportunity_key(opp: ExtractedOpportunity) -> str:
    """Stable identity so the same program found on several pages is only scored once."""
    ident = f"{opp.name.lower().strip()}|{(opp.organizer or '').lower().strip()}"
    return hashlib.sha1(ident.encode("utf-8")).hexdigest()[:16]


def rule_rejections(opp: ExtractedOpportunity, profile: StartupProfile, today: date) -> list[str]:
    reasons = []
    remaining = days_left(opp.deadline, today)
    if remaining is not None and remaining < 0:
        reasons.append(f"deadline passed ({opp.deadline})")
    haystack = f"{opp.name} {opp.summary} {' '.join(opp.eligibility)}".lower()
    for word in profile.exclude_keywords:
        if word.lower() in haystack:
            reasons.append(f"excluded keyword: {word}")
    return reasons


def clamp_score(fit: FitAssessment) -> FitAssessment:
    score = max(0, min(100, fit.score))
    if fit.verdict == "ineligible":
        score = min(score, 20)
    return fit.model_copy(update={"score": score})


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ScoredOpportunity:
    opportunity: ExtractedOpportunity
    source_url: str
    fit: FitAssessment | None = None
    draft: ApplicationDraft | None = None
    rejected: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return opportunity_key(self.opportunity)

    @property
    def score(self) -> int:
        return self.fit.score if self.fit else 0

    def record(self) -> dict:
        opp = self.opportunity
        return {
            "key": self.key,
            "name": opp.name,
            "organizer": opp.organizer,
            "type": opp.type,
            "url": opp.url or self.source_url,
            "source_url": self.source_url,
            "deadline": opp.deadline,
            "score": self.score,
            "verdict": self.fit.verdict if self.fit else "rejected",
            "opportunity": opp.model_dump(),
            "fit": self.fit.model_dump() if self.fit else None,
            "draft": self.draft.model_dump() if self.draft else None,
            "rejected": self.rejected,
        }


@dataclass
class ScanReport:
    started: str
    queries: list[str]
    documents_found: int = 0
    documents_processed: int = 0
    opportunities: list[ScoredOpportunity] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def shortlist(self, min_score: int) -> list[ScoredOpportunity]:
        keep = [o for o in self.opportunities if o.fit and o.score >= min_score]
        return sorted(keep, key=lambda o: (-o.score, o.opportunity.deadline or "9999"))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def render_document(doc: Document) -> str:
    body = doc.content
    note = ""
    if len(body) > MAX_DOC_CHARS:
        body = body[:MAX_DOC_CHARS]
        note = "\n[page truncated by the crawler for length]"
    published = f' published="{doc.published}"' if doc.published else ""
    return (
        f'<untrusted_document url="{doc.url}" title="{doc.title}"{published}>\n'
        f"{body}{note}\n</untrusted_document>"
    )


class OpportunityScout:
    def __init__(
        self,
        profile: StartupProfile,
        *,
        search: SearchSource,
        fetcher: Fetcher,
        llm: StructuredLLM,
        store: Store,
        min_score: int = 60,
        draft: bool = True,
        max_workers: int = 4,
        min_content_chars: int = 600,
    ) -> None:
        self.profile = profile
        self.search = search
        self.fetcher = fetcher
        self.llm = llm
        self.store = store
        self.min_score = min_score
        self.draft = draft
        self.max_workers = max_workers
        self.min_content_chars = min_content_chars
        # Built once per run: identical across calls, so prompt caching applies.
        self._fit_system = fit_system(profile)
        self._draft_system = draft_system(profile)

    # -- public ---------------------------------------------------------------
    def run(
        self, queries: Iterable[str], results_per_query: int = 5, *, today: date | None = None
    ) -> ScanReport:
        today = today or date.today()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        queries = list(queries)
        report = ScanReport(started=now, queries=queries)

        docs: dict[str, Document] = {}
        for query in queries:
            try:
                for doc in self.search.search(query, results_per_query):
                    docs.setdefault(doc.url, doc)
            except SourceError as exc:
                report.errors.append(f"search '{query}': {exc}")
        report.documents_found = len(docs)

        new_docs = [d for d in docs.values() if not self.store.is_seen(d.url)]
        log.info("%d results, %d new pages to analyse", len(docs), len(new_docs))

        results: dict[str, ScoredOpportunity] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._process, doc, today): doc for doc in new_docs}
            for future in as_completed(futures):
                doc = futures[future]
                try:
                    found = future.result()
                except (LLMError, SourceError) as exc:
                    # Not marked seen, so the page is retried on the next run.
                    report.errors.append(f"{doc.url}: {exc}")
                    log.warning("page failed: %s: %s", doc.url, exc)
                    continue
                self.store.mark_seen(doc.url, now)
                report.documents_processed += 1
                for item in found:
                    current = results.get(item.key)
                    if current is None or item.score > current.score:
                        results[item.key] = item

        for item in results.values():
            if item.fit is not None:  # rule-rejected items are reported, not stored
                self.store.upsert_opportunity(item.key, item.record(), now)
        report.opportunities = sorted(results.values(), key=lambda o: -o.score)
        report.usage = self.llm.usage.summary()
        return report

    # -- per page -------------------------------------------------------------
    def _process(self, doc: Document, today: date) -> list[ScoredOpportunity]:
        if len(doc.content) < self.min_content_chars:
            try:
                doc = self.fetcher.fetch(doc.url)
            except SourceError:
                # Some sites refuse every scraper; the search snippet may still be enough.
                if len(doc.content.strip()) < 200:
                    raise
                log.info("scrape failed for %s; using the search result text", doc.url)

        extraction = self.llm.structured(
            system=EXTRACT_SYSTEM,
            prompt=f"Today is {today.isoformat()}.\n\n{render_document(doc)}",
            schema=ExtractionResult,
        )

        out: list[ScoredOpportunity] = []
        for opp in extraction.opportunities:
            item = ScoredOpportunity(opportunity=opp, source_url=doc.url)
            item.rejected = rule_rejections(opp, self.profile, today)
            if item.rejected:
                out.append(item)
                continue
            if self.store.has_opportunity(item.key):
                log.info("already tracked: %s", opp.name)
                continue
            item.fit = clamp_score(self._assess(opp, today))
            if self.draft and item.fit.score >= self.min_score:
                item.draft = self._draft(opp, today)
            out.append(item)
        return out

    def _assess(self, opp: ExtractedOpportunity, today: date) -> FitAssessment:
        remaining = days_left(opp.deadline, today)
        timing = f"{remaining} days until the deadline" if remaining is not None else "no deadline"
        return self.llm.structured(
            system=self._fit_system,
            prompt=(
                f"Today is {today.isoformat()} ({timing}).\n\n"
                f"<opportunity>\n{opp.model_dump_json(indent=2)}\n</opportunity>"
            ),
            schema=FitAssessment,
        )

    def _draft(self, opp: ExtractedOpportunity, today: date) -> ApplicationDraft:
        questions = opp.application_questions or DEFAULT_QUESTIONS
        numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        return self.llm.structured(
            system=self._draft_system,
            prompt=(
                f"Opportunity: {opp.name} by {opp.organizer or 'unknown organizer'} "
                f"({opp.type}).\nWhat they offer: {opp.funding_or_prize or 'not stated'}.\n"
                f"Summary: {opp.summary}\nEligibility: {'; '.join(opp.eligibility) or 'n/a'}\n\n"
                f"Draft answers to these questions:\n{numbered}"
            ),
            schema=ApplicationDraft,
        )
