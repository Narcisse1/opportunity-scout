"""Web API + single-page UI.

    uv run scout web            # http://127.0.0.1:8000

Scans take minutes, so they run as background jobs: the UI POSTs a job and polls it.

Access: anyone can use the app as a guest. Sample-page runs are unlimited; live web runs
spend the server's API credits, so guests get a small daily allowance per visitor and in
total (SCOUT_GUEST_RUNS_PER_VISITOR, SCOUT_GUEST_RUNS_PER_DAY) and smaller searches. The
owner sends SCOUT_ACCESS_TOKEN as `X-Access-Token` to lift the limits and to run the weekly
email digest. Set SCOUT_GUEST_ACCESS=off to require the token for everything.
"""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .config import env
from .emails import upcoming_deadlines, weekly_email
from .exports import to_csv, to_ics
from .intel import IntelMonitor, IntelReport
from .llm import StructuredLLM, make_llm, provider_ready, resolve_provider
from .models import Competitor, StartupProfile
from .opportunities import OpportunityScout, ScanReport
from .publish import intel_markdown, opportunities_markdown
from .sources import ExaSource, FallbackFetcher, FirecrawlSource, FixtureSource, MultiSource
from .store import Store

TRACKED = ["saved", "applying", "submitted", "won", "lost"]
PipelineStatus = Literal["new", "saved", "applying", "submitted", "won", "lost", "dismissed"]

STATIC = Path(__file__).resolve().parent / "static"
FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

MAX_QUERIES = 10
MAX_COMPETITORS = 5
MAX_JOBS_KEPT = 200

# Guest limits for live (credit-spending) runs.
GUEST_MAX_QUERIES = 3
GUEST_MAX_RESULTS = 3
GUEST_MAX_COMPETITORS = 2

Access = Literal["owner", "guest"]


class GuestQuota:
    """Counts live runs per visitor and in total, per UTC day."""

    def __init__(self, per_visitor: int, per_day: int) -> None:
        self.per_visitor, self.per_day = per_visitor, per_day
        self._day = ""
        self._by_visitor: dict[str, int] = {}
        self._total = 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self._day:
            self._day, self._by_visitor, self._total = today, {}, 0

    def remaining(self, visitor: str) -> int:
        with self._lock:
            self._roll()
            return max(
                0,
                min(
                    self.per_visitor - self._by_visitor.get(visitor, 0),
                    self.per_day - self._total,
                ),
            )

    def take(self, visitor: str) -> None:
        with self._lock:
            self._roll()
            if self._total >= self.per_day:
                raise HTTPException(
                    429,
                    "Today's live searches for guests are used up. Try 'Sample pages' mode, "
                    "or come back tomorrow.",
                )
            if self._by_visitor.get(visitor, 0) >= self.per_visitor:
                raise HTTPException(
                    429,
                    f"Guests can run {self.per_visitor} live searches a day. Try 'Sample "
                    "pages' mode, which has no limit.",
                )
            self._by_visitor[visitor] = self._by_visitor.get(visitor, 0) + 1
            self._total += 1


def visitor_id(request: Request) -> str:
    # Render and most hosts put the client address first in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class ScanRequest(BaseModel):
    profile: StartupProfile
    queries: list[str] = Field(min_length=1, max_length=MAX_QUERIES)
    min_score: int = Field(60, ge=0, le=100)
    results_per_query: int = Field(5, ge=1, le=10)
    draft: bool = True
    offline: bool = False


class IntelRequest(BaseModel):
    competitors: list[Competitor] = Field(min_length=1, max_length=MAX_COMPETITORS)
    profile: StartupProfile | None = None
    news_days: int = Field(7, ge=1, le=60)
    offline: bool = False
    fixture_set: Literal["week1", "week2"] = "week1"


class WeeklyDigestRequest(BaseModel):
    profile: StartupProfile
    queries: list[str] = Field(min_length=1, max_length=MAX_QUERIES)
    min_score: int = Field(60, ge=0, le=100)
    results_per_query: int = Field(4, ge=1, le=10)
    deadline_window_days: int = Field(45, ge=1, le=365)
    app_url: str = ""


class EmailPreviewRequest(BaseModel):
    company: str = Field(min_length=1, max_length=200)
    opportunities: list[dict[str, Any]] = Field(max_length=50)
    min_score: int = Field(60, ge=0, le=100)


class PipelineUpdate(BaseModel):
    status: PipelineStatus
    notes: str | None = Field(None, max_length=5000)
    # The scan result to start tracking (needed the first time; scans can be offline/in-memory).
    record: dict[str, Any] | None = None


class Job(BaseModel):
    id: str
    kind: Literal["opportunities", "intel"]
    status: Literal["queued", "running", "done", "error"] = "queued"
    created: float
    finished: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


def serialize_scan(report: ScanReport, min_score: int, today: date) -> dict[str, Any]:
    shortlist = {o.key for o in report.shortlist(min_score)}
    return {
        "documents_found": report.documents_found,
        "documents_processed": report.documents_processed,
        "opportunities": [
            {**o.record(), "shortlisted": o.key in shortlist} for o in report.opportunities
        ],
        "documents_already_seen": report.documents_already_seen,
        "documents_unreadable": report.documents_unreadable,
        "errors": report.errors,
        "notice": report.notice,
        "usage": report.usage,
        "markdown": opportunities_markdown(report, min_score, today),
    }


def serialize_intel(report: IntelReport, today: date) -> dict[str, Any]:
    return {
        "digest": report.digest.model_dump() if report.digest else None,
        "competitors": [
            {
                "name": c.competitor,
                "pages": [
                    {"label": p.label, "url": p.url, "status": p.status, "error": p.error}
                    for p in c.pages
                ],
                "news": [
                    {"url": d.url, "title": d.title, "published": d.published} for d in c.news
                ],
                "findings": [f.model_dump() for f in c.findings],
                "ungrounded_dropped": c.ungrounded_dropped,
                "error": c.error,
            }
            for c in report.competitors
        ],
        "usage": report.usage,
        "markdown": intel_markdown(report, today),
    }


def create_app(
    *,
    llm_factory: Callable[[], StructuredLLM] | None = None,
    store: Store | None = None,
    live_sources: Callable[[], tuple[Any, Any]] | None = None,
    workers: int = 2,
) -> FastAPI:
    app = FastAPI(title="Opportunity Scout", version="0.1.0")
    pool = ThreadPoolExecutor(max_workers=workers)
    jobs: dict[str, Job] = {}
    lock = threading.Lock()
    shared_store = store or Store(env("SCOUT_DB") or "./data/web.db")
    demo_intel_store = Store(":memory:")  # keeps the week1 -> week2 demo separate from real data
    new_llm = llm_factory or make_llm

    def default_live_sources():
        sources = []
        if key := env("FIRECRAWL_API_KEY"):
            sources.append(FirecrawlSource(key))
        if key := env("EXA_API_KEY"):
            sources.append(ExaSource(key))
        if not sources:
            raise HTTPException(
                400,
                "Live mode needs FIRECRAWL_API_KEY or EXA_API_KEY on the "
                "server. Use the offline demo instead.",
            )
        return MultiSource(sources), FallbackFetcher(sources)

    get_live = live_sources or default_live_sources

    quota = GuestQuota(
        int(env("SCOUT_GUEST_RUNS_PER_VISITOR") or 3), int(env("SCOUT_GUEST_RUNS_PER_DAY") or 25)
    )

    def guests_allowed() -> bool:
        return (env("SCOUT_GUEST_ACCESS") or "on").lower() not in ("off", "false", "0", "no")

    def access(
        x_access_token: str | None = Header(default=None),
        token: str | None = Query(default=None),  # for download links, which can't set headers
    ) -> Access:
        expected = env("SCOUT_ACCESS_TOKEN")
        given = x_access_token or token or ""
        if not expected:
            return "owner"  # private/local install: no token configured
        if given:
            if secrets.compare_digest(given, expected):
                return "owner"
            raise HTTPException(401, "Wrong access token")
        if guests_allowed():
            return "guest"
        raise HTTPException(401, "Missing access token")

    def owner_only(level: Access = Depends(access)) -> None:  # noqa: B008
        if level != "owner":
            raise HTTPException(401, "This action needs the owner's access token")

    def submit(kind: str, work: Callable[[], dict[str, Any]]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, created=time.time())
        with lock:
            jobs[job.id] = job
            excess = len(jobs) - MAX_JOBS_KEPT
            if excess > 0:
                for old in sorted(jobs.values(), key=lambda j: j.created)[:excess]:
                    jobs.pop(old.id, None)

        def run() -> None:
            job.status = "running"
            try:
                job.result = work()
                job.status = "done"
            except Exception as exc:  # surface any failure to the UI instead of a stuck job
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "error"
            finally:
                job.finished = time.time()

        pool.submit(run)
        return job

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        provider = resolve_provider()
        token_set = bool(env("SCOUT_ACCESS_TOKEN"))
        return {
            "llm_provider": provider,
            "llm_ready": provider_ready(provider) or llm_factory is not None,
            "firecrawl": bool(env("FIRECRAWL_API_KEY")),
            "exa": bool(env("EXA_API_KEY")),
            "token_required": token_set and not guests_allowed(),
            "guest_access": token_set and guests_allowed(),
            "guest_live_runs_left": quota.remaining(visitor_id(request)) if token_set else None,
        }

    def charge_guest(level: Access, request: Request, offline: bool) -> None:
        if level == "guest" and not offline:
            quota.take(visitor_id(request))

    @app.post("/api/opportunities/scan")
    def scan(req: ScanRequest, request: Request, level: Access = Depends(access)) -> Job:  # noqa: B008
        if req.offline:
            source = FixtureSource(FIXTURES / "opportunities")
            search, fetcher, run_store = source, source, Store(":memory:")
        else:
            search, fetcher = get_live()
            run_store = shared_store
            if level == "guest":
                req.queries = req.queries[:GUEST_MAX_QUERIES]
                req.results_per_query = min(req.results_per_query, GUEST_MAX_RESULTS)
        charge_guest(level, request, req.offline)

        def work() -> dict[str, Any]:
            today = date.today()
            scout = OpportunityScout(
                req.profile,
                search=search,
                fetcher=fetcher,
                llm=new_llm(),
                store=run_store,
                min_score=req.min_score,
                draft=req.draft,
            )
            report = scout.run(req.queries, req.results_per_query, today=today)
            return serialize_scan(report, req.min_score, today)

        return submit("opportunities", work)

    @app.post("/api/intel/run")
    def intel(req: IntelRequest, request: Request, level: Access = Depends(access)) -> Job:  # noqa: B008
        if req.offline:
            fetcher = FixtureSource(FIXTURES / "intel" / req.fixture_set)
            news_dir = FIXTURES / "intel" / f"{req.fixture_set}-news"
            news = FixtureSource(news_dir) if news_dir.is_dir() else None
            run_store = demo_intel_store
        else:
            news, fetcher = get_live()
            run_store = shared_store
            if level == "guest":
                req.competitors = req.competitors[:GUEST_MAX_COMPETITORS]
        charge_guest(level, request, req.offline)

        def work() -> dict[str, Any]:
            monitor = IntelMonitor(
                req.competitors,
                fetcher=fetcher,
                llm=new_llm(),
                store=run_store,
                news=news,
                profile=req.profile,
                news_days=req.news_days,
            )
            return serialize_intel(monitor.run(), date.today())

        return submit("intel", work)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    # --- weekly email digest ---------------------------------------------------
    @app.post("/api/digest/weekly", dependencies=[Depends(owner_only)])
    def weekly_digest(req: WeeklyDigestRequest) -> dict[str, Any]:
        """Run a live scan and return a ready-to-send email (called weekly by n8n).

        Pages reviewed in earlier weeks are skipped, so the email lists only new programs.
        """
        search, fetcher = get_live()
        today = date.today()
        scout = OpportunityScout(
            req.profile,
            search=search,
            fetcher=fetcher,
            llm=new_llm(),
            store=shared_store,
            min_score=req.min_score,
            draft=False,
        )
        report = scout.run(req.queries, req.results_per_query, today=today)
        new = [o.record() for o in report.shortlist(req.min_score)]
        tracked = shared_store.list_opportunities(statuses=["saved", "applying"])
        upcoming = upcoming_deadlines(tracked, today, req.deadline_window_days)
        email = weekly_email(
            req.profile.name, new, upcoming, today, app_url=req.app_url, notice=report.notice
        )
        return {
            **email,
            "new_count": len(new),
            "upcoming_count": len(upcoming),
            "documents_processed": report.documents_processed,
            "errors": report.errors[:10],
            "usage": report.usage,
        }

    @app.post("/api/digest/preview")
    def preview_digest(req: EmailPreviewRequest) -> dict[str, str]:
        """Render the weekly email from results already on screen (no scan, no AI calls)."""
        today = date.today()
        new = [
            r
            for r in req.opportunities
            if isinstance(r.get("name"), str)
            and r.get("fit")
            and int(r.get("score") or 0) >= req.min_score
        ]
        new.sort(key=lambda r: -int(r.get("score") or 0))
        tracked = shared_store.list_opportunities(statuses=["saved", "applying"])
        return weekly_email(req.company, new, upcoming_deadlines(tracked, today, 45), today)

    # --- application pipeline ------------------------------------------------
    # Pipeline actions are cheap (no AI or search calls), so guests can try them too.
    @app.get("/api/pipeline", dependencies=[Depends(access)])
    def pipeline() -> list[dict[str, Any]]:
        return shared_store.list_opportunities(statuses=TRACKED)

    @app.put("/api/pipeline/{key}", dependencies=[Depends(access)])
    def update_pipeline(key: str, body: PipelineUpdate) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if body.record is not None:
            if not isinstance(body.record.get("name"), str) or not body.record["name"].strip():
                raise HTTPException(422, "record.name is required")
            shared_store.upsert_opportunity(key, {**body.record, "key": key}, now)
        if not shared_store.set_status(key, body.status, body.notes, now):
            raise HTTPException(404, "Opportunity not tracked yet: send `record` the first time")
        return {"key": key, "status": body.status}

    @app.get("/api/pipeline.ics", dependencies=[Depends(access)])
    def pipeline_calendar() -> Response:
        records = shared_store.list_opportunities(statuses=["saved", "applying"])
        return Response(
            to_ics(records),
            media_type="text/calendar",
            headers={"Content-Disposition": 'attachment; filename="funding-deadlines.ics"'},
        )

    @app.get("/api/pipeline.csv", dependencies=[Depends(access)])
    def pipeline_csv() -> Response:
        return Response(
            to_csv(shared_store.list_opportunities(statuses=TRACKED)),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="funding-pipeline.csv"'},
        )

    @app.get("/api/opportunities", dependencies=[Depends(access)])
    def tracked(min_score: int = 0) -> list[dict[str, Any]]:
        return shared_store.list_opportunities(min_score)

    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
