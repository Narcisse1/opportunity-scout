"""Command line entry point.

scout opportunities --config scout.yaml --profile profile.yaml
scout intel         --config scout.yaml --profile profile.yaml
scout opportunities --offline        # demo on bundled fixtures (still calls Claude)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from .config import AppConfig, env, load_config, load_profile
from .http import SourceError
from .intel import IntelMonitor
from .llm import ClaudeLLM
from .opportunities import OpportunityScout
from .publish import NotionPublisher, intel_markdown, opportunities_markdown, write_obsidian
from .sources import ExaSource, FirecrawlSource, FixtureSource, MultiSource
from .store import Store

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
log = logging.getLogger("scout")


def build_search_and_fetcher(offline: bool, fixture_dir: Path | None = None):
    if offline:
        source = FixtureSource(fixture_dir or FIXTURES / "opportunities")
        return source, source
    sources = []
    firecrawl = exa = None
    if key := env("FIRECRAWL_API_KEY"):
        firecrawl = FirecrawlSource(key)
        sources.append(firecrawl)
    if key := env("EXA_API_KEY"):
        exa = ExaSource(key)
        sources.append(exa)
    if not sources:
        sys.exit("Set FIRECRAWL_API_KEY and/or EXA_API_KEY (or run with --offline).")
    fetcher = firecrawl or exa  # Firecrawl gives cleaner page markdown when both exist
    return MultiSource(sources), fetcher


def publish(cfg: AppConfig, folder: str, filename: str, title: str, markdown: str) -> None:
    path = write_obsidian(cfg.output.obsidian_vault, folder, filename, markdown)
    print(f"Report written to {path}")
    if cfg.output.notion:
        token, db = env("NOTION_TOKEN"), env("NOTION_DATABASE_ID")
        if not (token and db):
            print("Notion enabled but NOTION_TOKEN / NOTION_DATABASE_ID are not set; skipped.")
            return
        try:
            print(f"Notion page: {NotionPublisher(token, db).publish(title, markdown)}")
        except SourceError as exc:
            print(f"Notion publish failed: {exc}", file=sys.stderr)


def cmd_opportunities(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    profile = load_profile(args.profile)
    search, fetcher = build_search_and_fetcher(args.offline)
    store = Store(":memory:" if args.offline else cfg.database)
    llm = ClaudeLLM(cfg.model)
    scout = OpportunityScout(
        profile,
        search=search,
        fetcher=fetcher,
        llm=llm,
        store=store,
        min_score=cfg.opportunities.min_score,
        draft=cfg.opportunities.draft_applications,
        max_workers=cfg.max_workers,
    )
    today = date.today()
    report = scout.run(
        cfg.opportunities.queries,
        results_per_query=cfg.opportunities.results_per_query,
        today=today,
    )
    shortlist = report.shortlist(cfg.opportunities.min_score)
    md = opportunities_markdown(report, cfg.opportunities.min_score, today)
    publish(
        cfg,
        "Opportunities",
        f"{today.isoformat()} opportunities.md",
        f"Opportunity scan {today.isoformat()} ({len(shortlist)} shortlisted)",
        md,
    )
    print(
        json.dumps(
            {"shortlisted": len(shortlist), "errors": len(report.errors), "usage": report.usage},
            indent=2,
        )
    )
    return 0


def cmd_intel(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    profile = load_profile(args.profile) if args.profile else None
    if args.offline:
        fetcher = FixtureSource(FIXTURES / "intel" / args.fixture_set)
        news_dir = FIXTURES / "intel" / f"{args.fixture_set}-news"
        news = FixtureSource(news_dir) if news_dir.is_dir() else None
        store = Store(cfg.database.replace(".db", "-offline.db"))
    else:
        news, fetcher = build_search_and_fetcher(False)
        store = Store(cfg.database)
    monitor = IntelMonitor(
        cfg.intel.competitors,
        fetcher=fetcher,
        llm=ClaudeLLM(cfg.model),
        store=store,
        news=news,
        profile=profile,
        news_days=cfg.intel.news_days,
    )
    report = monitor.run()
    today = date.today()
    md = intel_markdown(report, today)
    publish(
        cfg,
        "Competitor Intel",
        f"{today.isoformat()} competitor briefing.md",
        f"Competitor briefing {today.isoformat()}",
        md,
    )
    print(json.dumps({"findings": len(report.findings), "usage": report.usage}, indent=2))
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from .web import serve

    print(f"Opportunity Scout UI on http://{args.host}:{args.port}")
    serve(args.host, args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="scout", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_opp = sub.add_parser("opportunities", help="find, score and draft applications")
    p_opp.add_argument("--config", default="scout.yaml")
    p_opp.add_argument("--profile", default="profile.yaml")
    p_opp.add_argument("--offline", action="store_true", help="use bundled fixture pages")
    p_opp.set_defaults(func=cmd_opportunities)

    p_intel = sub.add_parser("intel", help="monitor competitors and write a briefing")
    p_intel.add_argument("--config", default="scout.yaml")
    p_intel.add_argument("--profile", default=None)
    p_intel.add_argument("--offline", action="store_true", help="use bundled fixture pages")
    p_intel.add_argument("--fixture-set", default="week1", choices=["week1", "week2"])
    p_intel.set_defaults(func=cmd_intel)

    p_web = sub.add_parser("web", help="run the web UI")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.set_defaults(func=cmd_web)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
