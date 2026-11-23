from __future__ import annotations

from datetime import date

from conftest import FIXTURES, FakeLLM

from scout.intel import IntelMonitor, compute_diff, is_grounded
from scout.models import ChangeFinding, Competitor, CompetitorAnalysis, IntelDigest
from scout.publish import intel_markdown
from scout.sources import FixtureSource
from scout.store import Store

RIVAL = Competitor(
    name="RivalAI",
    news_query="RivalAI",
    pages={
        "pricing": "https://rival.example.com/pricing",
        "changelog": "https://rival.example.com/changelog",
    },
)


def test_compute_diff_ignores_whitespace_and_keeps_changes():
    old = "Starter\n$29 per month\n\nTeam\n"
    new = "Starter\n  $39 per month  \nTeam\n"
    diff = compute_diff(old, new)
    assert "-$29 per month" in diff and "+$39 per month" in diff
    assert compute_diff("a\n b \n", "a\nb\n\n") == ""


def test_compute_diff_caps_length():
    diff = compute_diff("", "\n".join(f"line {i}" for i in range(50)), max_lines=10)
    assert diff.endswith("[... 40 more changed lines not shown]")


def test_grounding():
    evidence = "+$39 per month\n+Launched Grant Finder:  an agent that matches startups"
    assert is_grounded("$39 per month", evidence)
    assert is_grounded("Launched Grant Finder: an agent", evidence)  # whitespace-insensitive
    assert is_grounded("“Launched Grant Finder”", '"Launched Grant Finder"')
    assert not is_grounded("$49 per month", evidence)
    assert not is_grounded("$39", evidence)  # too short to be meaningful evidence


def finding(quote: str, url: str, summary: str = "x") -> ChangeFinding:
    return ChangeFinding(
        category="pricing", summary=summary, evidence_quote=quote, source_url=url, importance="high"
    )


def analysis(_system, prompt):
    return CompetitorAnalysis(
        findings=[
            finding(
                "$39 per month", "https://rival.example.com/pricing", "Starter up from $29 to $39"
            ),
            finding(
                "Launched Grant Finder", "https://rival.example.com/changelog", "New grant agent"
            ),
            finding(
                "$12M Series A", "https://news.example.com/rivalai-series-a", "Raised Series A"
            ),
            finding("acquired by MegaCorp", "https://rival.example.com/pricing", "hallucinated"),
        ]
    )


def digest(_system, prompt):
    return IntelDigest(
        headline="RivalAI is moving into grant discovery.",
        implications=["Direct overlap"],
        recommended_actions=["Ship faster"],
    )


def monitor(store, week, llm, news=None):
    return IntelMonitor(
        [RIVAL], fetcher=FixtureSource(FIXTURES / "intel" / week), llm=llm, store=store, news=news
    )


def test_first_run_is_baseline_without_llm_calls():
    llm = FakeLLM({CompetitorAnalysis: analysis, IntelDigest: digest})
    report = monitor(Store(), "week1", llm).run()
    assert [p.status for p in report.competitors[0].pages] == ["baseline", "baseline"]
    assert llm.calls == [] and report.digest is None


def test_changes_are_analysed_and_hallucinations_dropped():
    store = Store()
    monitor(store, "week1", FakeLLM({})).run()

    llm = FakeLLM({CompetitorAnalysis: analysis, IntelDigest: digest})
    news = FixtureSource(FIXTURES / "intel" / "week2-news")
    report = monitor(store, "week2", llm, news=news).run()
    comp = report.competitors[0]

    assert [p.status for p in comp.pages] == ["changed", "changed"]
    assert {f.summary for f in comp.findings} == {
        "Starter up from $29 to $39",
        "New grant agent",
        "Raised Series A",
    }
    assert comp.ungrounded_dropped == 1
    assert report.digest.headline.startswith("RivalAI")

    prompt = next(p for name, _, p in llm.calls if name == "CompetitorAnalysis")
    assert "+$39 per month" in prompt and "-$29 per month" in prompt
    assert "Slack integration" not in prompt  # unchanged lines are not sent

    md = intel_markdown(report, date(2026, 11, 23))
    assert "> RivalAI is moving into grant discovery." in md
    assert "1 unverifiable finding(s) discarded" in md


def test_unchanged_pages_cost_nothing():
    store = Store()
    monitor(store, "week1", FakeLLM({})).run()
    llm = FakeLLM({CompetitorAnalysis: analysis, IntelDigest: digest})
    report = monitor(store, "week1", llm).run()
    assert [p.status for p in report.competitors[0].pages] == ["unchanged", "unchanged"]
    assert llm.calls == []


def test_news_is_not_reported_twice():
    store = Store()
    monitor(store, "week1", FakeLLM({})).run()
    news = FixtureSource(FIXTURES / "intel" / "week2-news")
    llm = FakeLLM({CompetitorAnalysis: analysis, IntelDigest: digest})
    monitor(store, "week2", llm, news=news).run()
    again = monitor(store, "week2", llm, news=news).run()
    assert again.competitors[0].news == []
