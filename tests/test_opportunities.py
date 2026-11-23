from __future__ import annotations

from datetime import date

from conftest import FIXTURES, FakeLLM

from scout.llm import LLMError
from scout.models import (
    ApplicationDraft,
    DraftAnswer,
    ExtractedOpportunity,
    ExtractionResult,
    FitAssessment,
)
from scout.opportunities import (
    OpportunityScout,
    clamp_score,
    days_left,
    opportunity_key,
    parse_date,
    rule_rejections,
)
from scout.publish import opportunities_markdown
from scout.sources import FixtureSource
from scout.store import Store

TODAY = date(2026, 11, 23)


def opp(name="Lumen Founders Grant", deadline="2027-02-15", **kw) -> ExtractedOpportunity:
    base = dict(
        name=name,
        organizer="Lumen Foundation",
        type="grant",
        url="https://example.org/lumen-founders-grant",
        deadline=deadline,
        funding_or_prize="$50,000",
        equity_taken=None,
        location_requirements="Africa",
        stage_requirements="pre-seed or seed",
        eligibility=["HQ in Africa"],
        summary="Non-dilutive grant for AI startups in Africa.",
        application_questions=["What problem are you solving?"],
    )
    base.update(kw)
    return ExtractedOpportunity(**base)


# --- deterministic rules --------------------------------------------------------


def test_parse_date_and_days_left():
    assert parse_date("2027-02-15") == date(2027, 2, 15)
    assert parse_date("2027-02-15T23:59:00Z") == date(2027, 2, 15)
    assert parse_date("February 15") is None
    assert parse_date(None) is None
    assert days_left("2026-11-30", TODAY) == 7
    assert days_left(None, TODAY) is None


def test_rule_rejections(profile):
    assert rule_rejections(opp(), profile, TODAY) == []
    assert "deadline passed" in rule_rejections(opp(deadline="2026-03-01"), profile, TODAY)[0]
    crypto = opp(summary="Grant for crypto token projects")
    assert rule_rejections(crypto, profile, TODAY) == ["excluded keyword: crypto"]


def test_opportunity_key_is_case_insensitive():
    assert opportunity_key(opp(name="Lumen Founders Grant")) == opportunity_key(
        opp(name="  lumen founders grant ")
    )


def test_clamp_score():
    fit = FitAssessment(score=140, verdict="strong", reasons=[], risks=[], eligibility_blockers=[])
    assert clamp_score(fit).score == 100
    inel = FitAssessment(score=70, verdict="ineligible", reasons=[], risks=[],
                         eligibility_blockers=["EU only"])
    assert clamp_score(inel).score == 20


# --- pipeline -------------------------------------------------------------------


def extraction_for(_system: str, prompt: str) -> ExtractionResult:
    if "lumen-founders-grant" in prompt:
        return ExtractionResult(opportunities=[opp()])
    if "agentforge" in prompt:
        return ExtractionResult(opportunities=[opp(
            name="AgentForge Global Hackathon", organizer="AgentForge", type="hackathon",
            url="https://example.org/agentforge-hackathon", deadline="2027-01-08",
            funding_or_prize="$25,000", application_questions=[])])
    if "northstar" in prompt:
        return ExtractionResult(opportunities=[opp(
            name="Northstar Accelerator", organizer="Northstar", type="accelerator",
            deadline="2026-03-01")])
    return ExtractionResult(opportunities=[])


def fit_for(_system: str, prompt: str) -> FitAssessment:
    score = 85 if "Lumen" in prompt else 45
    verdict = "strong" if score >= 75 else "poor"
    return FitAssessment(score=score, verdict=verdict, reasons=["matches sector"], risks=[],
                         eligibility_blockers=[])


def draft_for(_system: str, prompt: str) -> ApplicationDraft:
    return ApplicationDraft(
        answers=[DraftAnswer(question="What problem are you solving?",
                             answer="Farmers lose crops to disease. [[NEEDS: yield data]]",
                             needs_input=["yield data"])],
        missing_info=["yield data"],
    )


def make_llm():
    return FakeLLM({ExtractionResult: extraction_for, FitAssessment: fit_for,
                    ApplicationDraft: draft_for})


def make_scout(profile, llm, store):
    source = FixtureSource(FIXTURES / "opportunities")
    return OpportunityScout(profile, search=source, fetcher=source, llm=llm, store=store,
                            min_score=60, max_workers=2, min_content_chars=10)


def test_pipeline_scores_filters_and_drafts(profile):
    llm, store = make_llm(), Store()
    report = make_scout(profile, llm, store).run(
        ["grant accelerator hackathon AI"], results_per_query=10, today=TODAY)

    names = {o.opportunity.name: o for o in report.opportunities}
    assert names["Northstar Accelerator"].rejected == ["deadline passed (2026-03-01)"]
    assert names["Northstar Accelerator"].fit is None  # no paid call for expired programs
    shortlist = report.shortlist(60)
    assert [o.opportunity.name for o in shortlist] == ["Lumen Founders Grant"]
    assert shortlist[0].draft is not None
    assert llm.count("ApplicationDraft") == 1  # only the shortlisted one is drafted
    assert llm.count("FitAssessment") == 2  # Lumen + AgentForge, not the expired one
    assert report.errors == []

    # The profile lives in the system prompt (cacheable); the date lives in the user turn.
    fit_calls = [c for c in llm.calls if c[0] == "FitAssessment"]
    assert all("Example Startup" in system for _, system, _ in fit_calls)
    assert all("2026-11-23" not in system for _, system, _ in fit_calls)

    stored = store.list_opportunities()
    assert {r["name"] for r in stored} == {"Lumen Founders Grant", "AgentForge Global Hackathon"}


def test_second_run_is_incremental(profile):
    store = Store()
    make_scout(profile, make_llm(), store).run(["grant hackathon"], 10, today=TODAY)
    llm2 = make_llm()
    report = make_scout(profile, llm2, store).run(["grant hackathon"], 10, today=TODAY)
    assert llm2.calls == []  # every page already processed: zero tokens spent
    assert report.documents_processed == 0


def test_failed_page_is_reported_and_retried_next_run(profile):
    store = Store()

    def flaky(system, prompt):
        if "lumen-founders-grant" in prompt:
            return LLMError("output truncated")
        return extraction_for(system, prompt)

    llm = FakeLLM({ExtractionResult: flaky, FitAssessment: fit_for, ApplicationDraft: draft_for})
    report = make_scout(profile, llm, store).run(["grant hackathon"], 10, today=TODAY)
    assert any("lumen-founders-grant" in e for e in report.errors)
    assert not store.is_seen("https://example.org/lumen-founders-grant")

    report2 = make_scout(profile, make_llm(), store).run(["grant hackathon"], 10, today=TODAY)
    assert [o.opportunity.name for o in report2.shortlist(60)] == ["Lumen Founders Grant"]


def test_untrusted_content_is_wrapped(profile):
    llm = make_llm()
    make_scout(profile, llm, Store()).run(["AI trends"], 10, today=TODAY)
    prompts = [p for name, _, p in llm.calls if name == "ExtractionResult"]
    assert prompts and all(p.count("<untrusted_document") == 1 for p in prompts)


def test_markdown_report(profile):
    report = make_scout(profile, make_llm(), Store()).run(["grant hackathon accelerator"], 10,
                                                          today=TODAY)
    md = opportunities_markdown(report, 60, TODAY)
    assert md.startswith("---\ntype: opportunity-scan")
    assert "| 85 | [Lumen Founders Grant]" in md
    assert "2027-02-15 (84d)" in md
    assert "[[NEEDS: yield data]]" in md
    assert "Northstar Accelerator — deadline passed" in md
