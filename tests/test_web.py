from __future__ import annotations

import time

from conftest import FakeLLM
from fastapi.testclient import TestClient
from test_intel import analysis, digest
from test_opportunities import draft_for, extraction_for, fit_for

from scout.models import (
    ApplicationDraft,
    CompetitorAnalysis,
    ExtractionResult,
    FitAssessment,
    IntelDigest,
)
from scout.store import Store
from scout.web import create_app

PROFILE = {"name": "Example Startup", "one_liner": "AI crop diagnosis", "country": "Cameroon"}
RIVAL = {
    "name": "RivalAI",
    "news_query": "RivalAI",
    "pages": {
        "pricing": "https://rival.example.com/pricing",
        "changelog": "https://rival.example.com/changelog",
    },
}


def fake_llm():
    return FakeLLM(
        {
            ExtractionResult: extraction_for,
            FitAssessment: fit_for,
            ApplicationDraft: draft_for,
            CompetitorAnalysis: analysis,
            IntelDigest: digest,
        }
    )


def client() -> TestClient:
    return TestClient(create_app(llm_factory=fake_llm, store=Store()))


def wait(c: TestClient, job_id: str) -> dict:
    for _ in range(200):
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_index_and_health():
    c = client()
    page = c.get("/")
    assert page.status_code == 200 and "Opportunity Scout" in page.text
    health = c.get("/api/health").json()
    assert set(health) == {
        "llm_provider",
        "llm_ready",
        "firecrawl",
        "exa",
        "token_required",
        "guest_access",
        "guest_live_runs_left",
    }
    assert health["llm_ready"] is True  # a test LLM is injected


def test_offline_scan_job():
    c = client()
    job = c.post(
        "/api/opportunities/scan",
        json={
            "profile": PROFILE,
            "queries": ["grant hackathon accelerator"],
            "offline": True,
            "results_per_query": 10,
        },
    ).json()
    done = wait(c, job["id"])
    assert done["status"] == "done", done["error"]
    shortlisted = [o["name"] for o in done["result"]["opportunities"] if o["shortlisted"]]
    assert shortlisted == ["Lumen Founders Grant"]
    assert done["result"]["markdown"].startswith("---")


def test_offline_intel_demo_detects_changes_between_weeks():
    c = client()
    base = {"competitors": [RIVAL], "offline": True}
    week1 = wait(c, c.post("/api/intel/run", json={**base, "fixture_set": "week1"}).json()["id"])
    assert week1["result"]["digest"] is None
    week2 = wait(c, c.post("/api/intel/run", json={**base, "fixture_set": "week2"}).json()["id"])
    comp = week2["result"]["competitors"][0]
    assert [p["status"] for p in comp["pages"]] == ["changed", "changed"]
    assert len(comp["findings"]) == 3 and comp["ungrounded_dropped"] == 1


def test_validation_and_limits():
    c = client()
    assert (
        c.post("/api/opportunities/scan", json={"profile": PROFILE, "queries": []}).status_code
        == 422
    )
    too_many = {"profile": PROFILE, "queries": ["q"] * 11, "offline": True}
    assert c.post("/api/opportunities/scan", json=too_many).status_code == 422
    assert c.get("/api/jobs/nope").status_code == 404


def test_live_mode_without_keys_is_rejected(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    r = client().post("/api/opportunities/scan", json={"profile": PROFILE, "queries": ["q"]})
    assert r.status_code == 400 and "offline demo" in r.json()["detail"]


def test_token_required_when_guests_disabled(monkeypatch):
    monkeypatch.setenv("SCOUT_ACCESS_TOKEN", "s3cret")
    monkeypatch.setenv("SCOUT_GUEST_ACCESS", "off")
    c = client()
    body = {"profile": PROFILE, "queries": ["grant"], "offline": True}
    assert c.post("/api/opportunities/scan", json=body).status_code == 401
    ok = c.post("/api/opportunities/scan", json=body, headers={"X-Access-Token": "s3cret"})
    assert ok.status_code == 200
    assert c.get("/api/health").json()["token_required"] is True


class FakeLive:
    """Live search stand-in backed by the offline fixtures, so no network is used."""

    def __init__(self):
        from scout.sources import FixtureSource
        from scout.web import FIXTURES

        self.source = FixtureSource(FIXTURES / "opportunities")

    def __call__(self):
        return self.source, self.source


def guest_client(monkeypatch, per_visitor=2, per_day=10) -> TestClient:
    monkeypatch.setenv("SCOUT_ACCESS_TOKEN", "s3cret")
    monkeypatch.delenv("SCOUT_GUEST_ACCESS", raising=False)
    monkeypatch.setenv("SCOUT_GUEST_RUNS_PER_VISITOR", str(per_visitor))
    monkeypatch.setenv("SCOUT_GUEST_RUNS_PER_DAY", str(per_day))
    app = create_app(llm_factory=fake_llm, store=Store(), live_sources=FakeLive())
    return TestClient(app)


def test_guests_can_use_the_app_without_a_token(monkeypatch):
    c = guest_client(monkeypatch)
    health = c.get("/api/health").json()
    assert health["guest_access"] is True and health["token_required"] is False
    assert health["guest_live_runs_left"] == 2
    sample = {"profile": PROFILE, "queries": ["grant"], "offline": True}
    for _ in range(5):  # sample-page runs are unlimited
        assert c.post("/api/opportunities/scan", json=sample).status_code == 200
    assert c.get("/api/pipeline").status_code == 200
    assert c.get("/api/pipeline.csv").status_code == 200
    # a wrong token is still an error rather than silently becoming a guest
    bad = c.post("/api/opportunities/scan", json=sample, headers={"X-Access-Token": "nope"})
    assert bad.status_code == 401


def test_guest_live_runs_are_limited_and_capped(monkeypatch):
    c = guest_client(monkeypatch, per_visitor=2)
    live = {"profile": PROFILE, "queries": ["a", "b", "c", "d", "e"], "results_per_query": 10}
    first = c.post("/api/opportunities/scan", json=live)
    assert first.status_code == 200
    done = wait(c, first.json()["id"])
    assert done["status"] == "done", done["error"]
    assert c.post("/api/opportunities/scan", json=live).status_code == 200
    third = c.post("/api/opportunities/scan", json=live)
    assert third.status_code == 429 and "Sample" in third.json()["detail"]
    assert c.get("/api/health").json()["guest_live_runs_left"] == 0
    # the owner is not limited
    owner = c.post("/api/opportunities/scan", json=live, headers={"X-Access-Token": "s3cret"})
    assert owner.status_code == 200


def test_guest_daily_total_is_shared(monkeypatch):
    c = guest_client(monkeypatch, per_visitor=5, per_day=1)
    live = {"profile": PROFILE, "queries": ["grant"]}
    first = c.post("/api/opportunities/scan", json=live, headers={"X-Forwarded-For": "1.1.1.1"})
    assert first.status_code == 200
    other = c.post("/api/opportunities/scan", json=live, headers={"X-Forwarded-For": "2.2.2.2"})
    assert other.status_code == 429 and "used up" in other.json()["detail"]


def test_weekly_digest_is_owner_only_and_returns_an_email(monkeypatch):
    c = guest_client(monkeypatch)
    body = {
        "profile": PROFILE,
        "queries": ["grant hackathon accelerator"],
        "results_per_query": 10,
        "app_url": "https://scout.example.com",
    }
    assert c.post("/api/digest/weekly", json=body).status_code == 401
    r = c.post("/api/digest/weekly", json=body, headers={"X-Access-Token": "s3cret"})
    assert r.status_code == 200
    email = r.json()
    assert email["new_count"] == 1 and email["subject"].startswith("1 new funding program for")
    assert "Lumen Founders Grant" in email["html"] and "https://scout.example.com" in email["html"]
    assert "Lumen Founders Grant" in email["text"]
    # next week the same pages are skipped, so nothing is reported twice
    again = c.post("/api/digest/weekly", json=body, headers={"X-Access-Token": "s3cret"}).json()
    assert again["new_count"] == 0 and "nothing new" in again["subject"]


def test_email_preview_uses_results_on_screen(monkeypatch):
    c = guest_client(monkeypatch)
    sample = {"profile": PROFILE, "queries": ["grant"], "offline": True, "results_per_query": 10}
    done = wait(c, c.post("/api/opportunities/scan", json=sample).json()["id"])
    preview = c.post(
        "/api/digest/preview",
        json={"company": "Ekang Agritech", "opportunities": done["result"]["opportunities"]},
    ).json()
    assert "Lumen Founders Grant" in preview["html"] and "Ekang Agritech" in preview["subject"]
