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
    assert set(health) == {"llm_provider", "llm_ready", "firecrawl", "exa", "token_required"}
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


def test_access_token(monkeypatch):
    monkeypatch.setenv("SCOUT_ACCESS_TOKEN", "s3cret")
    c = client()
    body = {"profile": PROFILE, "queries": ["grant"], "offline": True}
    assert c.post("/api/opportunities/scan", json=body).status_code == 401
    ok = c.post("/api/opportunities/scan", json=body, headers={"X-Access-Token": "s3cret"})
    assert ok.status_code == 200
