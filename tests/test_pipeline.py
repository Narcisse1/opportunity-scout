from __future__ import annotations

import csv
import io
import time
from datetime import datetime, timezone

import pytest
from conftest import FakeLLM
from fastapi.testclient import TestClient
from test_opportunities import draft_for, extraction_for, fit_for

from scout.exports import to_csv, to_ics
from scout.models import ApplicationDraft, ExtractionResult, FitAssessment
from scout.store import Store
from scout.web import create_app

REC = {
    "key": "abc123",
    "name": "Lumen Founders Grant, 2027; Africa",
    "organizer": "Lumen Foundation",
    "type": "grant",
    "url": "https://example.org/lumen",
    "deadline": "2027-02-15",
    "score": 85,
    "verdict": "strong",
    "opportunity": {"funding_or_prize": "$50,000"},
}


# --- store ----------------------------------------------------------------------


def test_status_survives_rescans_and_migration(tmp_path):
    path = tmp_path / "s.db"
    store = Store(path)
    store.upsert_opportunity("abc123", REC, "t0")
    assert store.set_status("abc123", "applying", "draft v1 done", "t1")
    store.upsert_opportunity("abc123", {**REC, "score": 90}, "t2")  # a later scan
    store.close()

    [row] = Store(path).list_opportunities(statuses=["applying"])  # reopening re-runs migrations
    assert (row["status"], row["notes"], row["score"]) == ("applying", "draft v1 done", 90)


def test_set_status_validation():
    store = Store()
    assert store.set_status("missing", "saved", None, "t") is False
    with pytest.raises(ValueError):
        store.set_status("abc123", "maybe", None, "t")


# --- exports --------------------------------------------------------------------


def test_ics_has_escaped_all_day_event_with_reminder():
    ics = to_ics(
        [{**REC, "status": "saved", "notes": "ask Ana"}, {"name": "No deadline"}],
        now=datetime(2026, 11, 24, tzinfo=timezone.utc),
    )
    assert ics.startswith("BEGIN:VCALENDAR\r\n") and ics.endswith("END:VCALENDAR\r\n")
    assert ics.count("BEGIN:VEVENT") == 1  # items without a deadline are skipped
    assert "DTSTART;VALUE=DATE:20270215" in ics and "DTEND;VALUE=DATE:20270216" in ics
    assert "SUMMARY:Deadline: Lumen Founders Grant\\, 2027\\; Africa" in ics
    assert "TRIGGER:-P7D" in ics
    assert all(len(line.encode()) <= 75 for line in ics.split("\r\n"))


def test_csv_export():
    rows = list(csv.DictReader(io.StringIO(to_csv([{**REC, "status": "saved", "notes": ""}]))))
    assert rows[0]["name"] == REC["name"] and rows[0]["funding_or_prize"] == "$50,000"


# --- API ------------------------------------------------------------------------


def client() -> TestClient:
    def llm():
        return FakeLLM(
            {ExtractionResult: extraction_for, FitAssessment: fit_for, ApplicationDraft: draft_for}
        )

    return TestClient(create_app(llm_factory=llm, store=Store()))


def test_track_from_offline_scan_and_export():
    c = client()
    job = c.post(
        "/api/opportunities/scan",
        json={
            "profile": {"name": "X", "one_liner": "Y"},
            "queries": ["grant"],
            "offline": True,
        },
    ).json()
    for _ in range(200):
        job = c.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    opp = next(o for o in job["result"]["opportunities"] if o["name"] == "Lumen Founders Grant")

    assert c.put(f"/api/pipeline/{opp['key']}", json={"status": "saved"}).status_code == 404
    r = c.put(f"/api/pipeline/{opp['key']}", json={"status": "saved", "record": opp})
    assert r.status_code == 200
    c.put(f"/api/pipeline/{opp['key']}", json={"status": "applying", "notes": "due soon"})

    [tracked] = c.get("/api/pipeline").json()
    assert (tracked["status"], tracked["notes"]) == ("applying", "due soon")
    ics = c.get("/api/pipeline.ics")
    assert ics.headers["content-type"].startswith("text/calendar")
    assert "Deadline: Lumen Founders Grant" in ics.text
    assert "Lumen Founders Grant" in c.get("/api/pipeline.csv").text

    c.put(f"/api/pipeline/{opp['key']}", json={"status": "dismissed"})
    assert c.get("/api/pipeline").json() == []


def test_pipeline_rejects_bad_input():
    c = client()
    assert c.put("/api/pipeline/k", json={"status": "maybe"}).status_code == 422
    assert c.put("/api/pipeline/k", json={"status": "saved", "record": {}}).status_code == 422


def test_download_links_accept_token_query(monkeypatch):
    monkeypatch.setenv("SCOUT_ACCESS_TOKEN", "s3cret")
    c = client()
    assert c.get("/api/pipeline.ics").status_code == 401
    assert c.get("/api/pipeline.ics?token=s3cret").status_code == 200
