from __future__ import annotations

from datetime import date

from scout.emails import upcoming_deadlines, weekly_email

TODAY = date(2026, 11, 24)


def program(**kw):
    base = {
        "name": "Lumen Founders Grant",
        "type": "grant",
        "url": "https://grants.example.org/lumen",
        "deadline": "2026-12-15",
        "score": 82,
        "opportunity": {
            "organizer": "Lumen",
            "summary": "Grant for AI startups.",
            "funding_or_prize": "$25,000",
        },
        "fit": {"reasons": ["Open to Cameroon"], "eligibility_blockers": []},
    }
    return {**base, **kw}


def test_scraped_text_is_escaped():
    evil = program(name='<script>alert("x")</script>', url='javascript:"><img>')
    html = weekly_email("Ekang", [evil], [], TODAY)["html"]
    assert "<script>" not in html and "&lt;script&gt;" in html and '"><img>' not in html


def test_subject_and_text_versions():
    email = weekly_email("Ekang", [program(), program(name="Second")], [], TODAY)
    assert email["subject"] == "2 new funding programs for Ekang"
    assert "21 days left" in email["html"] and "- Second (fit 82)" in email["text"]
    quiet = weekly_email("Ekang", [], [], TODAY)
    assert "nothing new" in quiet["subject"] and "No new programs" in quiet["text"]


def test_upcoming_deadlines_window():
    rows = [
        program(name="soon", deadline="2026-12-01", status="saved"),
        program(name="later", deadline="2027-06-01", status="saved"),
        program(name="past", deadline="2026-11-01", status="applying"),
        program(name="rolling", deadline=None, status="saved"),
    ]
    assert [r["name"] for r in upcoming_deadlines(rows, TODAY, 45)] == ["soon"]
    email = weekly_email("Ekang", [], upcoming_deadlines(rows, TODAY, 45), TODAY)
    assert email["subject"] == "1 application deadline coming up for Ekang"
