"""Pydantic schemas. The `*Result`/`Assessment`/`Draft`/`Analysis`/`Digest` classes double as
structured-output schemas for Claude, so they stay flat and free of numeric constraints
(ranges are enforced in code instead)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Inputs written by the user
# ---------------------------------------------------------------------------


class StartupProfile(BaseModel):
    name: str
    one_liner: str
    description: str = ""
    stage: str = ""  # e.g. "pre-seed", "seed"
    sectors: list[str] = Field(default_factory=list)
    country: str = ""
    regions_served: list[str] = Field(default_factory=list)
    founded_year: int | None = None
    team: list[str] = Field(default_factory=list)
    funding_raised: str = ""
    traction: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    writing_samples: list[str] = Field(default_factory=list)


class Competitor(BaseModel):
    name: str
    pages: dict[str, str] = Field(default_factory=dict)  # label -> URL (pricing, changelog...)
    news_query: str | None = None


# ---------------------------------------------------------------------------
# Opportunity tracker (LLM outputs)
# ---------------------------------------------------------------------------

OpportunityType = Literal[
    "grant", "accelerator", "hackathon", "competition", "fellowship", "credits", "other"
]


class ExtractedOpportunity(BaseModel):
    name: str
    organizer: str | None
    type: OpportunityType
    url: str | None
    deadline: str | None = Field(description="YYYY-MM-DD if stated on the page, else null")
    funding_or_prize: str | None
    equity_taken: str | None
    location_requirements: str | None
    stage_requirements: str | None
    eligibility: list[str]
    summary: str
    application_questions: list[str]


class ExtractionResult(BaseModel):
    opportunities: list[ExtractedOpportunity]


class FitAssessment(BaseModel):
    score: int = Field(description="0-100 fit score")
    verdict: Literal["strong", "possible", "poor", "ineligible"]
    reasons: list[str]
    risks: list[str]
    eligibility_blockers: list[str]


class DraftAnswer(BaseModel):
    question: str
    answer: str
    needs_input: list[str]


class ApplicationDraft(BaseModel):
    answers: list[DraftAnswer]
    missing_info: list[str]


# ---------------------------------------------------------------------------
# Competitor intelligence (LLM outputs)
# ---------------------------------------------------------------------------

FindingCategory = Literal[
    "pricing", "product", "positioning", "hiring", "funding", "partnership", "other"
]


class ChangeFinding(BaseModel):
    category: FindingCategory
    summary: str
    evidence_quote: str = Field(description="Verbatim text copied from the evidence")
    source_url: str
    importance: Literal["high", "medium", "low"]


class CompetitorAnalysis(BaseModel):
    findings: list[ChangeFinding]


class IntelDigest(BaseModel):
    headline: str
    implications: list[str]
    recommended_actions: list[str]
