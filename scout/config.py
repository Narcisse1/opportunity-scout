"""Run configuration (YAML) and environment-based wiring of sources and publishers."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .models import Competitor, StartupProfile


class OpportunityConfig(BaseModel):
    queries: list[str] = Field(default_factory=list)
    results_per_query: int = 5
    min_score: int = 60
    draft_applications: bool = True


class IntelConfig(BaseModel):
    competitors: list[Competitor] = Field(default_factory=list)
    news_days: int = 7


class OutputConfig(BaseModel):
    obsidian_vault: str = "./vault"
    notion: bool = False


class AppConfig(BaseModel):
    model: str = "claude-opus-5-5"
    database: str = "./data/scout.db"
    max_workers: int = 4
    opportunities: OpportunityConfig = Field(default_factory=OpportunityConfig)
    intel: IntelConfig = Field(default_factory=IntelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_config(path: str | Path) -> AppConfig:
    return AppConfig.model_validate(load_yaml(path))


def load_profile(path: str | Path) -> StartupProfile:
    return StartupProfile.model_validate(load_yaml(path))


def env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None
