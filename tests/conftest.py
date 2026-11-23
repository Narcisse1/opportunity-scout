from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scout import http
from scout.llm import UsageMeter
from scout.models import StartupProfile

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


class FakeLLM:
    """Routes each structured() call to a handler chosen by the requested schema."""

    def __init__(self, handlers: dict[type, Callable[[str, str], object]]) -> None:
        self.handlers = handlers
        self.calls: list[tuple[str, str, str]] = []
        self.usage = UsageMeter()

    def structured(self, *, system, prompt, schema):
        self.calls.append((schema.__name__, system, prompt))
        result = self.handlers[schema](system, prompt)
        if isinstance(result, Exception):
            raise result
        return result

    def count(self, schema_name: str) -> int:
        return sum(1 for name, _, _ in self.calls if name == schema_name)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http, "_sleep", lambda _s: None)


@pytest.fixture
def profile() -> StartupProfile:
    return StartupProfile(
        name="Example Startup",
        one_liner="AI crop-disease diagnosis for smallholder farmers",
        stage="pre-seed",
        sectors=["agritech", "AI"],
        country="Cameroon",
        exclude_keywords=["crypto"],
    )
