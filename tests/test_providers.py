from __future__ import annotations

from types import SimpleNamespace

import pytest

from scout.llm import (
    ClaudeLLM,
    GeminiLLM,
    LLMError,
    LLMRefusal,
    make_llm,
    resolve_provider,
)
from scout.models import FitAssessment

FIT = FitAssessment(score=80, verdict="strong", reasons=["r"], risks=[], eligibility_blockers=[])

KEYS = [
    "LLM_PROVIDER",
    "SCOUT_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


# --- provider selection -----------------------------------------------------------


def test_resolve_provider(monkeypatch):
    assert resolve_provider() == "anthropic"
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert resolve_provider() == "gemini"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert resolve_provider() == "anthropic"  # both keys: Claude unless told otherwise
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert resolve_provider() == "gemini"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError):
        resolve_provider()


def test_make_llm_ignores_model_for_other_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    llm = make_llm(model="claude-opus-5-5")  # e.g. from scout.yaml
    assert isinstance(llm, GeminiLLM) and llm.model == "gemini-flash-latest"
    monkeypatch.setenv("SCOUT_MODEL", "gemini-2.5-pro")
    assert make_llm().model == "gemini-2.5-pro"


def test_claude_fallbacks_only_on_supported_models():
    client = SimpleNamespace()
    assert ClaudeLLM("claude-opus-5-5", client=client).use_fallbacks is True
    assert ClaudeLLM("claude-haiku-5-5", client=client).use_fallbacks is False


def test_cost_uses_model_prices():
    llm = ClaudeLLM("claude-haiku-5-5", client=SimpleNamespace())
    llm.usage.record(input_tokens=1_000_000, output_tokens=1_000_000)
    assert llm.usage.estimated_cost_usd() == pytest.approx(0.60)


# --- Gemini wrapper ---------------------------------------------------------------


class _ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def _gemini_response(parsed=None, text=None, finish="STOP", block=None):
    return SimpleNamespace(
        parsed=parsed,
        text=text,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=1200,
            candidates_token_count=300,
            cached_content_token_count=200,
            thoughts_token_count=50,
        ),
    )


class _GeminiClient:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_gemini_returns_parsed_and_meters_usage():
    client = _GeminiClient(_gemini_response(parsed=FIT))
    llm = GeminiLLM(client=client)
    assert llm.structured(system="SYS", prompt="hi", schema=FitAssessment) is FIT
    call = client.calls[0]
    assert call["model"] == "gemini-flash-latest" and call["contents"] == "hi"
    assert call["config"].system_instruction == "SYS"
    assert call["config"].response_mime_type == "application/json"
    summary = llm.usage.summary()
    assert (summary["input_tokens"], summary["output_tokens"], summary["cache_read_tokens"]) == (
        1000,
        350,
        200,
    )


def test_gemini_falls_back_to_text_json():
    client = _GeminiClient(_gemini_response(text=FIT.model_dump_json()))
    assert GeminiLLM(client=client).structured(system="s", prompt="p", schema=FitAssessment) == FIT


def test_gemini_retries_rate_limits_then_succeeds():
    client = _GeminiClient(_ApiError(429), _ApiError(503), _gemini_response(parsed=FIT))
    assert GeminiLLM(client=client).structured(system="s", prompt="p", schema=FitAssessment) is FIT
    assert len(client.calls) == 3


def test_gemini_does_not_retry_bad_requests():
    client = _GeminiClient(_ApiError(400))
    with pytest.raises(LLMError, match="400"):
        GeminiLLM(client=client).structured(system="s", prompt="p", schema=FitAssessment)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "response,exc",
    [
        (_gemini_response(finish="SAFETY"), LLMRefusal),
        (_gemini_response(block=SimpleNamespace(name="PROHIBITED_CONTENT")), LLMRefusal),
        (_gemini_response(finish="MAX_TOKENS"), LLMError),
        (_gemini_response(text="not json"), LLMError),
        (_gemini_response(), LLMError),
    ],
)
def test_gemini_typed_errors(response, exc):
    with pytest.raises(exc):
        GeminiLLM(client=_GeminiClient(response)).structured(
            system="s", prompt="p", schema=FitAssessment
        )
