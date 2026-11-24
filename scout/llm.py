"""Provider-agnostic structured LLM calls (Anthropic Claude or Google Gemini).

Every LLM step in this project asks for a Pydantic model and gets a validated instance back,
so downstream code never parses free text. Each provider wrapper also:

* turns refusals / safety blocks / truncation into typed exceptions instead of bad data,
* retries transient failures (rate limits, overloads),
* meters token usage so every run can report its cost.

Pick the provider with LLM_PROVIDER=anthropic|gemini (see `make_llm`).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from . import http

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_MODEL = DEFAULT_CLAUDE_MODEL  # backwards-compatible name
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Models that accept server-side refusal fallbacks (`fallbacks="default"`).
FALLBACK_MODELS = {
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
}

# USD per million tokens: (input, output, cache read, cache write).
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    # Gemini paid-tier list prices (approximate). The free tier costs nothing.
    "gemini": (0.30, 2.50, 0.075, 0.0),
}


def prices_for(model: str) -> tuple[float, float, float, float]:
    if model in PRICES:
        return PRICES[model]
    if model.startswith("gemini"):
        return PRICES["gemini"]
    return PRICES[DEFAULT_CLAUDE_MODEL]


class LLMError(RuntimeError):
    """The model call completed but did not produce a usable result."""


class LLMRefusal(LLMError):
    """The model (and any fallback) declined the request."""


@dataclass
class UsageMeter:
    model: str = DEFAULT_CLAUDE_MODEL
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, *, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens or 0
            self.output_tokens += output_tokens or 0
            self.cache_read_tokens += cache_read or 0
            self.cache_write_tokens += cache_write or 0

    def add(self, usage: Any) -> None:
        """Record an Anthropic `usage` object."""
        if usage is None:
            return
        self.record(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read=getattr(usage, "cache_read_input_tokens", 0),
            cache_write=getattr(usage, "cache_creation_input_tokens", 0),
        )

    def estimated_cost_usd(self) -> float:
        p_in, p_out, p_read, p_write = prices_for(self.model)
        per = 1_000_000
        return round(
            self.input_tokens * p_in / per
            + self.output_tokens * p_out / per
            + self.cache_read_tokens * p_read / per
            + self.cache_write_tokens * p_write / per,
            4,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "estimated_cost_usd": self.estimated_cost_usd(),
        }


class StructuredLLM(Protocol):
    """Anything that can turn (system, prompt) into a validated Pydantic object."""

    usage: UsageMeter

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> T: ...


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------


class ClaudeLLM:
    def __init__(
        self,
        model: str = DEFAULT_CLAUDE_MODEL,
        *,
        client: Any | None = None,
        use_fallbacks: bool | None = None,
        max_tokens: int = 16000,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=4)
        self.client = client
        self.model = model
        # Server-side refusal fallbacks exist only for some models; sending them elsewhere 400s.
        self.use_fallbacks = model in FALLBACK_MODELS if use_fallbacks is None else use_fallbacks
        self.max_tokens = max_tokens
        self.usage = UsageMeter(model=model)

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # System prompts are stable per step (volatile data such as today's date lives in
            # the user turn), so they are cache-friendly across the many calls of a run.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": prompt}],
            "output_format": schema,
        }
        response = self._call(kwargs)
        self.usage.add(getattr(response, "usage", None))

        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"model declined ({category or 'no category'})")
        if stop == "max_tokens":
            raise LLMError(f"output truncated at max_tokens={self.max_tokens}")

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LLMError(f"no structured output (stop_reason={stop})")
        return parsed

    def _call(self, kwargs: dict[str, Any]) -> Any:
        if self.use_fallbacks:
            try:
                return self.client.beta.messages.parse(
                    **kwargs, betas=[FALLBACK_BETA], fallbacks="default"
                )
            except (TypeError, AttributeError) as exc:
                # Older SDKs lack beta parse or the `fallbacks` parameter: degrade once, loudly.
                log.warning(
                    "refusal fallbacks unavailable in this SDK (%s); continuing without", exc
                )
                self.use_fallbacks = False
        return self.client.messages.parse(**kwargs)


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

RETRYABLE_CODES = {429, 500, 502, 503, 504}
BLOCKED_REASONS = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}


def _name(value: Any) -> str:
    """Enum or string -> bare upper-case name ('FinishReason.STOP' -> 'STOP')."""
    if value is None:
        return ""
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


class GeminiLLM:
    """Gemini via the google-genai SDK, using JSON-schema constrained output."""

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        max_output_tokens: int = 16000,
        attempts: int = 6,
    ) -> None:
        if client is None:
            from google import genai

            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            client = genai.Client(api_key=key)
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.attempts = attempts
        self.usage = UsageMeter(model=model)

    def _config(self, system: str, schema: type[BaseModel]) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=self.max_output_tokens,
            temperature=0.2,
        )

    def _generate(self, system: str, prompt: str, schema: type[BaseModel]) -> Any:
        config = self._config(system, schema)
        for attempt in range(self.attempts):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except Exception as exc:  # SDK raises google.genai.errors.APIError subclasses
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if code not in RETRYABLE_CODES or attempt == self.attempts - 1:
                    raise LLMError(f"gemini request failed: {exc}") from exc
                # Free-tier rate limits are per minute, so back off generously.
                delay = min(5 * 2**attempt, 60)
                log.info("gemini %s, retrying in %ss", code, delay)
                http._sleep(delay)
        raise LLMError("gemini request failed")  # pragma: no cover

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        response = self._generate(system, prompt, schema)

        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            cached = getattr(meta, "cached_content_token_count", 0) or 0
            self.usage.record(
                input_tokens=(getattr(meta, "prompt_token_count", 0) or 0) - cached,
                output_tokens=(getattr(meta, "candidates_token_count", 0) or 0)
                + (getattr(meta, "thoughts_token_count", 0) or 0),
                cache_read=cached,
            )

        feedback = getattr(response, "prompt_feedback", None)
        block = _name(getattr(feedback, "block_reason", None)) if feedback else ""
        if block and block != "BLOCK_REASON_UNSPECIFIED":
            raise LLMRefusal(f"gemini blocked the prompt ({block})")

        candidates = getattr(response, "candidates", None) or []
        finish = _name(getattr(candidates[0], "finish_reason", None)) if candidates else ""
        if finish in BLOCKED_REASONS:
            raise LLMRefusal(f"gemini declined ({finish})")
        if finish == "MAX_TOKENS":
            raise LLMError(f"output truncated at max_output_tokens={self.max_output_tokens}")

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        text = getattr(response, "text", None)
        if text:
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                raise LLMError(f"gemini output did not match {schema.__name__}: {exc}") from exc
        raise LLMError(f"no structured output (finish_reason={finish or 'unknown'})")


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def resolve_provider(provider: str | None = None) -> str:
    """Explicit argument > LLM_PROVIDER > whichever API key is configured (Anthropic first)."""
    choice = (provider or _env("LLM_PROVIDER") or "").lower()
    if choice in ("anthropic", "claude"):
        return "anthropic"
    if choice in ("gemini", "google"):
        return "gemini"
    if choice:
        raise ValueError(f"unknown LLM_PROVIDER {choice!r}; use 'anthropic' or 'gemini'")
    if _env("ANTHROPIC_API_KEY") or _env("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic"


def provider_ready(provider: str) -> bool:
    if provider == "gemini":
        return bool(_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
    return bool(_env("ANTHROPIC_API_KEY") or _env("ANTHROPIC_AUTH_TOKEN"))


def make_llm(provider: str | None = None, model: str | None = None) -> StructuredLLM:
    """Build the configured LLM. A `model` meant for the other provider is ignored, so a
    config file can name a Claude model while LLM_PROVIDER switches to Gemini."""
    name = resolve_provider(provider)
    model = _env("SCOUT_MODEL") or model
    if name == "gemini":
        return GeminiLLM(model if model and model.startswith("gemini") else DEFAULT_GEMINI_MODEL)
    return ClaudeLLM(model if model and model.startswith("claude") else DEFAULT_CLAUDE_MODEL)
