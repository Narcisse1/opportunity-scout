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
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from . import http

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "auto"  # newest stable Flash model the key can use (see pick_gemini_model)
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


_SPECIALISED = ("image", "tts", "audio", "live", "embedding", "vision", "learnlm", "robotics")
_PREVIEW = ("preview", "exp")


def rank_gemini_models(models: Any) -> list[str]:
    """General-purpose Flash models that support generateContent, best first.

    Google renames and retires Gemini model IDs often, so the app asks the API which models
    this key can use instead of hard-coding one. Order: stable Flash (newest first), then
    stable Flash-Lite, then previews. Each model has its own quota, so later entries are
    fallbacks when an earlier one is overloaded or out of free-tier requests."""
    ranked = []
    for m in models:
        name = str(getattr(m, "name", "") or "").removeprefix("models/")
        actions = getattr(m, "supported_actions", None) or getattr(
            m, "supported_generation_methods", None
        )
        if actions and "generateContent" not in actions:
            continue
        if not name.startswith("gemini") or "flash" not in name:
            continue
        if any(word in name for word in _SPECIALISED):
            continue
        tier = 2 if any(w in name for w in _PREVIEW) else 1 if "lite" in name else 0
        version = tuple(int(n) for n in re.findall(r"\d+", name.split("flash")[0]))
        # Within a tier: newest version first, then bare names over dated variants.
        ranked.append(((tier, tuple(-v for v in version), len(name)), name))
    return [name for _, name in sorted(ranked)]


def pick_gemini_model(models: Any) -> str | None:
    ranked = rank_gemini_models(models)
    return ranked[0] if ranked else None


class LLMQuotaExceeded(LLMError):
    """Every usable model is out of quota; retrying now would only waste time."""


_candidates: dict[int, list[str]] = {}  # id(client) -> ranked models, shared per process
_unavailable: dict[str, float] = {}  # model -> unix time until which it should not be used
_discover_lock = threading.Lock()

_RETRY_IN = re.compile(r"retry in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.I)


def _retry_after_seconds(message: str, default: float) -> float:
    m = _RETRY_IN.search(message)
    if not m or not any(m.groups()):
        return default
    h, mins, secs = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mins * 60 + secs


def _is_daily_quota(message: str) -> bool:
    text = message.lower()
    return "perday" in text or "per day" in text or "requests per day" in text


class GeminiLLM:
    """Gemini via the google-genai SDK, using JSON-schema constrained output."""

    def __init__(
        self,
        model: str | None = DEFAULT_GEMINI_MODEL,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        max_output_tokens: int = 16000,
        attempts: int = 6,
    ) -> None:
        if client is None:
            client = _gemini_client(api_key)
        self.client = client
        self.model = None if model in (None, "", "auto") else model
        self.max_output_tokens = max_output_tokens
        self.attempts = attempts
        self.usage = UsageMeter(model=self.model or "gemini")

    def _ranked_models(self) -> list[str]:
        with _discover_lock:
            key = id(self.client)
            if key not in _candidates:
                try:
                    ranked = rank_gemini_models(self.client.models.list())
                except Exception as exc:
                    raise LLMError(f"could not list Gemini models for this key: {exc}") from exc
                if not ranked:
                    raise LLMError("no Gemini Flash model with generateContent is available")
                log.warning("Gemini models available (best first): %s", ", ".join(ranked))
                _candidates[key] = ranked
            return _candidates[key]

    def _next_model(self) -> str:
        """Best model not currently marked unavailable; raises when all are exhausted."""
        now = time.time()
        for name in self._ranked_models():
            if _unavailable.get(name, 0) <= now:
                return name
        soonest = min(_unavailable.get(n, now) for n in self._ranked_models()) - now
        hours = max(soonest, 0) / 3600
        when = f"in about {hours:.0f}h" if hours >= 1 else "within the hour"
        raise LLMQuotaExceeded(
            "Gemini free-tier quota is used up for every available model "
            f"(resets {when}). Enable billing on the Gemini API key or switch to Claude "
            "for higher limits."
        )

    def _mark_unavailable(self, model: str, seconds: float, why: str) -> None:
        with _discover_lock:
            _unavailable[model] = max(_unavailable.get(model, 0), time.time() + seconds)
        log.warning("Gemini model %s set aside for %.0fs (%s)", model, seconds, why)

    def _set_model(self, model: str) -> None:
        self.model = model
        self.usage.model = model

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
        pinned = self.model is not None  # an explicit model is tried first, then the ranking
        if not pinned or _unavailable.get(self.model, 0) > time.time():
            self._set_model(self._next_model())
        overloaded = 0
        for attempt in range(self.attempts):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except Exception as exc:  # SDK raises google.genai.errors.APIError subclasses
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                message = str(exc)
                if code == 404:
                    # Renamed or retired model: never use it again in this process.
                    self._mark_unavailable(self.model, 365 * 86400, "not found")
                    self._set_model(self._next_model())
                    continue
                if code == 429 and _is_daily_quota(message):
                    # Daily quotas are per model: move on instead of waiting hours.
                    wait = _retry_after_seconds(message, 6 * 3600)
                    self._mark_unavailable(self.model, wait, "daily quota used up")
                    self._set_model(self._next_model())
                    continue
                if code == 503:
                    overloaded += 1
                    if overloaded >= 2:
                        # Persistently overloaded: rest this model briefly, try another one.
                        self._mark_unavailable(self.model, 300, "overloaded")
                        try:
                            self._set_model(self._next_model())
                            overloaded = 0
                            continue
                        except LLMQuotaExceeded:
                            pass  # nothing else to try; fall through to backoff
                if code not in RETRYABLE_CODES or attempt == self.attempts - 1:
                    raise LLMError(f"gemini request failed: {exc}") from exc
                # Per-minute rate limits and brief overloads: back off and retry.
                delay = _retry_after_seconds(message, min(5 * 2**attempt, 60))
                log.info("gemini %s, retrying in %.0fs", code, delay)
                http._sleep(min(delay, 60))
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


_clients: dict[str, Any] = {}


def _gemini_client(api_key: str | None) -> Any:
    """One SDK client per key, so model discovery happens once per process, not per job."""
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    with _discover_lock:
        if key not in _clients:
            _clients[key] = genai.Client(api_key=key or None)
        return _clients[key]


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
