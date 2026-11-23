"""Thin, typed wrapper around the Claude Messages API.

Every LLM step in this project returns a Pydantic model via structured outputs
(`messages.parse`), so downstream code never parses free text. The wrapper also:

* enables server-side refusal fallbacks (`fallbacks="default"`) when the SDK supports it,
* turns refusals / truncation into typed exceptions instead of silently bad data,
* meters token usage so every run can report its cost.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens for claude-opus-5-5 (input, output, cache read, cache write).
PRICES = {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00}


class LLMError(RuntimeError):
    """The model call completed but did not produce a usable result."""


class LLMRefusal(LLMError):
    """The model (and any fallback) declined the request."""


@dataclass
class UsageMeter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        with self._lock:
            self.calls += 1
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def estimated_cost_usd(self) -> float:
        per = 1_000_000
        return round(
            self.input_tokens * PRICES["input"] / per
            + self.output_tokens * PRICES["output"] / per
            + self.cache_read_tokens * PRICES["cache_read"] / per
            + self.cache_write_tokens * PRICES["cache_write"] / per,
            4,
        )

    def summary(self) -> dict[str, Any]:
        return {
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


class ClaudeLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        client: Any | None = None,
        use_fallbacks: bool = True,
        max_tokens: int = 16000,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=4)
        self.client = client
        self.model = model
        self.use_fallbacks = use_fallbacks
        self.max_tokens = max_tokens
        self.usage = UsageMeter()

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
                log.warning("refusal fallbacks unavailable in this SDK (%s); continuing without", exc)
                self.use_fallbacks = False
        return self.client.messages.parse(**kwargs)
