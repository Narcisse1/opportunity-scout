"""Shared HTTP helper with bounded retries for third-party APIs."""

from __future__ import annotations

import time
from typing import Any

import httpx

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}

# Indirection so tests can disable real sleeping.
_sleep = time.sleep


class SourceError(RuntimeError):
    """A data source (search, scrape, publish target) failed."""


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json: Any | None = None,
    headers: dict[str, str] | None = None,
    attempts: int = 4,
    timeout: float = 90.0,
) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.request(method, url, json=json, headers=headers, timeout=timeout)
        except httpx.TransportError as exc:
            last = exc
            delay = min(2**attempt, 20)
        else:
            if resp.status_code < 400:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise SourceError(f"{method} {url}: response was not JSON") from exc
            if resp.status_code not in RETRYABLE:
                raise SourceError(f"{method} {url} -> {resp.status_code}: {resp.text[:300]}")
            last = SourceError(f"{method} {url} -> {resp.status_code}")
            retry_after = resp.headers.get("retry-after", "")
            delay = float(retry_after) if retry_after.isdigit() else min(2**attempt, 20)
        if attempt < attempts - 1:
            _sleep(delay)
    raise SourceError(f"{method} {url} failed after {attempts} attempts: {last}") from last
