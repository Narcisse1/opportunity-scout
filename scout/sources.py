"""Web data sources: Firecrawl and Exa for live runs, fixtures for offline demos and tests.

Every source returns plain `Document`s so the pipelines never depend on a vendor's shape.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import httpx

from .http import SourceError, request_json


@dataclass(frozen=True)
class Document:
    url: str
    title: str
    content: str
    source: str
    published: str | None = None


class SearchSource(Protocol):
    name: str

    def search(self, query: str, limit: int = 5, since_days: int | None = None) -> list[Document]: ...


class Fetcher(Protocol):
    def fetch(self, url: str) -> Document: ...


class FirecrawlSource:
    """Firecrawl v2: web search with inline markdown scraping, plus single-page scrape."""

    name = "firecrawl"

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        base_url: str = "https://api.firecrawl.dev/v2",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client()
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def search(self, query: str, limit: int = 5, since_days: int | None = None) -> list[Document]:
        body: dict = {
            "query": query,
            "limit": limit,
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
        }
        if since_days:
            body["tbs"] = "qdr:d" if since_days <= 1 else "qdr:w" if since_days <= 7 else "qdr:m"
        data = request_json(
            self.client, "POST", f"{self.base_url}/search", json=body, headers=self.headers
        )
        items = data.get("data", [])
        if isinstance(items, dict):  # v2 groups results by type: {"web": [...], "news": [...]}
            items = items.get("web", []) + items.get("news", [])
        docs = []
        for item in items:
            url = item.get("url")
            if not url:
                continue
            meta = item.get("metadata") or {}
            docs.append(
                Document(
                    url=url,
                    title=item.get("title") or meta.get("title") or "",
                    content=item.get("markdown") or item.get("description") or "",
                    source=self.name,
                    published=item.get("date") or meta.get("publishedTime"),
                )
            )
        return docs

    def fetch(self, url: str) -> Document:
        data = request_json(
            self.client,
            "POST",
            f"{self.base_url}/scrape",
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            headers=self.headers,
        )
        page = data.get("data") or {}
        meta = page.get("metadata") or {}
        if not page.get("markdown"):
            raise SourceError(f"firecrawl returned no content for {url}")
        return Document(
            url=url,
            title=meta.get("title", ""),
            content=page["markdown"],
            source=self.name,
            published=meta.get("publishedTime"),
        )


class ExaSource:
    """Exa neural search; strong for 'find pages like this' discovery and dated news."""

    name = "exa"

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        base_url: str = "https://api.exa.ai",
        max_chars: int = 12000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client()
        self.headers = {"x-api-key": api_key}
        self.max_chars = max_chars

    def _docs(self, results: list[dict]) -> list[Document]:
        return [
            Document(
                url=r["url"],
                title=r.get("title") or "",
                content=r.get("text") or "",
                source=self.name,
                published=r.get("publishedDate"),
            )
            for r in results
            if r.get("url")
        ]

    def search(self, query: str, limit: int = 5, since_days: int | None = None) -> list[Document]:
        body: dict = {
            "query": query,
            "numResults": limit,
            "type": "auto",
            "contents": {"text": {"maxCharacters": self.max_chars}},
        }
        if since_days:
            body["startPublishedDate"] = (date.today() - timedelta(days=since_days)).isoformat()
        data = request_json(
            self.client, "POST", f"{self.base_url}/search", json=body, headers=self.headers
        )
        return self._docs(data.get("results", []))

    def fetch(self, url: str) -> Document:
        data = request_json(
            self.client,
            "POST",
            f"{self.base_url}/contents",
            json={"urls": [url], "text": {"maxCharacters": self.max_chars * 3}},
            headers=self.headers,
        )
        docs = self._docs(data.get("results", []))
        if not docs or not docs[0].content:
            raise SourceError(f"exa returned no content for {url}")
        return docs[0]


class MultiSource:
    """Fan a query out to several search sources and merge results by URL."""

    name = "multi"

    def __init__(self, sources: list[SearchSource]) -> None:
        if not sources:
            raise ValueError("MultiSource needs at least one source")
        self.sources = sources

    def search(self, query: str, limit: int = 5, since_days: int | None = None) -> list[Document]:
        merged: dict[str, Document] = {}
        errors = []
        for source in self.sources:
            try:
                for doc in source.search(query, limit, since_days):
                    existing = merged.get(doc.url)
                    if existing is None or len(doc.content) > len(existing.content):
                        merged[doc.url] = doc
            except SourceError as exc:
                errors.append(f"{source.name}: {exc}")
        if errors and not merged:
            raise SourceError("; ".join(errors))
        return list(merged.values())


_WORD = re.compile(r"[a-z0-9]+")


class FixtureSource:
    """Serves JSON documents from disk. Used for `--offline` demos and tests."""

    name = "fixtures"

    def __init__(self, directory: str | Path) -> None:
        self.docs: dict[str, Document] = {}
        for path in sorted(Path(directory).glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            for item in raw if isinstance(raw, list) else [raw]:
                doc = Document(
                    url=item["url"],
                    title=item.get("title", ""),
                    content=item["content"],
                    source=self.name,
                    published=item.get("published"),
                )
                self.docs[doc.url] = doc

    def search(self, query: str, limit: int = 5, since_days: int | None = None) -> list[Document]:
        terms = set(_WORD.findall(query.lower()))
        scored = []
        for doc in self.docs.values():
            words = set(_WORD.findall(f"{doc.title} {doc.content}".lower()))
            hits = len(terms & words)
            if hits:
                scored.append((hits, doc.url, doc))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [doc for _, _, doc in scored[:limit]]

    def fetch(self, url: str) -> Document:
        try:
            return self.docs[url]
        except KeyError as exc:
            raise SourceError(f"no fixture for {url}") from exc
