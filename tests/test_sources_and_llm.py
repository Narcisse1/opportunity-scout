from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
import respx

from scout.http import SourceError
from scout.llm import ClaudeLLM, LLMError, LLMRefusal
from scout.models import FitAssessment
from scout.publish import NotionPublisher, markdown_to_blocks
from scout.sources import ExaSource, FirecrawlSource, MultiSource
from scout.store import Store, content_hash

# --- Firecrawl / Exa parsing & retries --------------------------------------------


@respx.mock
def test_firecrawl_search_v2_shape():
    route = respx.post("https://api.firecrawl.dev/v2/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "web": [
                        {"url": "https://a.org", "title": "A", "markdown": "# A page"},
                        {
                            "url": "https://b.org",
                            "metadata": {"title": "B"},
                            "description": "short",
                        },
                        {"title": "no url"},
                    ]
                },
            },
        )
    )
    docs = FirecrawlSource("fc-key").search("ai grants", limit=3, since_days=7)
    assert [(d.url, d.title, d.content) for d in docs] == [
        ("https://a.org", "A", "# A page"),
        ("https://b.org", "B", "short"),
    ]
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer fc-key"
    assert b'"tbs":"qdr:w"' in sent.content.replace(b" ", b"")


@respx.mock
def test_firecrawl_retries_on_429_then_succeeds():
    respx.post("https://api.firecrawl.dev/v2/scrape").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "1"}),
            httpx.Response(200, json={"data": {"markdown": "body", "metadata": {"title": "T"}}}),
        ]
    )
    doc = FirecrawlSource("k").fetch("https://a.org")
    assert doc.content == "body" and doc.title == "T"


@respx.mock
def test_client_errors_are_not_retried():
    route = respx.post("https://api.exa.ai/search").mock(return_value=httpx.Response(401))
    with pytest.raises(SourceError, match="401"):
        ExaSource("bad").search("x")
    assert route.call_count == 1


@respx.mock
def test_exa_search():
    respx.post("https://api.exa.ai/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://n.org",
                        "title": "N",
                        "text": "news",
                        "publishedDate": "2026-11-20",
                    }
                ]
            },
        )
    )
    [doc] = ExaSource("k").search("rival", since_days=7)
    assert doc.published == "2026-11-20" and doc.source == "exa"


class _Fixed:
    def __init__(self, name, docs=None, error=None):
        self.name, self.docs, self.error = name, docs or [], error

    def search(self, query, limit=5, since_days=None):
        if self.error:
            raise SourceError(self.error)
        return self.docs


def test_multisource_merges_and_tolerates_partial_failure():
    from scout.sources import Document

    short = Document("https://a.org", "A", "x", "s1")
    long = Document("https://a.org", "A", "much longer", "s2")
    merged = MultiSource([_Fixed("s1", [short]), _Fixed("s2", [long]), _Fixed("s3", error="down")])
    assert [d.content for d in merged.search("q")] == ["much longer"]
    with pytest.raises(SourceError):
        MultiSource([_Fixed("s3", error="down")]).search("q")


class _Fetch:
    def __init__(self, name, doc=None):
        self.name, self.doc, self.calls = name, doc, 0

    def fetch(self, url):
        self.calls += 1
        if self.doc is None:
            raise SourceError(f"{self.name} refused {url}")
        return self.doc


def test_fallback_fetcher_tries_next_provider():
    from scout.sources import Document, FallbackFetcher

    good = Document("https://fb.com/x", "X", "page text", "exa")
    first, second = _Fetch("firecrawl"), _Fetch("exa", good)
    assert FallbackFetcher([first, second]).fetch("https://fb.com/x") is good
    with pytest.raises(SourceError, match="firecrawl refused.*exa refused"):
        FallbackFetcher([_Fetch("firecrawl"), _Fetch("exa")]).fetch("https://fb.com/x")


def test_scout_uses_search_text_when_every_scraper_refuses(profile):
    from datetime import date

    from conftest import FakeLLM

    from scout.models import ExtractionResult
    from scout.opportunities import OpportunityScout
    from scout.sources import Document

    snippet = Document("https://fb.com/post", "Grant", "Grant details. " * 30, "exa")
    llm = FakeLLM({ExtractionResult: lambda s, p: ExtractionResult(opportunities=[])})
    scout = OpportunityScout(
        profile,
        search=_Fixed("s", [snippet]),
        fetcher=_Fetch("firecrawl"),
        llm=llm,
        store=Store(),
        min_content_chars=5000,
    )
    report = scout.run(["grant"], today=date(2026, 11, 24))
    assert report.errors == [] and report.documents_processed == 1
    assert "Grant details." in llm.calls[0][2]


# --- Claude wrapper ---------------------------------------------------------------


def _response(**kw):
    base = dict(
        stop_reason="end_turn",
        stop_details=None,
        parsed_output=None,
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=200,
            cache_read_input_tokens=500,
            cache_creation_input_tokens=0,
        ),
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Client:
    def __init__(self, response, beta_ok=True):
        self.kwargs = None

        def parse(**kwargs):
            self.kwargs = kwargs
            return response

        def beta_parse(**kwargs):
            if not beta_ok:
                raise TypeError("unexpected keyword argument 'fallbacks'")
            return parse(**kwargs)

        self.messages = SimpleNamespace(parse=parse)
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=beta_parse))


FIT = FitAssessment(score=80, verdict="strong", reasons=[], risks=[], eligibility_blockers=[])


def test_claude_wrapper_sends_cached_system_and_fallbacks():
    client = _Client(_response(parsed_output=FIT))
    llm = ClaudeLLM(client=client)
    assert llm.structured(system="SYS", prompt="hi", schema=FitAssessment) is FIT
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["fallbacks"] == "default" and kw["output_format"] is FitAssessment
    assert llm.usage.summary()["estimated_cost_usd"] == pytest.approx(0.0081)


def test_claude_wrapper_degrades_without_fallback_support():
    client = _Client(_response(parsed_output=FIT), beta_ok=False)
    llm = ClaudeLLM(client=client)
    llm.structured(system="s", prompt="p", schema=FitAssessment)
    assert "fallbacks" not in client.kwargs and llm.use_fallbacks is False


@pytest.mark.parametrize(
    "resp,exc",
    [
        (
            _response(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber")),
            LLMRefusal,
        ),
        (_response(stop_reason="max_tokens"), LLMError),
        (_response(), LLMError),
    ],
)
def test_claude_wrapper_raises_typed_errors(resp, exc):
    with pytest.raises(exc):
        ClaudeLLM(client=_Client(resp)).structured(system="s", prompt="p", schema=FitAssessment)


# --- store & notion ---------------------------------------------------------------


def test_store_snapshots_and_seen(tmp_path):
    store = Store(tmp_path / "db" / "s.db")
    assert store.latest_snapshot("u") is None
    store.add_snapshot("u", "v1", "2026-11-01T00:00:00")
    store.add_snapshot("u", "v2", "2026-11-08T00:00:00")
    assert store.latest_snapshot("u") == (content_hash("v2"), "v2")
    assert content_hash("a  b\n") == content_hash("a b")
    store.mark_seen("x", "now")
    store.mark_seen("x", "later")
    assert store.is_seen("x") and not store.is_seen("y")


def test_markdown_to_notion_blocks():
    md = "---\ntype: x\n---\n\n# Title\n\n## Section\n- **bold** item\n> quote\nplain\n|---|\n"
    blocks = markdown_to_blocks(md)
    assert [b["type"] for b in blocks] == [
        "heading_1",
        "heading_2",
        "bulleted_list_item",
        "quote",
        "paragraph",
    ]
    assert blocks[2]["bulleted_list_item"]["rich_text"][0]["text"]["content"] == "bold item"


@respx.mock
def test_notion_publish_paginates_children():
    create = respx.post("https://api.notion.com/v1/pages").mock(
        return_value=httpx.Response(200, json={"id": "page1", "url": "https://notion.so/page1"})
    )
    append = respx.patch("https://api.notion.com/v1/blocks/page1/children").mock(
        return_value=httpx.Response(200, json={})
    )
    md = "\n".join(f"- item {i}" for i in range(230))
    url = NotionPublisher("tok", "db").publish("Report", md)
    assert url == "https://notion.so/page1"
    assert create.call_count == 1 and append.call_count == 2
