"""Tests for BaseScraper modularity, deduplication and robots.txt compliance."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import requests

from nlp_predictor.scraping import BaseScraper, run_real_scrapers

ROBOTS_TXT = "User-agent: *\nDisallow: /private/\nCrawl-delay: 3\n"


class _DummySocialScraper(BaseScraper):
    """Example non-HTML scraper (e.g., Twitter/X or wire feed) using
    process_text_item.
    """

    def __init__(self, history_file: str, posts: list[dict[str, str]]) -> None:
        super().__init__(
            history_file=history_file, min_content_length=20, fetch_delay=0.0
        )
        self.posts = posts

    def run(self) -> pd.DataFrame:
        rows = []
        for post in self.posts:
            item = self.process_text_item(
                source=post["source"],
                url=post["url"],
                raw_title=post["title"],
                raw_date=post["date"],
                raw_text=post["text"],
            )
            if item:
                rows.append(item)
        self.save_history()
        if not rows:
            return pd.DataFrame(columns=self.ARTICLE_COLUMNS)
        return pd.DataFrame(rows)[self.ARTICLE_COLUMNS]


def test_text_scraper_pipeline_and_deduplication(tmp_path):
    history_path = str(tmp_path / "scraped_history.parquet")
    posts = [
        {
            "source": "Twitter",
            "url": "https://x.com/fed/status/1",
            "title": "Fed announcement",
            "date": "2026-06-04T12:00:00Z",
            "text": "Federal Reserve signals rate cut expectations in Q3.",
        },
        # Duplicate content under a different URL -> skipped by doc_hash
        {
            "source": "Twitter",
            "url": "https://x.com/repost/status/2",
            "title": "Repost",
            "date": "2026-06-04T12:05:00Z",
            "text": "Federal Reserve signals rate cut expectations in Q3!",
        },
        # Too short (< 20 chars) -> skipped
        {
            "source": "Twitter",
            "url": "https://x.com/short/status/3",
            "title": "Short",
            "date": "2026-06-04T12:10:00Z",
            "text": "Too short.",
        },
    ]

    scraper = _DummySocialScraper(history_file=history_path, posts=posts)
    df = scraper.run()

    assert len(df) == 1
    assert list(df.columns) == BaseScraper.ARTICLE_COLUMNS
    assert df.iloc[0]["source"] == "Twitter"
    assert df.iloc[0]["url"] == "https://x.com/fed/status/1"

    # Re-running with the same history file skips already-seen items
    scraper_second = _DummySocialScraper(history_file=history_path, posts=posts)
    assert scraper_second.run().empty


def test_run_real_scrapers_accepts_custom_scrapers(tmp_path):
    history_path = str(tmp_path / "scraped_history.parquet")
    posts = [
        {
            "source": "WireAPI",
            "url": "https://wire.example.com/1",
            "title": "OPEC output cut",
            "date": "2026-06-04T10:00:00Z",
            "text": "OPEC+ unexpectedly announces deep crude oil production cuts.",
        }
    ]
    custom = _DummySocialScraper(history_file=history_path, posts=posts)
    combined = run_real_scrapers(scrapers=[custom])

    assert len(combined) == 1
    assert combined.iloc[0]["source"] == "WireAPI"


def _scraper_with_robots(
    tmp_path, robots_status: int = 200, robots_text: str = ROBOTS_TXT
) -> _DummySocialScraper:
    """Build a scraper whose session serves a canned robots.txt and page body."""
    scraper = _DummySocialScraper(
        history_file=str(tmp_path / "history.parquet"), posts=[]
    )

    def fake_get(url: str, timeout: int | None = None) -> MagicMock:
        if url.endswith("/robots.txt"):
            return MagicMock(status_code=robots_status, text=robots_text)
        return MagicMock(status_code=200, text=f"<html>{url}</html>")

    scraper.session.get = MagicMock(side_effect=fake_get)
    return scraper


def _requested_urls(scraper: BaseScraper) -> list[str]:
    """Return every URL passed to the scraper's mocked ``session.get``."""
    return [call.args[0] for call in scraper.session.get.call_args_list]


@patch("nlp_predictor.scraping.base_scraper.time.sleep")
def test_fetch_html_skips_disallowed_urls(mock_sleep, tmp_path):
    scraper = _scraper_with_robots(tmp_path)

    assert scraper.fetch_html("https://example.com/private/page") is None
    assert scraper.fetch_html("https://example.com/public/page") == (
        "<html>https://example.com/public/page</html>"
    )
    assert "https://example.com/private/page" not in _requested_urls(scraper)


@patch("nlp_predictor.scraping.base_scraper.time.sleep")
def test_robots_txt_is_fetched_once_per_domain(mock_sleep, tmp_path):
    scraper = _scraper_with_robots(tmp_path)

    scraper.fetch_html("https://example.com/a")
    scraper.fetch_html("https://example.com/b")
    scraper.fetch_html("https://other.com/c")

    robots_requests = [
        url for url in _requested_urls(scraper) if url.endswith("/robots.txt")
    ]
    assert robots_requests == [
        "https://example.com/robots.txt",
        "https://other.com/robots.txt",
    ]


def test_robots_txt_client_error_allows_all(tmp_path):
    scraper = _scraper_with_robots(tmp_path, robots_status=403, robots_text="")

    parser = scraper._get_robots_parser("https://example.com/private/page")

    assert parser.can_fetch("any-agent", "https://example.com/private/page")


def test_robots_txt_server_error_disallows_all(tmp_path):
    scraper = _scraper_with_robots(tmp_path, robots_status=503, robots_text="")

    parser = scraper._get_robots_parser("https://example.com/page")

    assert not parser.can_fetch("any-agent", "https://example.com/page")


def test_robots_txt_network_error_disallows_all(tmp_path):
    scraper = _scraper_with_robots(tmp_path)
    scraper.session.get = MagicMock(side_effect=requests.exceptions.ConnectionError)

    parser = scraper._get_robots_parser("https://example.com/page")

    assert not parser.can_fetch("any-agent", "https://example.com/page")


@patch("nlp_predictor.scraping.base_scraper.time.sleep")
def test_fetch_html_waits_for_crawl_delay(mock_sleep, tmp_path):
    scraper = _scraper_with_robots(tmp_path)

    scraper.fetch_html("https://example.com/public/page")

    mock_sleep.assert_called_once_with(3.0)


class _FakeAiohttpResponse:
    """Minimal async context manager mimicking an ``aiohttp`` 200 response."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.status = 200

    async def __aenter__(self) -> "_FakeAiohttpResponse":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def text(self) -> str:
        return f"<html>{self.url}</html>"


class _FakeAiohttpSession:
    """Minimal ``aiohttp.ClientSession`` replacement recording requested URLs."""

    requested: list[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _FakeAiohttpSession.requested = []

    async def __aenter__(self) -> "_FakeAiohttpSession":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    def get(self, url: str, timeout: int | None = None) -> _FakeAiohttpResponse:
        _FakeAiohttpSession.requested.append(url)
        return _FakeAiohttpResponse(url)


@patch("nlp_predictor.scraping.base_scraper.aiohttp.ClientSession", _FakeAiohttpSession)
async def test_fetch_urls_concurrently_honors_robots_and_crawl_delay(tmp_path):
    scraper = _scraper_with_robots(tmp_path)
    urls = [
        "https://example.com/private/page",
        "https://example.com/public/1",
        "https://example.com/public/2",
    ]

    with patch(
        "nlp_predictor.scraping.base_scraper.asyncio.sleep", new=AsyncMock()
    ) as mock_async_sleep:
        results = await scraper.fetch_urls_concurrently(urls)

    assert [url for url, _ in results] == urls[1:]
    assert _FakeAiohttpSession.requested == urls[1:]
    assert mock_async_sleep.await_count == 2
    mock_async_sleep.assert_awaited_with(3.0)
