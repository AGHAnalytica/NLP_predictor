"""Offline tests for Bankier.pl, MarketWatch, PAP Biznes and RSS feed-only mode."""

from collections.abc import Sequence
from unittest.mock import patch

import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.scraping import (
    BankierScraper,
    BaseScraper,
    MarketWatchScraper,
    PapBiznesScraper,
)

BANKIER_ARTICLE_URL = "https://www.bankier.pl/wiadomosc/KNF-odpowiada-9230843.html"
MARKETWATCH_ARTICLE_URL = "https://www.marketwatch.com/story/oracle-stock-630d30ec"
PAP_ARTICLE_URL = "https://biznes.pap.pl/wiadomosci/firmy/kghm-ebitda-w-iii-kw"

ARTICLE_HTML = (
    "<html><body><article><h1>KNF odpowiada</h1><p>"
    + "Komisja Nadzoru Finansowego odniosła się do spadku kursu akcji XTB "
    "na Giełdzie Papierów Wartościowych w Warszawie. "
    * 4
    + "</p></article></body></html>"
)


def _rss(items: Sequence[dict[str, str]]) -> str:
    """Build an RSS 2.0 document from ``title``/``link``/``date``/``description``."""
    item_xml = "".join(
        f"<item><title>{item['title']}</title><link>{item['link']}</link>"
        f"<pubDate>{item['date']}</pubDate>"
        f"<description>{item.get('description', '')}</description></item>"
        for item in items
    )
    return f'<?xml version="1.0"?><rss><channel>{item_xml}</channel></rss>'


def _serve_feeds(scraper: BaseScraper, feeds: dict[str, str]):
    """Patch ``fetch_html`` so each feed URL returns its canned RSS document."""
    scraper.rss_feeds = list(feeds)
    return patch.object(scraper, "fetch_html", side_effect=feeds.get)


def test_strip_query_string_removes_tracking_params():
    tracked = f"{BANKIER_ARTICLE_URL}?utm_source=RSS&utm_campaign=Gielda"
    assert BaseScraper.strip_query_string(tracked) == BANKIER_ARTICLE_URL
    assert BaseScraper.strip_query_string(BANKIER_ARTICLE_URL) == BANKIER_ARTICLE_URL


def test_scrapers_read_feeds_and_thresholds_from_config(tmp_path):
    scraping_cfg = config.hyperparams.scraping
    history = str(tmp_path / "history.parquet")

    bankier = BankierScraper(history_file=history)
    marketwatch = MarketWatchScraper(history_file=history)
    pap = PapBiznesScraper(history_file=history)

    assert bankier.rss_feeds == scraping_cfg.bankier_rss_feeds
    assert marketwatch.rss_feeds == scraping_cfg.marketwatch_rss_feeds
    assert pap.rss_feeds == scraping_cfg.pap_biznes_rss_feeds
    assert bankier.min_content_length == scraping_cfg.min_article_length
    assert marketwatch.min_content_length == scraping_cfg.feed_only_min_length
    assert pap.min_content_length == scraping_cfg.feed_only_min_length


def test_bankier_fetches_article_shared_by_two_feeds_once(tmp_path):
    item = {
        "title": "KNF odpowiada na zarzuty. &amp;quot;Mamy świadomość&amp;quot;",
        "date": "Wed, 18 Nov 2026 12:50:00 +0100",
    }
    feeds = {
        "https://bankier.test/wiadomosci.xml": _rss(
            [{**item, "link": f"{BANKIER_ARTICLE_URL}?utm_campaign=Wiadomosci"}]
        ),
        "https://bankier.test/gielda.xml": _rss(
            [{**item, "link": f"{BANKIER_ARTICLE_URL}?utm_campaign=Gielda"}]
        ),
    }
    fetched_urls: list[str] = []

    async def fake_fetch_pages(urls, max_concurrent=None):
        fetched_urls.extend(urls)
        return [(url, ARTICLE_HTML) for url in urls]

    scraper = BankierScraper(history_file=str(tmp_path / "history.parquet"))
    with (
        _serve_feeds(scraper, feeds),
        patch.object(scraper, "fetch_urls_concurrently", fake_fetch_pages),
    ):
        df = scraper.run()

    assert fetched_urls == [BANKIER_ARTICLE_URL]
    assert len(df) == 1
    assert list(df.columns) == BaseScraper.ARTICLE_COLUMNS
    assert df.iloc[0]["source"] == "Bankier.pl"
    assert df.iloc[0]["url"] == BANKIER_ARTICLE_URL
    assert df.iloc[0]["title"] == 'KNF odpowiada na zarzuty. "Mamy świadomość"'
    assert "Komisja Nadzoru Finansowego" in df.iloc[0]["content"]


def test_marketwatch_builds_content_from_feed_without_fetching_pages(tmp_path):
    feeds = {
        "https://mw.test/topstories": _rss(
            [
                {
                    "title": "Oracle stock is cheap, analyst says.",
                    "link": f"{MARKETWATCH_ARTICLE_URL}?mod=mw_rss_topstories",
                    "date": "Wed, 18 Nov 2026 15:57:00 GMT",
                    "description": "Bernstein sees upside even in a gloomy case.",
                },
                {
                    "title": "Too short",
                    "link": "https://www.marketwatch.com/story/short-1",
                    "date": "Wed, 18 Nov 2026 15:00:00 GMT",
                },
            ]
        )
    }
    scraper = MarketWatchScraper(history_file=str(tmp_path / "history.parquet"))
    with (
        _serve_feeds(scraper, feeds),
        patch.object(scraper, "fetch_urls_concurrently") as fetch_pages,
    ):
        df = scraper.run()

    fetch_pages.assert_not_called()
    assert len(df) == 1
    assert df.iloc[0]["source"] == "MarketWatch"
    assert df.iloc[0]["url"] == MARKETWATCH_ARTICLE_URL
    assert df.iloc[0]["content"] == (
        "Oracle stock is cheap, analyst says. "
        "Bernstein sees upside even in a gloomy case."
    )
    assert df.iloc[0]["date"] == pd.Timestamp("2026-11-18 15:57:00", tz="UTC")


def test_pap_biznes_does_not_duplicate_title_from_description(tmp_path):
    title = "Skorygowana EBITDA grupy KGHM wyniosła w III kw. 3,996 mld zł"
    description = (
        f"&lt;span property=&quot;schema:name&quot;&gt;{title}&lt;/span&gt;"
        "&lt;span&gt;admin&lt;/span&gt;"
    )
    feeds = {
        "https://pap.test/rss": _rss(
            [
                {
                    "title": title,
                    "link": PAP_ARTICLE_URL,
                    "date": "Wed, 18 Nov 2026 16:21:14 +0000",
                    "description": description,
                }
            ]
        )
    }
    scraper = PapBiznesScraper(history_file=str(tmp_path / "history.parquet"))
    with _serve_feeds(scraper, feeds):
        df = scraper.run()

    assert len(df) == 1
    assert df.iloc[0]["source"] == "PAP Biznes"
    assert df.iloc[0]["content"] == f"{title} admin"
    assert df.iloc[0]["content"].count(title) == 1


def test_feed_only_items_are_deduplicated_across_runs(tmp_path):
    history = str(tmp_path / "history.parquet")
    feeds = {
        "https://mw.test/topstories": _rss(
            [
                {
                    "title": "Fed holds rates steady",
                    "link": MARKETWATCH_ARTICLE_URL,
                    "date": "Wed, 18 Nov 2026 15:57:00 GMT",
                    "description": "Powell signals patience on further cuts.",
                }
            ]
        )
    }
    first = MarketWatchScraper(history_file=history)
    with _serve_feeds(first, feeds):
        assert len(first.run()) == 1

    second = MarketWatchScraper(history_file=history)
    with _serve_feeds(second, feeds):
        assert second.run().empty


def test_every_scraper_source_has_a_weight():
    assert {"AP News", "BBC", "Bankier.pl", "CNBC", "MarketWatch", "PAP Biznes"} <= set(
        config.sources
    )
