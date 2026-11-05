"""Tests for BaseScraper modularity (text/social/API vs HTML/RSS) and deduplication."""

import pandas as pd

from nlp_predictor.scraping import BaseScraper, run_real_scrapers


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
