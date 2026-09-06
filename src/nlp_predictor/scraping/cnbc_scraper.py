import logging
import xml.etree.ElementTree as ET

import pandas as pd

from nlp_predictor.scraping.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class CNBCScraper(BaseScraper):
    """
    Scraper for CNBC news using official RSS feeds.
    Extracts metadata from RSS and fetches full article content.
    Filters out CNBC Pro (paid) articles.
    """

    def __init__(self, history_file: str = "data/scraped_history.parquet"):
        super().__init__(history_file)

        self.rss_feeds = [
            "https://www.cnbc.com/id/19836768/device/rss/rss.html",
            "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        ]

    def run(self) -> pd.DataFrame:
        """
        Main loop.
        Returns a Pandas DataFrame with new articles cleaned up.
        If no new articles are found, returns an empty DataFrame with the correct
        columns. Returns an empty DataFrame with the correct columns if no new articles
        are found.
        Articles are filtered for duplicates and media content before processing.
        Fetches full HTML for each article to extract content and metadata, then passes
        through the cleaner pipeline defined in process_article().

        Returns:
            pd.DataFrame: A DataFrame containing new articles with columns:
            doc_hash, source, url, date, title, content. If no new articles are
            found, returns an empty DataFrame with the correct columns.
        """
        articles_data = []

        for rss_url in self.rss_feeds:
            logger.info(f"Fetching RSS feed: {rss_url}")
            rss_xml = self.fetch_html(rss_url)

            if not rss_xml:
                continue

            try:
                root = ET.fromstring(rss_xml)

                for item in root.findall(".//item"):
                    raw_title = item.findtext("title")
                    article_url = item.findtext("link")
                    raw_date = item.findtext("pubDate")

                    if not raw_title or not article_url or not raw_date:
                        continue

                    raw_title = raw_title.strip()
                    article_url = article_url.strip()
                    raw_date = raw_date.strip()

                    if "cnbc.com/pro/" in article_url:
                        logger.debug(f"Skipped paid CNBC Pro article: {article_url}")
                        continue

                    if self.is_duplicate(doc_hash="", url=article_url):
                        logger.debug(
                            f"Article already exists in DB (URL): {article_url}"
                        )
                        continue

                    logger.info(f"Fetching full content: {article_url}")
                    article_html = self.fetch_html(article_url)

                    if not article_html:
                        continue

                    processed = self.process_article(
                        source="CNBC",
                        url=article_url,
                        raw_title=raw_title,
                        raw_date=raw_date,
                        raw_html=article_html,
                    )

                    if processed:
                        articles_data.append(processed)

            except ET.ParseError as e:
                logger.error(f"XML parsing error for {rss_url}: {e}")

        self.save_history()
        logger.info(f"CNBCScraper finished. Fetched {len(articles_data)} new articles.")

        expected_columns = ["doc_hash", "source", "url", "date", "title", "content"]

        if not articles_data:
            return pd.DataFrame(columns=expected_columns)

        df = pd.DataFrame(articles_data)
        return df[expected_columns]
