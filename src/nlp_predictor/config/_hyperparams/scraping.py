"""Hyperparameters and endpoint URLs for live news scrapers."""

from pydantic import BaseModel


class ScrapingHyperparams(BaseModel):
    """Scraper thresholds, concurrency, HTTP headers, and feed/hub endpoints."""

    min_article_length: int = 200
    history_retention_hours: float = 4.0
    fetch_delay: float = 0.5
    max_concurrent: int = 15
    request_timeout: int = 15
    user_agent: str = "AGH_Analytica_scraper/1.0 (contact: agh_analytica@gmail.com)"
    apnews_base_url: str = "https://apnews.com"
    apnews_hub_urls: list[str] = [
        "https://apnews.com/hub/business",
        "https://apnews.com/hub/climate-and-environment",
    ]
    bbc_rss_feeds: list[str] = [
        "http://feeds.bbci.co.uk/news/business/rss.xml",
        "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    ]
    cnbc_rss_feeds: list[str] = [
        "https://www.cnbc.com/id/19836768/device/rss/rss.html",
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    ]

