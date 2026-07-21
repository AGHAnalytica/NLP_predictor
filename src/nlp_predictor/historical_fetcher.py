import hashlib
import logging
import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


def _generate_doc_hash(identifier: str) -> str:
    """
    Generates a unique document hash (16 characters) from the provided string.
    """
    return hashlib.sha256(str(identifier).encode("utf-8")).hexdigest()[:16]


CURRENT_DIR = os.path.abspath('../..')
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.join(CURRENT_DIR, "data", "historical", "alpaca_news.csv")


def fetch_alpaca_historical_news(
    start_date: str,
    end_date: str,
    max_articles: int = 1000,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Fetches historical news articles from the Alpaca API using pagination tokens
    and saves them in a format compatible with the Group 1 pipeline.
    """
    api_key = os.getenv("ALPACA_API_KEY_ID")
    api_secret = os.getenv("ALPACA_API_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError(
            "Missing ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY in the .env file fool!"
        )

    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    params = {
        "start": start_date,
        "end": end_date,
        "sort": "asc",
        "limit": 50,  # 50 because that's the max limit for a single request
        "include_content": True,
        "exclude_contentless": True,
    }

    collected_articles = []
    page_token = None
    total_fetched = 0

    logger.info(f"Starting historical fetch from Alpaca: {start_date} to {end_date}...")

    while total_fetched < max_articles:
        if page_token:
            params["page_token"] = page_token

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 429:
                logger.warning("Rate limit exceeded. Waiting for 10 seconds...")
                time.sleep(10)
                continue

            response.raise_for_status()
            data = response.json()

            articles = data.get("news", [])
            if not articles:
                logger.info("No more articles found in the given date range.")
                break

            for item in articles:
                article_url = (
                    item.get("url") or f"https://alpaca.markets/news/{item['id']}"
                )

                record = {
                    "doc_hash": _generate_doc_hash(article_url),
                    "source": item.get("source", "alpaca"),
                    "url": article_url,
                    "date": item.get("created_at"),
                    "title": item.get("headline", ""),
                    "content": item.get("content", item.get("summary", "")),
                }
                collected_articles.append(record)
                total_fetched += 1

                if total_fetched > max_articles:
                    break

            logger.info(f"Fetched totally: {total_fetched}/{max_articles} articles...")

            page_token = data.get("next_page_token")
            if not page_token:
                logger.info("Pagination finished (no next_page_token :( ).")
                break

            time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error fetching data from Alpaca API: {e}")
            break

    df = pd.DataFrame(collected_articles)

    if not df.empty:
        expected_cols = ["doc_hash", "source", "url", "date", "title", "content"]
        df = df[expected_cols]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} historical articles to {output_path}")

    return df


if __name__ == "__main__":  # testing example values calm down...
    fetch_alpaca_historical_news(
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-12-31T00:00:00Z",
        max_articles=1_000_000, # quite a lot of articles to scrap, it takes some time, but it works.
    )
