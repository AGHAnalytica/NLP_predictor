import hashlib
import logging
import re
import unicodedata
from datetime import UTC, datetime

import dateparser
import trafilatura


def normalize_date_to_utc(raw_date_str: str) -> datetime | None:
    """
    Normalize a raw date string into a timezone-aware UTC datetime object.
    Articles may have publication dates in various formats and timezones. This function
    uses dateparser to flexibly parse the date string and then converts it to UTC.
    If the input is empty or parsing fails, it returns None.

    Args:
        raw_date_str (str): The raw date string extracted from the article metadata.

    Returns:
        A timezone-aware datetime object set to UTC, or None if the parsing
        fails or the input is empty.
    """
    if not raw_date_str:
        return None

    parsed_date = dateparser.parse(
        raw_date_str, settings={"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True}
    )

    if parsed_date:
        return parsed_date.astimezone(UTC)

    return None


def clean_html_content(raw_html: str) -> str:
    """
    Extracts the main article text from raw HTML using trafilatura,
    normalizes it, and cleans it. Returns the cleaned text or an empty
    string if extraction fails.

    Args:
        raw_html (str): The raw HTML content of the article.

    Returns:
        str: The cleaned article text, or an empty string if extraction fails.
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""
    try:
        text = trafilatura.extract(
            raw_html, include_comments=False, include_tables=False
        )
    except Exception as e:
        logging.warning(f"Trafilatura error during extraaction: {e}")
        return ""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    clean_text = re.sub(r"\s+", " ", text).strip()

    return clean_text


def is_valid_article(clean_text: str) -> bool:
    """
    Checks if the article has at least 200 characters of cleaned content.
    Returns True if valid, False otherwise. Articles that are too short are likely not
    useful for analysis.

    Args:
        clean_text (str): The cleaned article text to validate.

    Returns:
        bool: True if the article is valid (at least 200 characters), False otherwise.
    """
    if not clean_text:
        return False

    return len(clean_text) >= 200


def generate_doc_hash(text: str) -> str:
    """
    Generates a SHA-256 hash for the given text to be used for deduplication.
    Articles with identical content will produce the same hash, allowing us to identify
    duplicates even if the URL or title changes. The text is standardized
    by lowercasing and removing non-alphanumeric characters before hashing to improve
    the chances of catching duplicates that have minor formatting differences.

    Args:
        text: The cleaned article text.

    Returns:
        A hexadecimal string representation of the hash.
    """
    if not text:
        return ""

    standardized_text = re.sub(r"\W+", "", text.lower())
    if not standardized_text:
        return ""

    return hashlib.sha256(standardized_text.encode("utf-8")).hexdigest()
