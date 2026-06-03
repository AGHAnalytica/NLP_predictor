from typing import Optional
import dateparser
from datetime import timezone, datetime
import re
import hashlib
import unicodedata
import trafilatura
import logging


def normalize_date_to_utc(raw_date_str: str) -> Optional[datetime]:
    """Normalize a raw date string into a timezone-aware UTC datetime object.
    Returns:
        A timezone-aware datetime object set to UTC, or None if the parsing
        fails or the input is empty.
    """
    if not raw_date_str:
        return None

    parsed_date = dateparser.parse(raw_date_str, settings={'TIMEZONE': 'UTC', 'RETURN_AS_TIMEZONE_AWARE': True})

    if parsed_date:
        return parsed_date.astimezone(timezone.utc)

    return None

def clean_html_content(raw_html: str) -> str:
    """
    Extracts the main article text from raw HTML using trafilatura,
    normalizes it, and cleans it.
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""
    try:
        text = trafilatura.extract(raw_html, include_comments=False, include_tables=False)
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
    Checks if the article has at least 200 characters
    """
    if not clean_text:
        return False

    return len(clean_text) >= 200


def generate_doc_hash(text: str) -> str:
    """
    Generates a SHA-256 hash for the given text to be used for deduplication.
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

    return hashlib.sha256(standardized_text.encode('utf-8')).hexdigest()
