"""Text cleaning, timestamp normalization, and document hashing utilities."""

import hashlib
import logging
import re
import unicodedata
from datetime import UTC, datetime

import dateparser
import trafilatura

from nlp_predictor.config import config

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"\W+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
MIN_ARTICLE_LENGTH = int(config.hyperparams.scraping.min_article_length)


def normalize_date_to_utc(raw_date_str: str) -> datetime | None:
    """Parse an arbitrary date string into a timezone-aware UTC datetime.

    Args:
        raw_date_str: Raw publication timestamp string from article metadata.

    Returns:
        UTC-aware ``datetime`` instance, or ``None`` if empty or unparseable.
    """
    if not raw_date_str:
        return None

    parsed_date = dateparser.parse(
        raw_date_str, settings={"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True}
    )
    if parsed_date:
        return parsed_date.astimezone(UTC)
    return None


def normalize_plain_text(raw_text: str) -> str:
    """Normalize Unicode (NFKC) and collapse runs of whitespace in plain text.

    Args:
        raw_text: Unstructured plain text.

    Returns:
        Normalized single-spaced string, or ``""`` if empty.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", raw_text)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def clean_html_content(raw_html: str) -> str:
    """Extract main article body from raw HTML via Trafilatura and normalize whitespace.

    Args:
        raw_html: Raw HTML markup of the article page.

    Returns:
        Cleaned plain-text body, or ``""`` if extraction fails.
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""
    try:
        text = trafilatura.extract(
            raw_html, include_comments=False, include_tables=False
        )
    except Exception as e:
        logger.warning(f"Trafilatura error during extraction: {e}")
        return ""
    if not text:
        return ""
    return normalize_plain_text(text)


def strip_html_fragments(raw_html: str) -> str:
    """Fast regex-based HTML tag stripper for pre-extracted HTML snippets/fragments.

    Args:
        raw_html: HTML fragment or plain text string.

    Returns:
        Normalized plain text without HTML tags.
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""
    stripped = _HTML_TAG_RE.sub(" ", raw_html)
    return normalize_plain_text(stripped)


def is_valid_article(clean_text: str, min_length: int = MIN_ARTICLE_LENGTH) -> bool:
    """Return whether cleaned text meets the minimum character length threshold.

    Args:
        clean_text: Normalized article text.
        min_length: Minimum required character count (defaults to 200).

    Returns:
        ``True`` if ``len(clean_text) >= min_length``, ``False`` otherwise.
    """
    if not clean_text:
        return False
    return len(clean_text) >= min_length


def generate_doc_hash(text: str) -> str:
    """Compute a deterministic SHA-256 hex digest over alphanumeric-normalized text.

    Args:
        text: Cleaned article or post text.

    Returns:
        64-character hex SHA-256 digest, or ``""`` if input contains no alphanumeric
        chars.
    """
    if not text:
        return ""

    standardized_text = _NON_ALNUM_RE.sub("", text.lower())
    if not standardized_text:
        return ""

    return hashlib.sha256(standardized_text.encode("utf-8")).hexdigest()
