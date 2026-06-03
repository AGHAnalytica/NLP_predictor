from typing import Optional
import dateparser
from datetime import timezone, datetime
import re
import hashlib
import unicodedata
from bs4 import BeautifulSoup, FeatureNotFound


def normalize_date_to_utc(raw_date_str: str) -> Optional[datetime]:
    """Normalize a raw date string into a timezone-aware UTC datetime object.
    Returns:
        A timezone-aware datetime object set to UTC, or None if the parsing
        fails or the input is empty.
    """
    if not raw_date_str:
        return None

    parsed_date = dateparser.parse(
        raw_date_str,
        settings={"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True},
    )

    if parsed_date:
        return parsed_date.astimezone(timezone.utc)

    return None


def clean_html_content(raw_html: str) -> str:
    """
    Extracts clean text from html file deleting all the trash
    """
    if not raw_html or not isinstance(raw_html, str):
        return ""

    try:
        soup = BeautifulSoup(raw_html, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(raw_html, "html.parser")

    tags_to_destroy = [
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "form",
        "noscript",
        "svg",
        "canvas",
        "video",
        "audio",
        "iframe",
        "button",
    ]
    for tag in soup.find_all(tags_to_destroy):
        tag.decompose()

    trash_pattern = re.compile(
        r"\b(sidebar|ads?|advert(?:isement)?|sponsor(?:ed)?|share|comments?|social|menu|modal|newsletter|popup|cookie|promo)\b",
        re.I,
    )

    for element in soup.find_all(True):
        attrs = element.attrs

        class_id_str = " ".join(attrs.get("class", [])) + " " + str(attrs.get("id", ""))
        if trash_pattern.search(class_id_str):
            element.decompose()

    text = "\n".join(soup.stripped_strings)

    text = unicodedata.normalize("NFKC", text)

    boilerplate_patterns = [
        r"(?:subscribe|join|sign up)\s+to\s+our\s+newsletter.*",
        r"read\s+more[:\s].*",
        r"click\s+here\s+(?:to|for).*",
        r"©\s*\d{4}.*",
        r"all\s+rights\s+reserved",
        r"cookie\s+policy",
        r"terms\s+of\s+(?:service|use)",
    ]

    combined_pattern = re.compile(
        "|".join(boilerplate_patterns), re.IGNORECASE | re.MULTILINE
    )
    text = combined_pattern.sub("", text)

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

    return hashlib.sha256(standardized_text.encode("utf-8")).hexdigest()
