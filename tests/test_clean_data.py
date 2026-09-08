"""
Tests for clean_data preprocessing utilities.
"""

from datetime import UTC, datetime

from nlp_predictor.preprocessing.clean_data import (
    clean_html_content,
    generate_doc_hash,
    is_valid_article,
    normalize_date_to_utc,
)


def test_generate_doc_hash():
    # 1. Empty or None input produces empty string
    assert generate_doc_hash("") == ""
    assert generate_doc_hash(None) == ""

    # 2. Non-empty produces 64-char SHA-256 hex string
    h1 = generate_doc_hash("Breaking news: Oil prices surge on global markets.")
    assert len(h1) == 64
    assert isinstance(h1, str)
    assert all(c in "0123456789abcdef" for c in h1)

    # 3. Deterministic
    h2 = generate_doc_hash("Breaking news: Oil prices surge on global markets.")
    assert h1 == h2

    # 4. Invariant to punctuation, whitespace, and capitalization
    h3 = generate_doc_hash("breaking   NEWS: oil prices surge, on global markets! ")
    assert h1 == h3


def test_clean_html_content():
    # 1. Empty or non-string input
    assert clean_html_content("") == ""
    assert clean_html_content(None) == ""

    # 2. Real HTML content extraction
    raw_html = (
        "<html><body>"
        "<h1>Market Update</h1>"
        "<p>Federal Reserve signals potential rate cut following inflation report.</p>"
        "<div><p>Markets rallied broadly following the announcement.</p></div>"
        "</body></html>"
    )
    cleaned = clean_html_content(raw_html)
    assert "Federal Reserve signals" in cleaned
    assert "<p>" not in cleaned
    assert "<html>" not in cleaned


def test_is_valid_article():
    # Short or empty
    assert not is_valid_article("")
    assert not is_valid_article(None)
    assert not is_valid_article("Too short article.")

    # 200+ characters
    long_text = "A" * 200
    assert is_valid_article(long_text)
    assert not is_valid_article("A" * 199)


def test_normalize_date_to_utc():
    # 1. Empty
    assert normalize_date_to_utc("") is None
    assert normalize_date_to_utc(None) is None
    assert normalize_date_to_utc("not-a-real-date-xyz") is None

    # 2. ISO UTC
    d1 = normalize_date_to_utc("2026-06-04T12:00:00Z")
    assert d1 == datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    # 3. Timezone conversion (+02:00 -> UTC)
    d2 = normalize_date_to_utc("2026-06-04 14:00:00+02:00")
    assert d2 == datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)
