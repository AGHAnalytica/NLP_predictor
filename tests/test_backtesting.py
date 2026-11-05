from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from nlp_predictor.backtesting.data.alpaca_news import (
    AlpacaNewsFetcher,
    _format_rfc3339,
)
from nlp_predictor.backtesting.data.clean_news import (
    clean_historical_news,
    extract_explicit_tickers,
    filter_historical_news,
    is_macro_event,
    is_spam_title,
)
from nlp_predictor.backtesting.data.price_fetcher import (
    YahooPriceFetcher,
    _shift_exclusive_end_date,
)
from nlp_predictor.config import config
from nlp_predictor.config._backtesting import BacktestingConfig, TimeframeConfig


def test_backtesting_config_loaded():
    assert config.backtesting is not None
    assert isinstance(config.backtesting, BacktestingConfig)
    assert isinstance(config.backtesting.timeframe, TimeframeConfig)
    assert config.backtesting.timeframe.start_date == "2020-01-01"
    assert config.backtesting.timeframe.end_date == "2025-12-31"
    assert isinstance(config.backtesting.data.market_prices_dir, Path)
    assert "SPY" in config.backtesting.data.macro_tickers
    assert isinstance(config.paths.historical.xfinlink, Path)
    assert isinstance(config.paths.historical.news_filtered, Path)
    assert config.backtesting.alpaca.page_limit == 50
    assert "sp500" in config.backtesting.xfinlink.indices
    assert config.backtesting.filtering.min_article_length == 200


def test_format_rfc3339():
    assert _format_rfc3339("2022-01-01", is_end=False) == "2022-01-01T00:00:00Z"
    assert _format_rfc3339("2022-12-31", is_end=True) == "2022-12-31T23:59:59Z"
    # Existing timestamp remains unchanged
    assert (
        _format_rfc3339("2022-05-01T12:30:00Z", is_end=False) == "2022-05-01T12:30:00Z"
    )


def test_shift_exclusive_end_date():
    assert _shift_exclusive_end_date("2025-12-31") == "2026-01-01"
    assert _shift_exclusive_end_date("2024-02-28") == "2024-02-29"
    assert _shift_exclusive_end_date(None) is None


@patch("nlp_predictor.backtesting.data.alpaca_news.requests.get")
@patch.dict(
    "os.environ",
    {"ALPACA_API_KEY_ID": "fake_key", "ALPACA_API_SECRET_KEY": "fake_secret"},
)
def test_alpaca_fetcher_uses_config_dates(mock_get, tmp_path):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"news": [], "next_page_token": None}
    mock_get.return_value = mock_resp

    fetcher = AlpacaNewsFetcher(output_dir=str(tmp_path))
    count = fetcher.fetch()

    assert count == 0
    assert mock_get.called
    call_args, call_kwargs = mock_get.call_args
    params = call_kwargs["params"]
    assert params["start"] == "2020-01-01T00:00:00Z"
    assert params["end"] == "2025-12-31T23:59:59Z"


def test_price_fetcher_initialization(tmp_path):
    fetcher = YahooPriceFetcher(output_dir=tmp_path)
    assert fetcher.output_dir == tmp_path
    assert tmp_path.exists()


@patch("yfinance.download")
def test_price_fetcher_uses_config_dates(mock_download, tmp_path):
    sample_df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-02")],
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [104.0],
            "Adj Close": [104.0],
            "Volume": [1000],
        }
    )
    mock_download.return_value = sample_df

    fetcher = YahooPriceFetcher(output_dir=tmp_path)
    res = fetcher.fetch_ticker("SPY")

    assert not res.empty
    assert mock_download.called
    call_args, call_kwargs = mock_download.call_args
    assert call_args[0] == "SPY"
    assert call_kwargs["start"] == config.backtesting.timeframe.start_date
    assert call_kwargs["end"] == "2026-01-01"

    # Cached file created
    cached_file = tmp_path / "SPY_daily.parquet"
    assert cached_file.exists()


@patch("yfinance.download")
def test_price_fetcher_fetch_tickers_default(mock_download, tmp_path):
    sample_df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-02")],
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [104.0],
            "Adj Close": [104.0],
            "Volume": [1000],
        }
    )
    mock_download.return_value = sample_df

    fetcher = YahooPriceFetcher(output_dir=tmp_path)
    res = fetcher.fetch_tickers(tickers=["SPY", "QQQ"], delay_seconds=0.0)

    assert "SPY" in res
    assert "QQQ" in res


def test_clean_historical_news(tmp_path):
    input_file = tmp_path / "raw_news.parquet"
    output_file = tmp_path / "clean_news.parquet"

    valid_html = (
        "<html><body><h1>Federal Reserve Announcement</h1>"
        + "<p>"
        + (
            "The Federal Reserve announced interest rate cuts today to stimulate growth"
            " across financial markets worldwide. " * 3
        )
        + "</p></body></html>"
    )

    raw_data = pd.DataFrame(
        [
            {
                "doc_hash": "hash1",
                "source": "alpaca",
                "url": "https://example.com/1",
                "date": "2024-01-01 10:00:00",
                "title": "Fed rate cut decision",
                "content": valid_html,
            },
            {
                "doc_hash": "hash2",
                "source": "alpaca",
                "url": "https://example.com/2",
                "date": "2024-01-01 11:00:00",
                "title": "Short",
                "content": "Too short.",
            },
        ]
    )
    raw_data.to_parquet(input_file, index=False)

    clean_historical_news(input_path=input_file, output_path=output_file)

    assert output_file.exists()
    cleaned_df = pd.read_parquet(output_file)
    assert len(cleaned_df) == 1
    assert "<p>" not in cleaned_df["content"].iloc[0]
    assert cleaned_df["title"].iloc[0] == "Fed rate cut decision"


def test_clean_historical_news_missing_file(tmp_path):
    # Missing input file should log error and return without exception
    missing_file = tmp_path / "does_not_exist.parquet"
    out_file = tmp_path / "clean.parquet"
    clean_historical_news(input_path=missing_file, output_path=out_file)
    assert not out_file.exists()


def test_spam_and_macro_and_ticker_filters():
    assert is_spam_title("10 Biggest Movers In Pre-Market")
    assert is_spam_title("Earnings Scheduled For Oct 25")
    assert not is_spam_title("Apple Reports Record Q4 iPhone Revenue")

    assert is_macro_event("Federal Reserve hikes interest rates", "Full statement...")
    assert not is_macro_event("Local Bakery Opens New Store", "Great pastries today.")

    html = '<p>Watch <a href="/stock/AAPL#">Apple</a> ($MSFT) and NASDAQ:NVDA</p>'
    assert extract_explicit_tickers(html) == {"AAPL", "MSFT", "NVDA"}


def test_filter_historical_news_pipeline(tmp_path):
    part_dir = tmp_path / "partitioned"
    part_dir.mkdir(parents=True)
    out_file = tmp_path / "filtered.parquet"
    inst_csv = tmp_path / "instruments.csv"
    pd.DataFrame([{"ticker": "AAPL", "full_name": "Apple Inc."}]).to_csv(
        inst_csv, index=False
    )

    long_aapl = (
        '<p><a href="/stock/AAPL#">Apple</a> announced major supply chain '
        "expansions and record earnings growth across all global regions today. "
        * 3
        + "</p>"
    )
    long_spam = "<p>" + ("10 biggest movers in the market today. " * 8) + "</p>"
    long_irrelevant = (
        "<p>"
        + (
            'Shares of <a href="/stock/ZZZZ#">Unknown Corp</a> traded sideways '
            "in quiet local trading sessions today. " * 3
        )
        + "</p>"
    )

    df = pd.DataFrame(
        [
            {
                "doc_hash": "h1",
                "source": "benzinga",
                "url": "https://example.com/1",
                "date": "2024-05-01T12:00:00Z",
                "title": "Apple Expands Supply Chain",
                "content": long_aapl,
            },
            {
                "doc_hash": "h2",
                "source": "benzinga",
                "url": "https://example.com/2",
                "date": "2024-05-01T13:00:00Z",
                "title": "12 Stocks Moving In Tuesday's Pre-Market Session",
                "content": long_spam,
            },
            {
                "doc_hash": "h3",
                "source": "benzinga",
                "url": "https://example.com/3",
                "date": "2024-05-01T14:00:00Z",
                "title": "Unknown Corp Quiet Day",
                "content": long_irrelevant,
            },
        ]
    )
    df.to_parquet(part_dir / "batch.parquet", index=False)

    filtered = filter_historical_news(
        input_dir=part_dir,
        output_path=out_file,
        instruments_path=inst_csv,
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["title"] == "Apple Expands Supply Chain"
    assert out_file.exists()
