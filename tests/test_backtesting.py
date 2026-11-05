from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from nlp_predictor.backtesting.data.alpaca_news import (
    AlpacaNewsFetcher,
    _format_rfc3339,
)
from nlp_predictor.backtesting.data.clean_news import clean_historical_news
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
