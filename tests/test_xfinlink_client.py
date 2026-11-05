"""Tests for XfinlinkClient SDK wrapper and caching."""

import pandas as pd
import xfinlink as xfl

from nlp_predictor.backtesting.data.xfinlink_client import XfinlinkClient


def test_xfinlink_status_and_usage(tmp_path, monkeypatch):
    """Test retrieving API status and account usage."""
    client = XfinlinkClient(api_key="test_key", output_dir=tmp_path)

    fake_status = {"status": "healthy", "version": "1.0"}
    fake_usage = {"plan": "max", "requests_today": 12}

    monkeypatch.setattr(xfl, "status", lambda: fake_status)
    monkeypatch.setattr(xfl, "usage", lambda: fake_usage)

    assert client.get_status() == fake_status
    assert client.get_usage() == fake_usage


def test_xfinlink_fetch_constituents_and_caching(tmp_path, monkeypatch):
    """Test fetching constituents and caching to Parquet."""
    client = XfinlinkClient(api_key="test_key", output_dir=tmp_path)

    fake_df = pd.DataFrame(
        [
            {"entity_id": 1001, "ticker": "AAPL", "name": "Apple Inc."},
            {"entity_id": 1002, "ticker": "SIVB", "name": "SVB Financial"},
        ]
    )

    monkeypatch.setattr(xfl, "index", lambda index_name, as_of: fake_df)

    df = client.fetch_index_constituents(
        index_name="sp500", as_of="2023-01-01", save=True
    )

    assert len(df) == 2
    assert "ticker" in df.columns

    saved_file = tmp_path / "constituents_sp500_2023-01-01.parquet"
    assert saved_file.exists()
    loaded_df = pd.read_parquet(saved_file)
    assert len(loaded_df) == 2


def test_xfinlink_fetch_index_events(tmp_path, monkeypatch):
    """Test fetching historical index additions and removals log."""
    client = XfinlinkClient(api_key="test_key", output_dir=tmp_path)

    fake_events = pd.DataFrame(
        [
            {
                "ticker": "TSLA",
                "event_type": "added",
                "effective_date": "2020-12-21",
            },
            {
                "ticker": "SIVB",
                "event_type": "removed",
                "effective_date": "2023-03-15",
            },
        ]
    )

    def mock_events(index_name, start, end, limit, offset):
        if offset == 0:
            return fake_events
        return pd.DataFrame()

    monkeypatch.setattr(xfl, "index_events", mock_events)

    events_df = client.fetch_index_events("sp500", save=True)
    assert len(events_df) == 2
    saved = tmp_path / "index_events_sp500.parquet"
    assert saved.exists()


def test_xfinlink_skip_existing(tmp_path):
    """Test that skip_existing avoids calling SDK when file exists."""
    client = XfinlinkClient(api_key="test_key", output_dir=tmp_path)

    # Pre-create price file
    target_parquet = tmp_path / "prices_AAPL_daily.parquet"
    pd.DataFrame([{"close": 150.0}]).to_parquet(target_parquet)

    # Call fetch_prices with skip_existing=True without monkeypatching xfl
    # If it tries to call xfl it would fail or hit real API
    df = client.fetch_prices("AAPL", skip_existing=True)
    assert len(df) == 1
    assert df["close"].iloc[0] == 150.0


def test_xfinlink_dump_universe_mocked(tmp_path, monkeypatch):
    """Test comprehensive universe dump orchestrator with all endpoints."""
    client = XfinlinkClient(api_key="test_key", output_dir=tmp_path)

    monkeypatch.setattr(
        xfl,
        "fundamentals",
        lambda **kw: pd.DataFrame([{"period": "2024-12-31", "revenue": 100}]),
    )
    monkeypatch.setattr(
        xfl,
        "metrics",
        lambda **kw: pd.DataFrame([{"pe_ratio": 25.4, "market_cap": 200}]),
    )
    monkeypatch.setattr(
        xfl,
        "prices",
        lambda **kw: pd.DataFrame([{"date": "2024-12-31", "close": 150.0}]),
    )
    monkeypatch.setattr(
        xfl,
        "insiders",
        lambda **kw: pd.DataFrame([{"insider": "Cook", "shares": 5000}]),
    )
    monkeypatch.setattr(
        xfl,
        "holdings",
        lambda **kw: pd.DataFrame([{"manager": "Vanguard", "value": 10000}]),
    )

    stats = client.dump_universe_data(
        tickers=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        fundamentals_periods=["annual", "quarterly"],
        metrics_periods=["annual", "quarterly", "ttm"],
        fetch_insiders_data=True,
        fetch_holdings_data=True,
        skip_existing=False,
        delay_seconds=0.0,
    )

    assert stats["fundamentals"] == 2  # annual + quarterly
    assert stats["metrics"] == 3  # annual + quarterly + ttm
    assert stats["prices"] == 1
    assert stats["insiders"] == 1
    assert stats["holdings"] == 1

    assert (tmp_path / "fundamentals_AAPL_annual_original.parquet").exists()
    assert (tmp_path / "fundamentals_AAPL_quarterly_original.parquet").exists()
    assert (tmp_path / "metrics_AAPL_annual.parquet").exists()
    assert (tmp_path / "metrics_AAPL_quarterly.parquet").exists()
    assert (tmp_path / "metrics_AAPL_ttm.parquet").exists()
    assert (tmp_path / "prices_AAPL_daily.parquet").exists()
    assert (tmp_path / "insiders_AAPL.parquet").exists()
    assert (tmp_path / "holdings_AAPL.parquet").exists()
