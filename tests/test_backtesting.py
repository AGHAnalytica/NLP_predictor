from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from nlp_predictor.backtesting import (
    PYBROKER_PANEL_COLUMNS,
    BacktestDataLoader,
    HistoricalReplayer,
    PyBrokerBacktester,
    build_universe_instruments_csv,
)
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
from nlp_predictor.config._backtesting import (
    BacktestingConfig,
    HistoricalReplayConfig,
    PyBrokerStrategyConfig,
    TimeframeConfig,
)
from nlp_predictor.lower_path.macro_overlay import MacroOverlay
from nlp_predictor.preprocessing.feature_ingestor import FeatureIngestor
from nlp_predictor.preprocessing.feature_store import FeatureStore
from nlp_predictor.signal.signal_generator import SignalGenerator
from nlp_predictor.signal.signal_log import SignalLog
from nlp_predictor.tick.tick_runner import TickRunner
from nlp_predictor.upper_path.upper_path_analyzer import UpperPathAnalyzer


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


def _create_synthetic_backtest_fixtures(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Create synthetic Xfinlink Parquet prices, PIT index history, and signals."""
    xfl_dir = tmp_path / "xfinlink"
    xfl_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2024-01-02", periods=10, freq="B")
    aapl_prices = pd.DataFrame(
        {
            "ticker": ["AAPL"] * len(dates),
            "date": dates.strftime("%Y-%m-%d"),
            "open": [100.0 + i * 2.0 for i in range(len(dates))],
            "high": [102.0 + i * 2.0 for i in range(len(dates))],
            "low": [99.0 + i * 2.0 for i in range(len(dates))],
            "close": [101.0 + i * 2.0 for i in range(len(dates))],
            "adj_close": [101.0 + i * 2.0 for i in range(len(dates))],
            "volume": [1_000_000] * len(dates),
        }
    )
    aapl_prices.to_parquet(xfl_dir / "prices_AAPL_daily.parquet", index=False)

    msft_prices = pd.DataFrame(
        {
            "ticker": ["MSFT"] * len(dates),
            "date": dates.strftime("%Y-%m-%d"),
            "open": [200.0 - i * 2.0 for i in range(len(dates))],
            "high": [202.0 - i * 2.0 for i in range(len(dates))],
            "low": [198.0 - i * 2.0 for i in range(len(dates))],
            "close": [199.0 - i * 2.0 for i in range(len(dates))],
            "adj_close": [199.0 - i * 2.0 for i in range(len(dates))],
            "volume": [800_000] * len(dates),
        }
    )
    msft_prices.to_parquet(xfl_dir / "prices_MSFT_daily.parquet", index=False)

    # AAPL is in index the whole period; MSFT was removed before 2024
    constituents = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "added_date": "2010-01-01",
                "removed_date": None,
            },
            {
                "ticker": "MSFT",
                "added_date": "2010-01-01",
                "removed_date": "2023-12-31",
            },
        ]
    )
    constituents.to_parquet(
        xfl_dir / "constituents_sp500_latest.parquet",
        index=False,
    )

    signals_df = pd.DataFrame(
        [
            {
                "tick_id": "t1",
                "tick_time": f"{dates[0].strftime('%Y-%m-%d')} 14:30:00",
                "variant": "dcim_16h",
                "instrument": "AAPL",
                "signal": "UP",
                "confidence": 0.85,
                "instrument_severity": 0.75,
                "macro_context": "AGREE",
                "n_topics": 4,
                "topic_ids": ["id1"],
                "rationale": "Strong bullish consensus",
            },
            {
                "tick_id": "t1",
                "tick_time": f"{dates[0].strftime('%Y-%m-%d')} 14:30:00",
                "variant": "dcim_16h",
                "instrument": "MSFT",
                "signal": "UP",
                "confidence": 0.90,
                "instrument_severity": 0.80,
                "macro_context": "AGREE",
                "n_topics": 3,
                "topic_ids": ["id2"],
                "rationale": "Removed from index before 2024",
            },
            {
                "tick_id": "t2",
                "tick_time": f"{dates[3].strftime('%Y-%m-%d')} 16:00:00",
                "variant": "baseline_16h",
                "instrument": "AAPL",
                "signal": "UP",
                "confidence": 0.60,
                "instrument_severity": 0.55,
                "macro_context": "NONE",
                "n_topics": 2,
                "topic_ids": ["id3"],
                "rationale": "Baseline signal",
            },
        ]
    )
    return xfl_dir, signals_df


def test_backtest_data_loader_pit_and_panel_assembly(tmp_path):
    xfl_dir, signals_df = _create_synthetic_backtest_fixtures(tmp_path)
    log_path = tmp_path / "signals_log.parquet"
    sig_log = SignalLog(path=str(log_path))
    sig_log.append(signals_df)

    loader = BacktestDataLoader(
        xfinlink_dir=xfl_dir,
        market_prices_dir=tmp_path / "market_prices",
        signal_log=sig_log,
        universe_index="sp500",
    )

    panel = loader.build_pybroker_dataframe(
        variant="dcim_16h",
        start_date="2024-01-01",
        end_date="2024-01-31",
        enforce_point_in_time=True,
    )

    assert list(panel.columns) == PYBROKER_PANEL_COLUMNS
    assert set(panel["symbol"].unique()) == {"AAPL", "MSFT"}

    aapl_rows = panel[panel["symbol"] == "AAPL"]
    msft_rows = panel[panel["symbol"] == "MSFT"]
    assert (aapl_rows["in_index"] == 1).all()
    assert (msft_rows["in_index"] == 0).all()

    first_aapl = aapl_rows.iloc[0]
    assert first_aapl["signal_dir"] == 1
    assert first_aapl["confidence"] == 0.85


def test_pybroker_backtester_executes_and_respects_pit(tmp_path):
    xfl_dir, signals_df = _create_synthetic_backtest_fixtures(tmp_path)
    loader = BacktestDataLoader(
        xfinlink_dir=xfl_dir,
        market_prices_dir=tmp_path / "market_prices",
        universe_index="sp500",
    )

    strat_cfg = PyBrokerStrategyConfig(
        initial_cash=100_000.0,
        fee_mode="order_percent",
        fee_amount=0.0005,
        max_long_positions=5,
        holding_period_bars=2,
        min_confidence=0.20,
        position_sizing="confidence",
        max_position_weight=0.25,
        enforce_point_in_time_universe=True,
        exit_on_last_bar=True,
    )
    backtester = PyBrokerBacktester(
        data_loader=loader,
        strategy_config=strat_cfg,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )

    output = backtester.run_variant("dcim_16h", signals_df=signals_df)

    assert output.variant == "dcim_16h"
    assert not output.metrics_df.empty
    assert "total_return_pct" in output.metrics
    assert "sharpe" in output.metrics
    assert output.metrics["trade_count"] >= 1.0
    assert output.metrics["end_market_value"] > 100_000.0

    # MSFT was excluded by point-in-time index check; only AAPL should be traded
    traded_symbols = set(output.orders_df["symbol"].unique())
    assert traded_symbols == {"AAPL"}

    # Multi-variant summary run
    summary_df = backtester.run_all_variants(
        signals_df=signals_df,
        save_outputs=False,
    )
    assert len(summary_df) == 2
    assert set(summary_df["variant"]) == {"baseline_16h", "dcim_16h"}


def test_historical_replayer_no_pruning_and_chronological_replay(tmp_path):
    store_path = tmp_path / "hist_features.parquet"
    log_path = tmp_path / "hist_signals.parquet"
    weights_path = tmp_path / "weights.csv"
    pd.DataFrame([{"source_name": "benzinga", "source_weight": 0.9}]).to_csv(
        weights_path, index=False
    )

    replay_cfg = HistoricalReplayConfig(
        ingest_batch_size=2,
        tick_frequency="1D",
        decision_hour_utc=14,
        decision_minute_utc=0,
        retain_window_hours=1_000_000.0,
        skip_empty_windows=True,
    )
    store = FeatureStore(
        path=str(store_path),
        max_window_hours=replay_cfg.retain_window_hours,
    )

    # Fake collaborators avoiding live OpenAI API calls
    fake_extractor = MagicMock()
    fake_extractor.extract_topics.side_effect = lambda df: pd.DataFrame(
        [
            {
                "doc_hash": row["doc_hash"],
                "topic_id": f"tid_{row['doc_hash']}",
                "topic_name": row["title"],
                "topic_description": row["content"],
                "date": row["date"],
            }
            for row in df.to_dict(orient="records")
        ]
    )

    fake_lower = MagicMock()
    fake_lower.generate_embeddings_batch.side_effect = lambda texts: [
        [0.1, 0.2, 0.3] for _ in texts
    ]
    fake_lower.process_data.return_value = pd.DataFrame()

    fake_upper = UpperPathAnalyzer.__new__(UpperPathAnalyzer)
    fake_upper.map_topics = MagicMock(
        side_effect=lambda df: [
            [{"instrument": "AAPL", "sentiment": 1.0}] for _ in range(len(df))
        ]
    )

    fake_macro = MagicMock()
    fake_macro.build.return_value = MacroOverlay()

    signal_log = SignalLog(path=str(log_path))
    ingestor = FeatureIngestor(
        store=store,
        extractor=fake_extractor,
        upper=fake_upper,
        lower=fake_lower,
        weights_path=str(weights_path),
    )
    tick_runner = TickRunner(
        store=store,
        upper=fake_upper,
        lower=fake_lower,
        macro=fake_macro,
        signal=SignalGenerator.build_default(),
        log=signal_log,
        variants=[
            {
                "name": "dcim_16h",
                "model": "dcim",
                "window_hours": 16.0,
                "params": {
                    "half_life_hours": 12.0,
                    "beta_penalty": 0.2,
                    "gamma_consensus": 0.3,
                    "delta_velocity": 0.25,
                    "kappa_scale": 0.7,
                    "recent_window_hours": 1.5,
                },
            }
        ],
    )
    replayer = HistoricalReplayer(
        store=store,
        ingestor=ingestor,
        tick_runner=tick_runner,
        signal_log=signal_log,
        replay_config=replay_cfg,
    )

    # Create two articles 3 years apart to verify 36h pruning preserves old articles
    articles_df = pd.DataFrame(
        [
            {
                "doc_hash": "h_2021",
                "source": "benzinga",
                "date": "2021-03-10T10:00:00Z",
                "title": "Apple 2021 launch",
                "content": "Strong 2021 hardware release.",
            },
            {
                "doc_hash": "h_2024",
                "source": "benzinga",
                "date": "2024-03-10T10:00:00Z",
                "title": "Apple 2024 launch",
                "content": "Strong 2024 AI release.",
            },
        ]
    )

    cached_count = replayer.ingest_historical_articles(articles_df=articles_df)
    assert cached_count == 2
    assert len(store.load()) == 2

    # Replay ticks: only dates with in-window topics should be scheduled
    logged_signals = replayer.replay_ticks(
        start_date="2021-03-10",
        end_date="2021-03-12",
    )
    assert not logged_signals.empty
    assert (logged_signals["instrument"] == "AAPL").all()
    assert (logged_signals["tick_time"] == "2021-03-10 14:00:00").all()


def test_build_universe_instruments_csv(tmp_path):
    xfl_dir, _ = _create_synthetic_backtest_fixtures(tmp_path)
    df = pd.read_parquet(xfl_dir / "constituents_sp500_latest.parquet")
    df["entity_name"] = ["Apple Inc.", "Microsoft Corp."]
    df.to_parquet(xfl_dir / "constituents_sp500_latest.parquet", index=False)

    out_csv = tmp_path / "universe_instruments.csv"
    written_path = build_universe_instruments_csv(
        output_csv_path=out_csv,
        xfinlink_dir=xfl_dir,
        indices=["sp500"],
    )
    assert written_path.exists()
    res = pd.read_csv(written_path)
    assert set(res["ticker"]) == {"AAPL", "MSFT"}
