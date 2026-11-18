"""PyBroker backtesting engine evaluating single and multi-variant NLP signals."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pybroker as pb
from pybroker import ExecContext, FeeMode, PositionMode, Strategy, StrategyConfig

from nlp_predictor.backtesting.data_loader import (
    CUSTOM_PYBROKER_COLUMNS,
    BacktestDataLoader,
)
from nlp_predictor.config import config
from nlp_predictor.config._backtesting import PyBrokerStrategyConfig

logger = logging.getLogger(__name__)

_FEE_MODE_MAP: dict[str, FeeMode | None] = {
    "order_percent": FeeMode.ORDER_PERCENT,
    "per_order": FeeMode.PER_ORDER,
    "per_share": FeeMode.PER_SHARE,
    "none": None,
}

_POSITION_MODE_MAP: dict[str, PositionMode] = {
    "default": PositionMode.DEFAULT,
    "long_only": PositionMode.LONG_ONLY,
    "short_only": PositionMode.SHORT_ONLY,
}


@dataclass(frozen=True)
class BacktestRunOutput:
    """Structured output container for a single-variant PyBroker backtest run.

    Holds ``variant``, ``metrics_df``, ``metrics``, ``portfolio_df``,
    ``trades_df``, and ``orders_df`` tables returned by PyBroker.
    """

    variant: str
    metrics_df: pd.DataFrame
    metrics: dict[str, float]
    portfolio_df: pd.DataFrame
    trades_df: pd.DataFrame
    orders_df: pd.DataFrame


class PyBrokerBacktester:
    """Executes event-driven portfolio simulations via :mod:`pybroker`.

    Consumes survivorship-free daily price + signal panels from
    :class:`BacktestDataLoader`, enforces point-in-time index membership, applies
    confidence-weighted or equal-weighted position sizing, and exports PyBroker's
    native metrics, equity curve, trades, and orders to Parquet.
    """

    def __init__(
        self,
        *,
        data_loader: BacktestDataLoader,
        strategy_config: PyBrokerStrategyConfig | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        """Initialize the PyBroker backtester with injected data loader and config.

        Args:
            data_loader: Collaborator loading aligned OHLCV and signal panels.
            strategy_config: Optional typed strategy configuration override.
            start_date: Default backtest start date (``YYYY-MM-DD``).
            end_date: Default backtest end date (``YYYY-MM-DD``).
        """
        self.data_loader = data_loader
        self.strategy_config = strategy_config or config.backtesting.strategy
        self.start_date = start_date or config.backtesting.timeframe.start_date
        self.end_date = end_date or config.backtesting.timeframe.end_date
        pb.register_columns(list(CUSTOM_PYBROKER_COLUMNS))

    @classmethod
    def build_default(cls) -> "PyBrokerBacktester":
        """Construct a production :class:`PyBrokerBacktester` with default wiring."""
        return cls(data_loader=BacktestDataLoader.build_default())

    def _build_pybroker_config(self) -> StrategyConfig:
        """Translate :attr:`strategy_config` into a :class:`pybroker.StrategyConfig`."""
        cfg = self.strategy_config
        fee_key = cfg.fee_mode.strip().lower()
        pos_key = cfg.position_mode.strip().lower()

        fee_mode = _FEE_MODE_MAP.get(fee_key, FeeMode.ORDER_PERCENT)
        if not cfg.allow_short and pos_key == "default":
            position_mode = PositionMode.LONG_ONLY
        else:
            position_mode = _POSITION_MODE_MAP.get(pos_key, PositionMode.DEFAULT)

        max_short = cfg.max_short_positions if cfg.allow_short else None
        return StrategyConfig(
            initial_cash=float(cfg.initial_cash),
            fee_mode=fee_mode,
            fee_amount=float(cfg.fee_amount),
            enable_fractional_shares=bool(cfg.enable_fractional_shares),
            position_mode=position_mode,
            max_short_positions=max_short,
            buy_delay=int(cfg.buy_delay),
            sell_delay=int(cfg.sell_delay),
            bootstrap_samples=int(cfg.bootstrap_samples),
            bars_per_year=int(cfg.bars_per_year),
            exit_on_last_bar=bool(cfg.exit_on_last_bar),
        )

    def _compute_target_weight(
        self, confidence: float, is_short: bool = False
    ) -> float:
        """Compute portfolio allocation weight for a signal with given confidence."""
        cfg = self.strategy_config
        max_positions = (
            max(1, int(cfg.max_short_positions))
            if is_short
            else max(1, int(cfg.max_long_positions))
        )
        base_slot = 1.0 / float(max_positions)
        max_weight = float(cfg.max_position_weight)

        scheme = cfg.position_sizing.strip().lower()
        if scheme == "equal":
            raw_weight = base_slot
        else:
            raw_weight = base_slot * max(0.0, min(1.0, float(confidence)))

        return min(max_weight, max(0.0, raw_weight))

    def _execute_bar(self, ctxs: Mapping[str, ExecContext]) -> None:
        """Multi-symbol execution callback invoked by PyBroker on every trading bar."""
        cfg = self.strategy_config
        min_conf = float(cfg.min_confidence)
        hold_bars = int(cfg.holding_period_bars)
        enforce_pit = bool(cfg.enforce_point_in_time_universe)
        allow_short = bool(cfg.allow_short)

        long_candidates: list[tuple[float, ExecContext]] = []
        short_candidates: list[tuple[float, ExecContext]] = []

        for ctx in ctxs.values():
            sig_dir = int(ctx.signal_dir[-1])
            conf = float(ctx.confidence[-1])
            in_idx = int(ctx.in_index[-1]) == 1

            has_long = bool(ctx.long_pos())
            has_short = bool(ctx.short_pos())

            if has_long and (sig_dir < 0 or (enforce_pit and not in_idx)):
                ctx.sell_all_shares()
                continue

            if has_short and (sig_dir > 0 or (enforce_pit and not in_idx)):
                ctx.cover_all_shares()
                continue

            if enforce_pit and not in_idx:
                continue

            if conf < min_conf:
                continue

            if sig_dir > 0 and not has_long:
                long_candidates.append((conf, ctx))
            elif sig_dir < 0 and allow_short and not has_short:
                short_candidates.append((conf, ctx))

        long_candidates.sort(key=lambda item: item[0], reverse=True)
        for conf, ctx in long_candidates:
            target_weight = self._compute_target_weight(conf, is_short=False)
            shares = ctx.calc_target_shares(target_weight)
            if shares > 0:
                ctx.buy_shares = shares
                if hold_bars > 0:
                    ctx.hold_bars = hold_bars

        if allow_short:
            short_candidates.sort(key=lambda item: item[0], reverse=True)
            for conf, ctx in short_candidates:
                target_weight = self._compute_target_weight(conf, is_short=True)
                shares = ctx.calc_target_shares(target_weight)
                if shares > 0:
                    ctx.sell_shares = shares
                    if hold_bars > 0:
                        ctx.hold_bars = hold_bars

    def run_panel(
        self,
        panel_df: pd.DataFrame,
        *,
        variant: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestRunOutput:
        """Execute a PyBroker backtest directly on a prepared OHLCV + signal panel.

        Args:
            panel_df: DataFrame produced by
                :meth:`BacktestDataLoader.build_pybroker_dataframe`.
            variant: Variant identifier tag for the output container.
            start_date: Optional override for backtest start date.
            end_date: Optional override for backtest end date.

        Returns:
            :class:`BacktestRunOutput` containing metrics, portfolio, trades, and
            orders.
        """
        variant_name = variant or self.strategy_config.default_variant
        if panel_df is None or panel_df.empty:
            logger.warning(f"Empty panel for variant '{variant_name}'; skipping run.")
            return BacktestRunOutput(
                variant=variant_name,
                metrics_df=pd.DataFrame(columns=["name", "value"]),
                metrics={},
                portfolio_df=pd.DataFrame(),
                trades_df=pd.DataFrame(),
                orders_df=pd.DataFrame(),
            )

        symbols = sorted(panel_df["symbol"].unique().tolist())
        min_panel_date = pd.to_datetime(panel_df["date"].min()).strftime("%Y-%m-%d")
        max_panel_date = pd.to_datetime(panel_df["date"].max()).strftime("%Y-%m-%d")
        start_dt = start_date or min_panel_date or self.start_date
        end_dt = end_date or max_panel_date or self.end_date

        pb_config = self._build_pybroker_config()
        strategy = Strategy(
            data_source=panel_df,
            start_date=start_dt,
            end_date=end_dt,
            config=pb_config,
        )
        if int(self.strategy_config.max_long_positions) > 0:
            strategy.set_max_long_positions(
                int(self.strategy_config.max_long_positions)
            )
        strategy.add_execution(
            None,
            symbols,
            config=pb_config,
        )
        strategy.set_after_exec(self._execute_bar)

        logger.info(
            f"Running PyBroker backtest for variant='{variant_name}' "
            f"across {len(symbols)} symbol(s) [{start_dt} -> {end_dt}]..."
        )
        result = strategy.backtest(
            start_date=start_dt,
            end_date=end_dt,
            calc_bootstrap=bool(self.strategy_config.calc_bootstrap),
        )

        metrics_df = result.metrics_df.copy()
        metrics_dict = {
            str(row["name"]): float(row["value"])
            for row in metrics_df.to_dict(orient="records")
            if pd.notna(row.get("value"))
        }

        portfolio_df = result.portfolio.reset_index()
        portfolio_df["variant"] = variant_name

        trades_df = result.trades.reset_index()
        trades_df["variant"] = variant_name

        orders_df = result.orders.reset_index()
        orders_df["variant"] = variant_name

        return BacktestRunOutput(
            variant=variant_name,
            metrics_df=metrics_df,
            metrics=metrics_dict,
            portfolio_df=portfolio_df,
            trades_df=trades_df,
            orders_df=orders_df,
        )

    def run_variant(
        self,
        variant: str | None = None,
        *,
        signals_df: pd.DataFrame | None = None,
        tickers: Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BacktestRunOutput:
        """Assemble panel data and run PyBroker for a single severity variant.

        Args:
            variant: Severity variant name (defaults to ``default_variant`` in config).
            signals_df: Optional pre-loaded signals DataFrame.
            tickers: Optional explicit list of tickers to include.
            start_date: Optional start date override (``YYYY-MM-DD``).
            end_date: Optional end date override (``YYYY-MM-DD``).

        Returns:
            :class:`BacktestRunOutput` with all PyBroker evaluation tables.
        """
        variant_name = variant or self.strategy_config.default_variant
        start_dt = start_date or self.start_date
        end_dt = end_date or self.end_date

        panel_df = self.data_loader.build_pybroker_dataframe(
            signals_df=signals_df,
            variant=variant_name,
            tickers=tickers,
            start_date=start_dt,
            end_date=end_dt,
            enforce_point_in_time=self.strategy_config.enforce_point_in_time_universe,
        )
        return self.run_panel(
            panel_df,
            variant=variant_name,
            start_date=start_dt,
            end_date=end_dt,
        )

    def run_all_variants(
        self,
        *,
        signals_df: pd.DataFrame | None = None,
        variants: Sequence[str] | None = None,
        tickers: Sequence[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        save_outputs: bool = True,
    ) -> pd.DataFrame:
        """Evaluate multiple A/B/n severity variants and persist comparison tables.

        Args:
            signals_df: Optional signals DataFrame (loaded from :class:`SignalLog` if
                ``None``).
            variants: Sequence of variant names to evaluate (defaults to variants
                present in ``signals_df`` or ``config.severity.variants``).
            tickers: Optional ticker universe override.
            start_date: Backtest start date (``YYYY-MM-DD``).
            end_date: Backtest end date (``YYYY-MM-DD``).
            save_outputs: Whether to save consolidated Parquet outputs to
                ``config.paths.output.backtest_*``.

        Returns:
            Comparison DataFrame with one row per variant and all PyBroker metrics.
        """
        raw_signals = (
            signals_df if signals_df is not None else self.data_loader.signal_log.load()
        )
        if variants is not None:
            target_variants = list(variants)
        elif not raw_signals.empty and "variant" in raw_signals.columns:
            target_variants = sorted(
                raw_signals["variant"].dropna().astype(str).unique().tolist()
            )
        else:
            target_variants = [v["name"] for v in config.severity.variants_list]

        metric_rows: list[dict[str, Any]] = []
        portfolios: list[pd.DataFrame] = []
        trades: list[pd.DataFrame] = []
        orders: list[pd.DataFrame] = []

        for var_name in target_variants:
            output = self.run_variant(
                var_name,
                signals_df=raw_signals,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
            )
            if not output.metrics:
                continue
            row: dict[str, Any] = {"variant": var_name, **output.metrics}
            metric_rows.append(row)
            if not output.portfolio_df.empty:
                portfolios.append(output.portfolio_df)
            if not output.trades_df.empty:
                trades.append(output.trades_df)
            if not output.orders_df.empty:
                orders.append(output.orders_df)

        summary_df = pd.DataFrame.from_records(metric_rows)
        if save_outputs and not summary_df.empty:
            self._persist_results(
                summary_df=summary_df,
                portfolios=portfolios,
                trades=trades,
                orders=orders,
            )
        return summary_df

    @staticmethod
    def _persist_results(
        *,
        summary_df: pd.DataFrame,
        portfolios: list[pd.DataFrame],
        trades: list[pd.DataFrame],
        orders: list[pd.DataFrame],
    ) -> None:
        """Persist consolidated A/B/n backtest tables to Parquet."""
        out_paths = config.paths.output
        out_dir = Path(out_paths.backtest_results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_parquet(
            out_paths.backtest_metrics_parquet, index=False, engine="pyarrow"
        )
        if portfolios:
            pd.concat(portfolios, ignore_index=True).to_parquet(
                out_paths.backtest_portfolio_parquet, index=False, engine="pyarrow"
            )
        if trades:
            pd.concat(trades, ignore_index=True).to_parquet(
                out_paths.backtest_trades_parquet, index=False, engine="pyarrow"
            )
        if orders:
            pd.concat(orders, ignore_index=True).to_parquet(
                out_paths.backtest_orders_parquet, index=False, engine="pyarrow"
            )
        logger.info(f"Saved backtest outputs to {out_dir}")
