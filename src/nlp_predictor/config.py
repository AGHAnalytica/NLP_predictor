"""
Main configuration module for the NLP Predictor pipeline.

This module defines the global ``Config`` dictionary, which serves as the central
control point for the entire application. It contains input/output file paths,
language model (LLM) configurations, and the core mathematical and logical
hyperparameters driving the analysis process.

The configuration is divided into five main operational sections:

1. **LLM / EMBEDDING MODELS**:
    Defines the models used for reasoning (chat model) and text vectorization
    (embedding model) required by the Lower Path.

2. **FILE PATHS**:
    Dynamically generated absolute paths for test data files and output environments
    (Parquet format is strictly enforced for downstream databases).

3. **CLUSTERING PARAMETERS (Lower Path)**:
    Hyperparameters for the HDBSCAN algorithm, used to capture broad, macroeconomic
    narratives while filtering out isolated noise.

4. **SEVERITY MODEL / DCIM (Upper Path)**:
    Configuration and hyperparameters for the Directional Confirmed Impact Model (DCIM).
    It modulates the strength of news impact on specific financial instruments,
    accounting for: time half-life decay, "echo chamber" penalties, consensus bonuses,
    velocity (breaking news spikes), and overall scaling.

5. **SIGNAL GENERATION**:
    Rules for the final fusion layer. Defines the minimum threshold to trigger
    an active signal (neutral threshold) and how the macro environment (Lower Path)
    modulates individual instrument signals (alignment bonuses, conflict penalties,
    and hard vetos).

Attributes:
    BASE_DIR (Path): The root directory of the project (dynamically resolved
        as two levels up from this file's location).
    Config (dict): A dictionary containing all the aforementioned configuration
        variables and hyperparameters.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

Config = {
    # =========================================================================
    # LLM / EMBEDDING MODELS
    # =========================================================================
    # Central place to swap models for cost/quality experiments. The chat model is a
    # reasoning model; the embedding model is used only by the lower path.
    "LLM_MODEL": "gpt-5-nano",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    # =========================================================================
    # FILE PATHS
    # =========================================================================
    "BASE_DIR": BASE_DIR,
    # Input data paths
    "INSTRUMENT_PATH": str(
        BASE_DIR / "data" / "instruments" / "instruments_sample.csv"
    ),  # noqa: E501
    "TOPICS_PATH": str(BASE_DIR / "data" / "topics" / "topics_sample_large.csv"),  # noqa: E501
    "ARTICLES_PATH": str(BASE_DIR / "data" / "scraping" / "articles_sample.csv"),  # noqa: E501
    "WEIGHTS_PATH": str(BASE_DIR / "data" / "scraping" / "source_weights_sample.csv"),  # noqa: E501
    # Output save paths (Parquet format is strictly enforced for databases)
    "TOPICS_SAVE_PATH_CSV": str(
        BASE_DIR / "data" / "test_results" / "topics_report.csv"
    ),  # noqa: E501
    "TOPICS_SAVE_PATH_PARQUET": str(
        BASE_DIR / "data" / "test_results" / "topics_parquet"
    ),  # noqa: E501
    "GLOBAL_TRENDS_SAVE_PATH_CSV": str(
        BASE_DIR / "data" / "test_results" / "global_trends_report.csv"
    ),  # noqa: E501
    "GLOBAL_TRENDS_SAVE_PATH_PARQUET": str(
        BASE_DIR / "data" / "test_results" / "global_trends_parquet"
    ),  # noqa: E501
    "INSTRUMENT_TRENDS_SAVE_PATH_CSV": str(
        BASE_DIR / "data" / "test_results" / "instrument_trends_report.csv"
    ),  # noqa: E501
    "INSTRUMENT_TRENDS_SAVE_PATH_PARQUET": str(
        BASE_DIR / "data" / "test_results" / "instrument_trends_parquet"
    ),  # noqa: E501
    "SIGNALS_SAVE_PATH_CSV": str(
        BASE_DIR / "data" / "test_results" / "signals_report.csv"
    ),  # noqa: E501
    "SIGNALS_SAVE_PATH_PARQUET": str(
        BASE_DIR / "data" / "test_results" / "signals_parquet"
    ),  # noqa: E501
    # Rolling feature store: per-topic embeddings + instrument impacts, cached once
    # per article so the expensive LLM/embedding work is reused across 15-min ticks.
    "FEATURE_STORE_PATH": str(
        BASE_DIR / "data" / "feature_store" / "topic_features.parquet"
    ),  # noqa: E501
    # =========================================================================
    # CLUSTERING PARAMETERS (Lower Path - Macro Trends)
    # =========================================================================
    # Minimum number of valid topics required for a cluster to not be rejected as noise.
    "SEVERITY_THRESHOLD": 2,
    # Core HDBSCAN parameter: The minimum size of groupings to be considered a cluster.
    "MIN_CLUSTER_SIZE": 3,
    # =========================================================================
    # SEVERITY MODEL SELECTION (Upper Path)
    # =========================================================================
    # Which model the upper path uses to score each instrument's severity:
    #   "dcim"     - full Directional Confirmed Impact Model (5 phases below).
    #   "baseline" - minimal control: time-decayed, source-weighted sentiment sum.
    # Kept as a switch so the two can be A/B'd once an evaluation harness exists;
    # nothing downstream depends on which one is selected.
    "SEVERITY_MODEL": "dcim",
    # =========================================================================
    # DCIM HYPERPARAMETERS (Directional Confirmed Impact Model - Upper Path)
    # =========================================================================
    # [Time Decay] The time (in hours) it takes for an article to lose 50% of
    # its market impact.
    "DCIM_HALF_LIFE_HOURS": 4.0,
    # [Echo Chamber] How harshly to discount repeated articles from the SAME publisher.
    # Each extra article from a source is weighted 1 / (1 + BETA * rank), so a higher
    # value means the 2nd, 3rd, etc. article adds less. Kept light: legitimate repeated
    # coverage still counts, with only gentle diminishing returns (set 0.0 to disable).
    "DCIM_BETA_PENALTY": 0.2,
    # [Consensus Bonus] The multiplier applied when MULTIPLE DIFFERENT publishers agree
    # on the sentiment direction (bullish/bearish). Higher value rewards market-wide
    # agreement.
    "DCIM_GAMMA_CONSENSUS": 0.3,
    # [Velocity] The multiplier applied if a large portion of the articles are
    # "breaking news". Higher value increases the severity score during sudden news
    # spikes.
    "DCIM_DELTA_VELOCITY": 0.3,
    # [Scale] The base scaling factor applied before flattening the final score
    # with a tanh function. Higher value pushes the severity score towards -1.0 or 1.0
    # faster.
    "DCIM_KAPPA_SCALE": 1.0,
    # [Velocity Window] The maximum age (in hours) an article can be to be considered
    # "breaking/recent" for the velocity calculation.
    "DCIM_RECENT_WINDOW_HOURS": 0.5,
    # =========================================================================
    # ROLLING FEATURE STORE / SIGNAL WINDOW
    # =========================================================================
    # Default lookback (hours) of cached articles fed into severity. Each A/B variant
    # carries its OWN window_hours (below), so this is just the fallback for the legacy
    # single-model path; the experiment varies window per variant. Set near the target
    # ~12-16h holding horizon.
    "SIGNAL_WINDOW_HOURS": 16.0,
    # The longest window we ever retain in the store. Must be >= the largest variant
    # window_hours we A/B (currently 24h), with headroom, so no variant is starved of
    # data. Anything older is pruned on write (it can never re-enter any window again).
    "FEATURE_STORE_MAX_WINDOW_HOURS": 36.0,
    # =========================================================================
    # SEVERITY VARIANTS (A/B/n across demo accounts)
    # =========================================================================
    # Each entry is one self-contained strategy: a registered severity model + its
    # hyperparameters + its lookback window. Every variant is scored on the SAME cached
    # window each tick and routed to its own demo account, so realized PnL attributes
    # cleanly to the (model, params, window) triple.
    #
    # Params are tuned for ~12-16h holding positions, which drives three choices:
    #   - window_hours ~ the holding horizon, so the signal aggregates the news the
    #     position will actually live through (a 4h window would churn within the hold);
    #   - half_life_hours ~ the hold too, so an article stays relevant across it rather
    #     than decaying away after a couple of hours (intraday tuning);
    #   - kappa lowered as the window widens — more articles -> larger impact sum ->
    #     tanh saturates sooner, so longer windows need a gentler scale to keep range.
    # Velocity is de-emphasised (lower delta, wider recent window) since a 12-16h hold
    # does not care about sub-hour breaking spikes the way an intraday scalp would.
    # These are principled STARTING points to be tuned offline against the eval, not
    # final values.
    "SEVERITY_VARIANTS": [
        {
            "name": "baseline_16h",
            "model": "baseline",
            "window_hours": 16.0,
            "params": {"half_life_hours": 12.0, "kappa_scale": 0.8},
        },
        {
            "name": "dcim_12h_fast",
            "model": "dcim",
            "window_hours": 12.0,
            "params": {
                "half_life_hours": 8.0,
                "beta_penalty": 0.2,
                "gamma_consensus": 0.3,
                "delta_velocity": 0.3,
                "kappa_scale": 0.8,
                "recent_window_hours": 1.0,
            },
        },
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
        },
        {
            "name": "dcim_24h_slow",
            "model": "dcim",
            "window_hours": 24.0,
            "params": {
                "half_life_hours": 16.0,
                "beta_penalty": 0.2,
                "gamma_consensus": 0.3,
                "delta_velocity": 0.2,
                "kappa_scale": 0.6,
                "recent_window_hours": 2.0,
            },
        },
        {
            "name": "dcim_16h_consensus",
            "model": "dcim",
            "window_hours": 16.0,
            "params": {
                "half_life_hours": 12.0,
                "beta_penalty": 0.2,
                "gamma_consensus": 0.6,
                "delta_velocity": 0.2,
                "kappa_scale": 0.7,
                "recent_window_hours": 1.5,
            },
        },
        {
            "name": "abnvol_16h",
            "model": "abnormal_volume",
            "window_hours": 16.0,
            "params": {"half_life_hours": 12.0, "kappa_scale": 1.0},
        },
    ],
    # =========================================================================
    # SIGNAL GENERATION (Final fusion layer)
    # =========================================================================
    # |severity| below this -> NEUTRAL. The system refuses to trade weak/noisy signals.
    "SIGNAL_NEUTRAL_THRESHOLD": 0.15,
    # Base confidence is just |DCIM severity| (which already encodes article volume,
    # source weight, consensus and recency), then modulated by the macro factor below.
    # Macro modulation: confidence *= 1 + ALIGN_BONUS*agree - CONFLICT_PENALTY*conflict,
    # where coverage = fraction of the instrument's topics inside the macro narrative.
    "SIGNAL_MACRO_ALIGN_BONUS": 0.25,
    "SIGNAL_MACRO_CONFLICT_PENALTY": 0.5,
    # Macro veto: a strong opposing macro narrative can flip a WEAK instrument call to
    # NEUTRAL (only when coverage >= VETO_COVERAGE and |severity| < VETO_STRENGTH_CAP).
    "SIGNAL_MACRO_VETO": True,
    "SIGNAL_VETO_COVERAGE": 0.5,
    "SIGNAL_VETO_STRENGTH_CAP": 0.4,
}
