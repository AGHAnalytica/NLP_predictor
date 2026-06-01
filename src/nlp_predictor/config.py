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
