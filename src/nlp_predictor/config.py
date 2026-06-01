from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

Config = {
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
    # =========================================================================
    # CLUSTERING PARAMETERS (Lower Path - Macro Trends)
    # =========================================================================
    # Minimum number of valid topics required for a cluster to not be rejected as noise.
    "SEVERITY_THRESHOLD": 2,
    # Core HDBSCAN parameter: The minimum size of groupings to be considered a cluster.
    "MIN_CLUSTER_SIZE": 3,
    # =========================================================================
    # DCIM HYPERPARAMETERS (Directional Confirmed Impact Model - Upper Path)
    # =========================================================================
    # [Time Decay] The time (in hours) it takes for an article to lose 50% of
    # its market impact.
    "DCIM_HALF_LIFE_HOURS": 4.0,
    # [Echo Chamber] Determines how harshly to penalize multiple articles from the SAME
    # publisher. A higher value means the 2nd, 3rd, etc. article from the same source
    # adds very little impact.
    "DCIM_BETA_PENALTY": 0.5,
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
    "DCIM_KAPPA_SCALE": 2.0,
    # [Velocity Window] The maximum age (in hours) an article can be to be considered
    # "breaking/recent" for the velocity calculation.
    "DCIM_RECENT_WINDOW_HOURS": 0.5,
}
