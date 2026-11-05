"""
Quantify how much of the upper-path signal is driven by ARTICLE COUNT (the V1
"source of truth") versus the other DCIM inputs (sentiment, source weight, recency).

Three independent measurements, all against the *real* DCIM code:
  A) Empirical: corr(article count, |severity|) on the saved instrument report.
  B) Elasticity: perturb one input at a time around a baseline group, measure
     how far severity moves.
  C) Variance decomposition (Shapley/dominance over R^2): across a realistic
     random ensemble of instruments, attribute the variance of severity to each
     input. This is the honest "what % is article count" number.
"""

import ast
import itertools
import logging
import math

import numpy as np
import pandas as pd

from nlp_predictor.config import config
from nlp_predictor.upper_path import UpperPathAnalyzer

logger = logging.getLogger(__name__)

RNG = np.random.default_rng(42)
SENTIMENT_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
NOW = pd.Timestamp.now(tz="UTC")


def _analyzer() -> UpperPathAnalyzer:
    """Build the real analyzer without touching the API / instrument loading."""
    analyzer = UpperPathAnalyzer.__new__(UpperPathAnalyzer)
    analyzer.current_time = NOW
    return analyzer


def _make_group(
    n: int, sentiments: list[float], weights: list[float], ages_hours: list[float]
) -> pd.DataFrame:
    """Construct one instrument's article group in the shape the DCIM expects."""
    return pd.DataFrame(
        {
            "date": [NOW - pd.Timedelta(hours=h) for h in ages_hours],
            "source_name": [f"src{i}" for i in range(n)],
            "sentiment": sentiments,
            "source_weight": weights,
        }
    )


# ---------------------------------------------------------------- A) empirical
def measure_empirical() -> None:
    """Measure empirical correlation between article count and |severity|."""
    df = pd.read_csv(config.paths.output.instrument_trends_large_csv)
    df["n_topics"] = df["topic_ids"].apply(lambda s: len(ast.literal_eval(str(s))))
    df["abs_sev"] = df["severity"].abs()
    corr_val = df["n_topics"].corr(df["abs_sev"])
    logger.info("=== A) EMPIRICAL (saved large report) ===")
    table = (
        df[["instrument", "n_topics", "severity", "abs_sev"]]
        .sort_values("n_topics")
        .to_string(index=False)
    )
    logger.info(f"\n{table}")
    logger.info(
        f"Pearson corr(n_topics, |severity|) = {corr_val:.3f} (R^2 = {corr_val**2:.3f})"
    )
    logger.info(
        "Same-count spread: at n=1, |severity| ranges across instruments purely "
        "due to sentiment/source -> count alone does not determine the signal."
    )


# ----------------------------------------------------------------- B) elasticity
def measure_elasticity() -> None:
    """Perturb one input at a time around a baseline group and measure severity."""
    analyzer = _analyzer()
    base_n = 4
    base = _make_group(base_n, [0.5] * base_n, [0.8] * base_n, [2.0] * base_n)
    sev0 = analyzer._severity_dcim(base)

    g_count = _make_group(8, [0.5] * 8, [0.8] * 8, [2.0] * 8)
    sev_count = analyzer._severity_dcim(g_count)

    g_sent = _make_group(base_n, [1.0] * base_n, [0.8] * base_n, [2.0] * base_n)
    sev_sent = analyzer._severity_dcim(g_sent)

    g_zero = _make_group(base_n, [0.0] * base_n, [0.8] * base_n, [2.0] * base_n)
    sev_zero = analyzer._severity_dcim(g_zero)

    g_w = _make_group(base_n, [0.5] * base_n, [0.3] * base_n, [2.0] * base_n)
    sev_w = analyzer._severity_dcim(g_w)

    logger.info("=== B) ELASTICITY (perturb one input around a baseline) ===")
    logger.info(f"baseline (n=4, sent=0.5, w=0.8, age=2h):   severity = {sev0:+.3f}")
    logger.info(
        f"2x article count (n=8):                     {sev_count:+.3f}  "
        f"(delta={sev_count - sev0:+.3f})"
    )
    logger.info(
        f"sentiment 0.5 -> 1.0:                        {sev_sent:+.3f}  "
        f"(delta={sev_sent - sev0:+.3f})"
    )
    logger.info(
        f"sentiment -> 0.0 (gating test):              {sev_zero:+.3f}  "
        f"(delta={sev_zero - sev0:+.3f})"
    )
    logger.info(
        f"source weight 0.8 -> 0.3:                    {sev_w:+.3f}  "
        f"(delta={sev_w - sev0:+.3f})"
    )


# --------------------------------------------------- C) variance decomposition
def _random_instrument(
    analyzer: UpperPathAnalyzer, sources_df: pd.DataFrame
) -> tuple[float, dict[str, float]]:
    """Draw one realistic instrument and return (severity, features)."""
    n = int(RNG.integers(1, 21))
    sentiments = RNG.choice(SENTIMENT_LEVELS, size=n)
    src = sources_df.sample(n, replace=True, random_state=int(RNG.integers(1e9)))
    weights = src["source_weight"].to_numpy()
    names = src["source_name"].to_numpy()
    ages = RNG.uniform(0.0, 24.0, size=n)

    group = pd.DataFrame(
        {
            "date": [NOW - pd.Timedelta(hours=h) for h in ages],
            "source_name": names,
            "sentiment": sentiments,
            "source_weight": weights,
        }
    )
    sev = analyzer._severity_dcim(group)
    half_life = float(config.hyperparams.upper_path.dcim.half_life_hours)
    feats = {
        "count": float(n),
        "mean_sentiment": float(np.mean(sentiments)),
        "mean_abs_sentiment": float(np.mean(np.abs(sentiments))),
        "mean_weight": float(np.mean(weights)),
        "mean_recency": float(np.mean(np.exp(-ages / half_life))),
    }
    return sev, feats


def _r2(features_matrix: np.ndarray, targets: np.ndarray) -> float:
    """OLS R^2 of targets on columns features_matrix (with intercept)."""
    if features_matrix.shape[1] == 0:
        return 0.0
    design_matrix = np.column_stack([np.ones(len(targets)), features_matrix])
    beta, *_ = np.linalg.lstsq(design_matrix, targets, rcond=None)
    pred = design_matrix @ beta
    ss_res = float(np.sum((targets - pred) ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _shapley_r2(
    data: pd.DataFrame, predictors: list[str], target: str
) -> dict[str, float]:
    """Dominance analysis: partition total R^2 into per-predictor Shapley shares."""
    targets = data[target].to_numpy()
    num_preds = len(predictors)
    contrib = {key: 0.0 for key in predictors}
    for key in predictors:
        others = [col for col in predictors if col != key]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                with_k = list(subset) + [key]
                r2_with = _r2(data[with_k].to_numpy(), targets)
                r2_without = (
                    _r2(data[list(subset)].to_numpy(), targets) if subset else 0.0
                )
                weight = (
                    math.factorial(len(subset))
                    * math.factorial(num_preds - len(subset) - 1)
                    / math.factorial(num_preds)
                )
                contrib[key] += weight * (r2_with - r2_without)
    return contrib


def measure_decomposition(n_draws: int = 4000) -> None:
    """Run Shapley R^2 variance decomposition across random instruments."""
    analyzer = _analyzer()
    sources_df = pd.read_csv(config.paths.input.weights)
    rows = []
    sevs = []
    for _ in range(n_draws):
        sev, feats = _random_instrument(analyzer, sources_df)
        sevs.append(sev)
        rows.append(feats)
    data = pd.DataFrame(rows)
    data["abs_severity"] = np.abs(sevs)
    data["severity"] = sevs

    predictors = ["count", "mean_abs_sentiment", "mean_weight", "mean_recency"]

    logger.info("=== C) VARIANCE DECOMPOSITION (Shapley share of R^2) ===")
    logger.info(f"ensemble: {n_draws} random instruments through _severity_dcim")

    contrib_mag = _shapley_r2(data, predictors, "abs_severity")
    total_mag = sum(contrib_mag.values())
    logger.info("-- |severity| (signal MAGNITUDE) --")
    for key in predictors:
        share = contrib_mag[key] / total_mag * 100 if total_mag else 0
        logger.info(
            f"  {key:20s}: {contrib_mag[key]:.3f}  ({share:5.1f}% of explained)"
        )
    logger.info(f"  total R^2 = {total_mag:.3f}")
    logger.info(
        f"  >>> ARTICLE COUNT share of magnitude signal: "
        f"{contrib_mag['count'] / total_mag * 100:.1f}%"
    )

    data["sign"] = np.sign(data["severity"])
    dir_predictors = ["count", "mean_sentiment"]
    contrib_dir = _shapley_r2(data, dir_predictors, "sign")
    total_dir = sum(contrib_dir.values())
    logger.info("-- sign(severity) (signal DIRECTION) --")
    for key in dir_predictors:
        share = contrib_dir[key] / total_dir * 100 if total_dir else 0
        logger.info(
            f"  {key:20s}: {contrib_dir[key]:.3f}  ({share:5.1f}% of explained)"
        )
    logger.info(
        f"  >>> ARTICLE COUNT share of direction: "
        f"{contrib_dir['count'] / total_dir * 100:.1f}%"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    )
    measure_empirical()
    measure_elasticity()
    measure_decomposition()
