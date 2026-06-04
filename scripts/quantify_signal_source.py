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

import itertools
import math

import numpy as np
import pandas as pd

from nlp_predictor.config import Config
from nlp_predictor.instrument_trends import UpperPathAnalyzer

RNG = np.random.default_rng(42)
SENTIMENT_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
SOURCES = pd.read_csv(Config["WEIGHTS_PATH"])  # source_name, source_weight
NOW = pd.Timestamp.now(tz="UTC")


def _analyzer() -> UpperPathAnalyzer:
    """Build the real analyzer without touching the API / instrument loading."""
    a = UpperPathAnalyzer.__new__(UpperPathAnalyzer)
    a.current_time = NOW
    return a


def _make_group(n, sentiments, weights, ages_hours):
    """Construct one instrument's article group in the shape the DCIM expects."""
    return pd.DataFrame(
        {
            "date": [NOW - pd.Timedelta(hours=h) for h in ages_hours],
            "source_name": [f"src{i}" for i in range(n)],  # distinct by default
            "sentiment": sentiments,
            "source_weight": weights,
        }
    )


# ---------------------------------------------------------------- A) empirical
def measure_empirical():
    df = pd.read_csv(Config["INSTRUMENT_TRENDS_SAVE_PATH_CSV"].replace(
        "instrument_trends_report", "instrument_trends_report_large"))
    df["n_topics"] = df["topic_ids"].apply(lambda s: len(eval(s)))  # noqa: S307
    df["abs_sev"] = df["severity"].abs()
    r = df["n_topics"].corr(df["abs_sev"])
    print("\n=== A) EMPIRICAL (saved large report) ===")
    print(df[["instrument", "n_topics", "severity", "abs_sev"]]
          .sort_values("n_topics").to_string(index=False))
    print(f"\nPearson corr(n_topics, |severity|) = {r:.3f}  (R^2 = {r**2:.3f})")
    print("Same-count spread: at n=1, |severity| ranges across instruments purely")
    print("due to sentiment/source -> count alone does not determine the signal.")


# ----------------------------------------------------------------- B) elasticity
def measure_elasticity():
    a = _analyzer()
    # Baseline: 4 distinct sources, all sentiment +0.5, weight 0.8, age 2h.
    base_n = 4
    base = _make_group(base_n, [0.5] * base_n, [0.8] * base_n, [2.0] * base_n)
    sev0 = a._severity_dcim(base)

    # Double the count (8 identical-profile articles, still distinct sources).
    g_count = _make_group(8, [0.5] * 8, [0.8] * 8, [2.0] * 8)
    sev_count = a._severity_dcim(g_count)

    # Flip sentiment magnitude 0.5 -> 1.0 (count fixed).
    g_sent = _make_group(base_n, [1.0] * base_n, [0.8] * base_n, [2.0] * base_n)
    sev_sent = a._severity_dcim(g_sent)

    # Zero out sentiment (count fixed) -> the gating test.
    g_zero = _make_group(base_n, [0.0] * base_n, [0.8] * base_n, [2.0] * base_n)
    sev_zero = a._severity_dcim(g_zero)

    # Drop source weight 0.8 -> 0.3 (count fixed).
    g_w = _make_group(base_n, [0.5] * base_n, [0.3] * base_n, [2.0] * base_n)
    sev_w = a._severity_dcim(g_w)

    print("\n=== B) ELASTICITY (perturb one input around a baseline) ===")
    print(f"baseline (n=4, sent=0.5, w=0.8, age=2h):   severity = {sev0:+.3f}")
    print(f"2x article count (n=8):                     {sev_count:+.3f}  "
          f"(delta={sev_count - sev0:+.3f})")
    print(f"sentiment 0.5 -> 1.0:                        {sev_sent:+.3f}  "
          f"(delta={sev_sent - sev0:+.3f})")
    print(f"sentiment -> 0.0 (gating test):              {sev_zero:+.3f}  "
          f"(delta={sev_zero - sev0:+.3f})")
    print(f"source weight 0.8 -> 0.3:                    {sev_w:+.3f}  "
          f"(delta={sev_w - sev0:+.3f})")


# --------------------------------------------------- C) variance decomposition
def _random_instrument(a):
    """Draw one realistic instrument and return (severity, features)."""
    n = int(RNG.integers(1, 21))
    sentiments = RNG.choice(SENTIMENT_LEVELS, size=n)
    src = SOURCES.sample(n, replace=True, random_state=int(RNG.integers(1e9)))
    weights = src["source_weight"].to_numpy()
    names = src["source_name"].to_numpy()
    ages = RNG.uniform(0.0, 24.0, size=n)

    group = pd.DataFrame({
        "date": [NOW - pd.Timedelta(hours=h) for h in ages],
        "source_name": names,
        "sentiment": sentiments,
        "source_weight": weights,
    })
    sev = a._severity_dcim(group)
    feats = {
        "count": n,
        "mean_sentiment": float(np.mean(sentiments)),
        "mean_abs_sentiment": float(np.mean(np.abs(sentiments))),
        "mean_weight": float(np.mean(weights)),
        "mean_recency": float(np.mean(np.exp(-ages / 4.0))),  # decay proxy
    }
    return sev, feats


def _r2(X, y):
    """OLS R^2 of y on columns X (with intercept)."""
    if X.shape[1] == 0:
        return 0.0
    Xi = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    pred = Xi @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _shapley_r2(data, predictors, target):
    """Dominance analysis: partition total R^2 into per-predictor Shapley shares."""
    y = data[target].to_numpy()
    p = len(predictors)
    contrib = {k: 0.0 for k in predictors}
    for k in predictors:
        others = [x for x in predictors if x != k]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                with_k = list(subset) + [k]
                r2_with = _r2(data[with_k].to_numpy(), y)
                r2_without = _r2(data[list(subset)].to_numpy(), y) if subset else 0.0
                # Shapley weight for marginal contributions.
                w = (math.factorial(len(subset)) *
                     math.factorial(p - len(subset) - 1) / math.factorial(p))
                contrib[k] += w * (r2_with - r2_without)
    return contrib


def measure_decomposition(n_draws=4000):
    a = _analyzer()
    rows = []
    sevs = []
    for _ in range(n_draws):
        sev, feats = _random_instrument(a)
        sevs.append(sev)
        rows.append(feats)
    data = pd.DataFrame(rows)
    data["abs_severity"] = np.abs(sevs)
    data["severity"] = sevs

    predictors = ["count", "mean_abs_sentiment", "mean_weight", "mean_recency"]

    print("\n=== C) VARIANCE DECOMPOSITION (Shapley share of R^2) ===")
    print(f"ensemble: {n_draws} random instruments through the real _severity_dcim")

    # Magnitude model: what drives |severity|.
    contrib_mag = _shapley_r2(data, predictors, "abs_severity")
    total_mag = sum(contrib_mag.values())
    print("\n-- |severity| (signal MAGNITUDE) --")
    for k in predictors:
        share = contrib_mag[k] / total_mag * 100 if total_mag else 0
        print(f"  {k:20s}: {contrib_mag[k]:.3f}  ({share:5.1f}% of explained)")
    print(f"  total R^2 = {total_mag:.3f}")
    print(f"  >>> ARTICLE COUNT share of magnitude signal: "
          f"{contrib_mag['count'] / total_mag * 100:.1f}%")

    # Direction model: what drives the sign.
    data["sign"] = np.sign(data["severity"])
    dir_predictors = ["count", "mean_sentiment"]
    contrib_dir = _shapley_r2(data, dir_predictors, "sign")
    total_dir = sum(contrib_dir.values())
    print("\n-- sign(severity) (signal DIRECTION) --")
    for k in dir_predictors:
        share = contrib_dir[k] / total_dir * 100 if total_dir else 0
        print(f"  {k:20s}: {contrib_dir[k]:.3f}  ({share:5.1f}% of explained)")
    print(f"  >>> ARTICLE COUNT share of direction: "
          f"{contrib_dir['count'] / total_dir * 100:.1f}%")


if __name__ == "__main__":
    measure_empirical()
    measure_elasticity()
    measure_decomposition()

