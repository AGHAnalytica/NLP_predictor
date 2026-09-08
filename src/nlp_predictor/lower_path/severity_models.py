"""
Severity models — the pure, parameterized functions the A/B/n compares.

Each function maps ONE instrument's article group to a signed severity in [-1, 1], using
only its explicit ``params`` (no global Config reads, no ``self``, no LLM, no I/O). That
purity is the whole point: every variant is deterministic, so when two demo accounts
diverge in PnL the only thing that differed was the ``(function, params, window)``
triple — the model is the isolated variable.

Uniform signature::

    severity(group, params: dict, now: datetime, context: dict | None) -> float

``group`` carries the columns the feature store emits (``date``, ``source_name``,
``sentiment``, ``source_weight``); ``now`` is the reference time for decay; ``context``
holds cross-instrument info a model may need (only ``abnormal_volume`` uses it, for the
per-tick baseline article count). Functions ignore params they don't use, and read the
ones they do via ``.get`` with the legacy DCIM defaults, so a partial param set is safe.

Registered models:
    - ``baseline``         : time-decayed, source-weighted sentiment sum (the control).
    - ``dcim``             : full Directional Confirmed Impact Model (5 phases).
    - ``abnormal_volume``  : magnitude from coverage vs a baseline, direction from
                             sentiment — the explicit test of "article volume is the
                             source of truth".
"""

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd

# Legacy DCIM defaults — used when a variant omits a param, so behaviour matches the
# pre-registry single-model path unless a variant deliberately overrides it.
_DEFAULTS = {
    "half_life_hours": 4.0,
    "beta_penalty": 0.2,
    "gamma_consensus": 0.3,
    "delta_velocity": 0.3,
    "kappa_scale": 1.0,
    "recent_window_hours": 0.5,
}


def _p(params: dict, key: str) -> float:
    """Read a hyperparameter, falling back to the legacy default."""
    return float(params.get(key, _DEFAULTS[key]))


def _decay(group: pd.DataFrame, now: datetime, half_life_hours: float):
    """Article age in hours (clamped >= 0) and its exponential decay factor."""
    time_diffs = (now - group["date"]).dt.total_seconds() / 3600.0
    time_diffs = np.maximum(time_diffs, 0.0)
    decay_constant = np.log(2) / half_life_hours
    return time_diffs, np.exp(-decay_constant * time_diffs)


def severity_baseline(
    group: pd.DataFrame, params: dict, now: datetime, context: dict | None = None
) -> float:
    """
    Minimal control: tanh of the time-decayed, source-weighted sentiment sum.

    Deliberately omits the echo / consensus / velocity terms — the simplest defensible
    severity, kept so the full DCIM has something to beat.
    """
    if group.empty:
        return 0.0
    _, decay_factors = _decay(group, now, _p(params, "half_life_hours"))
    raw = group["source_weight"] * group["sentiment"] * decay_factors
    total = float(raw.sum())
    return float(np.clip(math.tanh(_p(params, "kappa_scale") * total), -1.0, 1.0))


def severity_dcim(
    group: pd.DataFrame, params: dict, now: datetime, context: dict | None = None
) -> float:
    """
    Directional Confirmed Impact Model — five phases:
    (1) exponential time decay, (2) echo-chamber penalty for repeats from one source,
    (3) consensus bonus when independent sources agree, (4) velocity multiplier for
    breaking-news spikes, (5) tanh squash into [-1, 1].
    """
    if group.empty:
        return 0.0

    # PHASE 1 - per-article impact
    time_diffs, decay_factors = _decay(group, now, _p(params, "half_life_hours"))
    raw_impacts = group["source_weight"] * group["sentiment"] * decay_factors

    temp_df = group.copy()
    temp_df["raw_impact"] = raw_impacts
    temp_df = temp_df.sort_values(by="date", ascending=False)

    # PHASE 2 - echo-chamber penalty (diminishing returns per extra same-source article
    # in the same direction)
    beta = _p(params, "beta_penalty")
    source_impacts = []
    for _, s_group in temp_df.groupby("source_name"):
        s_sorted = s_group.sort_values(by="date", ascending=False)
        pos_impacts = s_sorted[s_sorted["raw_impact"] > 0]["raw_impact"].to_numpy()
        neg_impacts = s_sorted[s_sorted["raw_impact"] < 0]["raw_impact"].to_numpy()

        pos_penalties = 1.0 + beta * np.arange(len(pos_impacts))
        neg_penalties = 1.0 + beta * np.arange(len(neg_impacts))

        pos_sum = (
            float(np.sum(pos_impacts / pos_penalties)) if len(pos_impacts) > 0 else 0.0
        )
        neg_sum = (
            float(np.sum(neg_impacts / neg_penalties)) if len(neg_impacts) > 0 else 0.0
        )
        source_impacts.append(pos_sum + neg_sum)

    source_impacts = np.array(source_impacts)
    total_impact_sum = np.sum(source_impacts)
    if total_impact_sum == 0:
        return 0.0

    # PHASE 3 - consensus bonus (independent sources agreeing on direction)
    U = len(source_impacts)
    if U >= 2:
        global_sign = np.sign(total_impact_sum)
        U_agree = np.sum(np.sign(source_impacts) == global_sign)
        C = 1.0 + _p(params, "gamma_consensus") * (U_agree / U)
    else:
        C = 1.0

    # PHASE 4 - velocity (share of articles that are "breaking/recent")
    N_total = len(group)
    N_recent = np.sum(time_diffs <= _p(params, "recent_window_hours"))
    V = N_recent / (N_total + 1.0)

    # PHASE 5 - scale + squash
    inside_tanh = _p(params, "kappa_scale") * C
    inside_tanh *= 1.0 + _p(params, "delta_velocity") * V
    inside_tanh *= total_impact_sum
    return float(np.clip(math.tanh(inside_tanh), -1.0, 1.0))


def severity_abnormal_volume(
    group: pd.DataFrame, params: dict, now: datetime, context: dict | None = None
) -> float:
    """
    Volume-first severity: MAGNITUDE comes from how unusual this instrument's coverage
    is versus a baseline, DIRECTION from the sign of its (decayed, source-weighted)
    sentiment. This is the explicit test of the V1 thesis — "the amount of articles is
    the source of truth" — with sentiment used only to point the trade up or down.

    ``context["baseline_count"]`` is the typical article count to compare against (the
    orchestrator supplies it; until per-instrument history exists it is the cross-
    sectional mean count across instruments in the same window). Only ABOVE-baseline
    coverage produces a signal; normal/quiet coverage scores ~0.
    """
    if group.empty:
        return 0.0

    _, decay_factors = _decay(group, now, _p(params, "half_life_hours"))
    net = float((group["source_weight"] * group["sentiment"] * decay_factors).sum())
    direction = np.sign(net)
    if direction == 0:
        return 0.0  # no directional sentiment -> no trade, however loud the coverage

    count = float(len(group))
    baseline = float((context or {}).get("baseline_count", 1.0))
    baseline = max(baseline, 1e-9)
    excess = max(0.0, count / baseline - 1.0)  # how far above normal coverage we are
    magnitude = math.tanh(_p(params, "kappa_scale") * excess)
    return float(np.clip(direction * magnitude, -1.0, 1.0))


# Name -> function. A variant in config.severity.variants selects a model by name.
SEVERITY_REGISTRY = {
    "baseline": severity_baseline,
    "dcim": severity_dcim,
    "abnormal_volume": severity_abnormal_volume,
}

# Columns every severity model needs from the group.
REQUIRED_COLUMNS = ("date", "source_name", "sentiment", "source_weight")


# =============================================================================
# MACRO (lower-path) severity — cluster -> unsigned narrative INTENSITY
# =============================================================================
# Distinct from the per-instrument models above: a macro severity is UNSIGNED (a
# magnitude, not a signed [-1, 1] call). The cluster's DIRECTION is decided separately
# by the lower-path LLM; this only says how big / credible / fresh the narrative is,
# which feeds the climate conviction tanh(intensity / MACRO_CONVICTION_SCALE). The group
# is a cluster's topic rows (NOT exploded per instrument), so there is no 'sentiment'
# column — only volume, recency (date) and publisher credibility (source_weight).

_MACRO_DEFAULTS = {"half_life_hours": 18.0}


def macro_severity_count(
    group: pd.DataFrame, params: dict | None = None, now: datetime | None = None
) -> float:
    """Raw topic count — the legacy control (every topic weighs 1, fresh or stale)."""
    return float(len(group))


def macro_severity_decayed(
    group: pd.DataFrame, params: dict | None = None, now: datetime | None = None
) -> float:
    """
    Source-weighted, time-decayed narrative volume: ``Σ source_weight × decay(age)``.

    Brings the upper path's two cheapest, most defensible ideas — recency and publisher
    credibility — to the cluster level, so a fresh, reputable narrative outweighs a
    stale, low-credibility one of the same raw size. The half-life is deliberately LONG
    (a macro regime persists longer than a single instrument's news pulse) and is itself
    an A/B knob via ``params['half_life_hours']``. Falls back to a raw count if the
    needed columns are absent (e.g. a legacy topics table without source weights).
    """
    if group.empty:
        return 0.0
    if "date" not in group.columns or "source_weight" not in group.columns:
        return float(len(group))
    half_life = float(
        (params or {}).get("half_life_hours", _MACRO_DEFAULTS["half_life_hours"])
    )
    now = now or datetime.now(UTC)
    _, decay_factors = _decay(group, now, half_life)
    return float((group["source_weight"] * decay_factors).sum())


# Name -> function. config.hyperparams.lower_path.macro_severity_model picks which one
# the lower path uses.
MACRO_SEVERITY_REGISTRY = {
    "count": macro_severity_count,
    "decayed_weighted": macro_severity_decayed,
}
