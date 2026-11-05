"""Pure, parameterized severity functions for instrument-level (upper path) and cluster-
level (lower path) scoring.

All functions are deterministic and side-effect-free (no I/O, no LLM calls, no global
config reads)
to guarantee clean A/B/n attribution across variants.
"""

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd

_DEFAULTS: dict[str, float] = {
    "half_life_hours": 4.0,
    "beta_penalty": 0.2,
    "gamma_consensus": 0.3,
    "delta_velocity": 0.3,
    "kappa_scale": 1.0,
    "recent_window_hours": 0.5,
}

REQUIRED_COLUMNS = ("date", "source_name", "sentiment", "source_weight")

_MACRO_DEFAULTS: dict[str, float] = {"half_life_hours": 18.0}


def _get_param(params: dict, key: str) -> float:
    """Extract a float hyperparameter from ``params`` with fallback to ``_DEFAULTS``."""
    return float(params.get(key, _DEFAULTS[key]))


def _decay(
    group: pd.DataFrame, now: datetime, half_life_hours: float
) -> tuple[pd.Series, np.ndarray]:
    """Compute non-negative article age in hours and exponential decay factors
    $e^{-\\lambda \\Delta t}$.
    """
    time_diffs = (now - group["date"]).dt.total_seconds() / 3600.0
    time_diffs = np.maximum(time_diffs, 0.0)
    decay_constant = np.log(2) / half_life_hours
    return time_diffs, np.exp(-decay_constant * time_diffs)


def severity_baseline(
    group: pd.DataFrame, params: dict, now: datetime, context: dict | None = None
) -> float:
    """Compute control severity as $\\tanh(\\kappa \\sum w_i s_i e^{-\\lambda \\Delta
    t_i}) \\in [-1, 1]$.

    Args:
        group: Instrument impact rows with columns in ``REQUIRED_COLUMNS``.
        params: Hyperparameters (``half_life_hours``, ``kappa_scale``).
        now: Reference UTC timestamp for age decay.
        context: Unused cross-instrument context dict (kept for uniform signature).

    Returns:
        Signed severity score in ``[-1.0, 1.0]``.
    """
    if group.empty:
        return 0.0
    _, decay_factors = _decay(group, now, _get_param(params, "half_life_hours"))
    raw = group["source_weight"] * group["sentiment"] * decay_factors
    total = float(raw.sum())
    return float(
        np.clip(math.tanh(_get_param(params, "kappa_scale") * total), -1.0, 1.0)
    )


def severity_dcim(
    group: pd.DataFrame, params: dict, now: datetime, context: dict | None = None
) -> float:
    """Compute 5-phase Directional Confirmed Impact Model (DCIM) severity in ``[-1.0,
    1.0]``.

    Phases:
        1. Exponential time decay of source-weighted sentiment: $I_i = w_i s_i
            e^{-\\lambda \\Delta t_i}$.
        2. Per-source echo-chamber penalty: $1 / (1 + \\beta k)$ for $k$-th same-sign
            repeat from one publisher.
        3. Cross-source consensus bonus: $C = 1 + \\gamma (U_{\\text{agree}} / U)$ when
            $U \\ge 2$.
        4. Breaking-news velocity multiplier: $1 + \\delta (N_{\\text{recent}} /
            (N_{\\text{total}} + 1))$.
        5. Hyperbolic tangent squash: $\\tanh(\\kappa \\cdot C \\cdot (1 + \\delta V)
            \\cdot \\sum I_{\\text{source}})$.

    Args:
        group: Instrument impact rows with columns in ``REQUIRED_COLUMNS``.
        params: DCIM hyperparameters (``half_life_hours``, ``beta_penalty``,
            ``gamma_consensus``, ``delta_velocity``, ``kappa_scale``,
            ``recent_window_hours``).
        now: Reference UTC timestamp for age decay.
        context: Unused cross-instrument context dict.

    Returns:
        Signed severity score in ``[-1.0, 1.0]``.
    """
    if group.empty:
        return 0.0

    # Phase 1: Per-article decayed impact
    time_diffs, decay_factors = _decay(
        group, now, _get_param(params, "half_life_hours")
    )
    raw_impacts = group["source_weight"] * group["sentiment"] * decay_factors

    temp_df = group.assign(raw_impact=raw_impacts).sort_values(
        by="date", ascending=False
    )

    # Phase 2: Per-source echo-chamber penalty
    beta = _get_param(params, "beta_penalty")
    source_impacts: list[float] = []
    for _, s_group in temp_df.groupby("source_name", sort=False):
        impacts_arr = s_group["raw_impact"].to_numpy()
        pos_impacts = impacts_arr[impacts_arr > 0]
        neg_impacts = impacts_arr[impacts_arr < 0]

        pos_sum = (
            float(np.sum(pos_impacts / (1.0 + beta * np.arange(len(pos_impacts)))))
            if len(pos_impacts) > 0
            else 0.0
        )
        neg_sum = (
            float(np.sum(neg_impacts / (1.0 + beta * np.arange(len(neg_impacts)))))
            if len(neg_impacts) > 0
            else 0.0
        )
        source_impacts.append(pos_sum + neg_sum)

    source_impacts_arr = np.asarray(source_impacts, dtype=float)
    total_impact_sum = float(np.sum(source_impacts_arr))
    if total_impact_sum == 0.0:
        return 0.0

    # Phase 3: Cross-source consensus bonus
    u_total = len(source_impacts_arr)
    if u_total >= 2:
        global_sign = np.sign(total_impact_sum)
        u_agree = int(np.sum(np.sign(source_impacts_arr) == global_sign))
        consensus = 1.0 + _get_param(params, "gamma_consensus") * (u_agree / u_total)
    else:
        consensus = 1.0

    # Phase 4: Breaking-news velocity
    n_total = len(group)
    n_recent = int(np.sum(time_diffs <= _get_param(params, "recent_window_hours")))
    velocity = n_recent / (n_total + 1.0)

    # Phase 5: Scale and squash
    inside_tanh = (
        _get_param(params, "kappa_scale")
        * consensus
        * (1.0 + _get_param(params, "delta_velocity") * velocity)
        * total_impact_sum
    )
    return float(np.clip(math.tanh(inside_tanh), -1.0, 1.0))


def severity_abnormal_volume(
    group: pd.DataFrame, params: dict, now: datetime, context: dict | None = None
) -> float:
    """Compute volume-driven severity where magnitude reflects excess coverage over
    baseline.

    Formula:
        $\\text{sign}(\\sum w_i s_i e^{-\\lambda \\Delta t_i}) \\cdot \\tanh(\\kappa
        \\cdot \\max(0, N / B - 1))$
        where $B$ is ``context['baseline_count']``.

    Args:
        group: Instrument impact rows with columns in ``REQUIRED_COLUMNS``.
        params: Hyperparameters (``half_life_hours``, ``kappa_scale``).
        now: Reference UTC timestamp for age decay.
        context: Dictionary providing ``baseline_count`` (mean topic count across
            instruments).

    Returns:
        Signed severity score in ``[-1.0, 1.0]`` (0.0 if coverage is at or below
        baseline).
    """
    if group.empty:
        return 0.0

    _, decay_factors = _decay(group, now, _get_param(params, "half_life_hours"))
    net = float((group["source_weight"] * group["sentiment"] * decay_factors).sum())
    direction = float(np.sign(net))
    if direction == 0.0:
        return 0.0

    count = float(len(group))
    baseline = max(float((context or {}).get("baseline_count", 1.0)), 1e-9)
    excess = max(0.0, count / baseline - 1.0)
    magnitude = math.tanh(_get_param(params, "kappa_scale") * excess)
    return float(np.clip(direction * magnitude, -1.0, 1.0))


SEVERITY_REGISTRY = {
    "baseline": severity_baseline,
    "dcim": severity_dcim,
    "abnormal_volume": severity_abnormal_volume,
}


def macro_severity_count(
    group: pd.DataFrame, params: dict | None = None, now: datetime | None = None
) -> float:
    """Compute unsigned macro cluster intensity as the raw topic count $N$."""
    return float(len(group))


def macro_severity_decayed(
    group: pd.DataFrame, params: dict | None = None, now: datetime | None = None
) -> float:
    """Compute unsigned macro cluster intensity as $\\sum w_i e^{-\\lambda \\Delta
    t_i}$.

    Falls back to raw topic count if ``date`` or ``source_weight`` columns are missing.

    Args:
        group: Clustered topic rows.
        params: Dictionary containing ``half_life_hours`` (defaults to 18.0h).
        now: Reference UTC timestamp for age decay.

    Returns:
        Non-negative narrative intensity score.
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


MACRO_SEVERITY_REGISTRY = {
    "count": macro_severity_count,
    "decayed_weighted": macro_severity_decayed,
}
