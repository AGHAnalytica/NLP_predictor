"""Hyperparameters for signal gating, macro confidence modulation, and veto
thresholds.
"""

from pydantic import BaseModel


class SignalHyperparams(BaseModel):
    """Thresholds and multipliers governing final signal generation and macro vetoes."""

    neutral_threshold: float
    macro_align_bonus: float
    macro_conflict_penalty: float
    macro_veto: bool
    veto_coverage: float
    veto_strength_cap: float
    macro_conviction_scale: float
