from pydantic import BaseModel


class SignalHyperparams(BaseModel):
    neutral_threshold: float
    macro_align_bonus: float
    macro_conflict_penalty: float
    macro_veto: bool
    veto_coverage: float
    veto_strength_cap: float
    macro_conviction_scale: float
