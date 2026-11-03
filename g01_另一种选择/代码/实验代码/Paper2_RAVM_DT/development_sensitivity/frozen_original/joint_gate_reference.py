"""D3f-2 mathematically minimal, *offline reference only*.

Not a deployable gate. Real experiment must use the frozen native Stage3
recent-mean and MWU scoring path and assert interface identity.
No Service, stream type, regime, attack mask or historical oracle inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

PIVOT = 0.2  # From existing MWU_MEAN comparator; not a risk probability.
EPS = 1e-12

@dataclass(frozen=True)
class GateResult:
    commit: bool
    score: float
    recent_mean: float
    parent_verification_mean_loss: float
    mwu_p: float
    margin: float
    weight: float


def joint_decision(
    recent_mean: float,
    parent_verification_mean_loss: float,
    mwu_p: float,
    *,
    margin: float,
    weight: float,
) -> GateResult:
    vals = (recent_mean,parent_verification_mean_loss,mwu_p,margin,weight)
    if not all(isfinite(v) for v in vals):
        raise ValueError("All gate inputs must be finite")
    if parent_verification_mean_loss < 0 or margin < 0 or weight < 0 or not 0 <= mwu_p <= 1:
        raise ValueError("Invalid gate input range")
    scale = max(parent_verification_mean_loss, EPS)
    score = recent_mean + weight * scale * (mwu_p - PIVOT)
    # A positive/zero mean difference cannot be rescued by the p-value.
    # Use a positive calibrated margin, so weight=0 matches RecentMean.
    commit = bool(recent_mean < 0 and score <= -margin)
    return GateResult(commit,score,recent_mean,parent_verification_mean_loss,mwu_p,margin,weight)
