"""Reference-only Stage4-R paired temporal-consensus override.

This is an isolated pure decision reference for future *development* research.
It is not the original RAVM-DT implementation, trained model, or a validated
algorithm. It must not be installed in the old experiment or manuscript.

The caps and guards are exploratory hypotheses fixed before any new 430xx
outcome inspection; they carry NO statistical/physical safety guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Mapping, Sequence

Method = Literal["RAVM", "DualMean"]
Variant = Literal["LEGACY", "CAP_ONLY", "TEMPORAL_GUARD"]

# Reuse audited original development reference scale, *not* new source reference.
R_DEV = 0.00560364755324872
CAP_MAX = 0.01 * R_DEV
MARGIN = {"RAVM": 0.0, "DualMean": 0.0001681094265974}  # exact frozen selected value
REC_KEY = {"RAVM": "recent_median", "DualMean": "recent_mean"}
HIST_KEY = {"RAVM": "history_robust", "DualMean": "history_mean"}


@dataclass(frozen=True)
class Verdict:
    commit: bool
    reason: str
    recent_pass: bool
    history_pass: bool
    temporal_guard_pass: bool | None
    history_cap: float | None


def decide(method: Method, variant: Variant, scores: Mapping[str, float],
           recent_block_deltas: Sequence[float] | None,
           prior_history_block_count: int,
           *, fallback: bool = False) -> Verdict:
    """Return an online decision from strictly pre-service evidence.

    - Never consumes service, real regime, attack mask or future outcomes.
    - recent_block_deltas: *chronological* 6 paired blockwise MSE differences.
    - For shared fairness, identical CAP_ONLY/TEMPORAL_GUARD formulas apply to
      RAVM and DualMean; only their native recent/historical score keys differ.
    - Override never reverses an original accepted commitment.
    """
    if method not in REC_KEY or variant not in ("LEGACY", "CAP_ONLY", "TEMPORAL_GUARD"):
        raise ValueError("Unknown method or variant")
    if fallback:
        return Verdict(False, "FROZEN_FALLBACK", False, False, None, None)
    if not REC_KEY[method] in scores or not HIST_KEY[method] in scores:
        raise ValueError("Missing required scores")
    # Fail closed, rather than silently allowing untrusted nonfinite evidence.
    recent = float(scores[REC_KEY[method]])
    historic = float(scores[HIST_KEY[method]])
    if not all(isfinite(float(x)) for x in scores.values()) or not isfinite(recent) or not isfinite(historic):
        return Verdict(False, "INVALID_SCORE", False, False, None, None)
    rp = recent <= -MARGIN[method]
    hp = historic <= 0.0
    if rp and hp:
        return Verdict(True, "LEGACY_ACCEPT", True, True, None, None)
    if not rp:
        return Verdict(False, "RECENT_FAIL", False, hp, None, None)
    if variant == "LEGACY":
        return Verdict(False, "HISTORY_VETO", True, False, None, None)
    # Both enhanced variants require the identical mature 12-block historical support.
    if prior_history_block_count < 12:
        return Verdict(False, "HISTORY_SUPPORT_INSUFFICIENT", True, False, False, None)
    # This bounded *loss-scale heuristic* is not the original risk epsilon test.
    cap = min(CAP_MAX, max(0.0, -recent))
    if not (0.0 < historic <= cap):
        return Verdict(False, "HISTORY_EXCEEDS_BOUNDED_CAP", True, False, None, cap)
    if variant == "CAP_ONLY":
        return Verdict(True, "CAP_ONLY_OVERRIDE", True, False, None, cap)
    if recent_block_deltas is None or len(recent_block_deltas) != 6:
        return Verdict(False, "RECENT_BLOCKS_UNAVAILABLE", True, False, False, cap)
    deltas = [float(x) for x in recent_block_deltas]
    if not all(isfinite(x) for x in deltas):
        return Verdict(False, "NONFINITE_RECENT_BLOCK", True, False, False, cap)
    # Chronological candidate-level consistency: >=5/6 blocks favor the
    # candidate and the three *latest* blocks must all favor it.
    support = sum(d <= 0.0 for d in deltas) >= 5 and all(d <= 0.0 for d in deltas[3:])
    if not support:
        return Verdict(False, "TEMPORAL_CONSENSUS_FAIL", True, False, False, cap)
    return Verdict(True, "TEMPORAL_BOUNDED_OVERRIDE", True, False, True, cap)
