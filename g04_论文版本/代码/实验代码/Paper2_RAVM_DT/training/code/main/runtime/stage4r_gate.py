"""Pre-service Stage4-R adapter around the unmodified reference rule.

It accepts only original six scores, ordered recent block means and a prior
history count from the actual Stage3A DAG. No provider, scenario metadata,
attack mask or service object can enter this interface.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from temporal_guard_reference import decide as reference_decide

SCORE_KEYS = frozenset(("recent_mean", "recent_median", "history_mean",
                        "history_robust", "cumulative_mean", "mwu_p"))
ENHANCED = frozenset(("CAP_ONLY", "TEMPORAL_GUARD"))


def _recent_order_and_deltas(score_detail: Mapping | None,
                             recent_segments: Sequence[object],
                             recent_window_counts: Sequence[int],
                             total_verification_blocks: int) -> tuple[float, ...]:
    """Check that six d_j map to the six chronological actual recent blocks."""
    if (score_detail is None or len(recent_segments) != 6 or
            len(recent_window_counts) != 6 or
            not isinstance(total_verification_blocks, int) or
            total_verification_blocks < 6):
        raise ValueError("RECENT_BLOCKS_UNAVAILABLE")
    starts = []
    ticks = set()
    for block in recent_segments:
        ts = np.asarray(block.ts)
        if (getattr(block, "role", None) != "verification" or
                ts.ndim != 1 or len(ts) < 2 or
                not np.issubdtype(ts.dtype, np.integer)):
            raise ValueError("RECENT_BLOCK_TIME_INVALID")
        increments = np.diff(ts)
        if (increments <= 0).any() or (increments != increments[0]).any():
            raise ValueError("RECENT_BLOCK_TIME_INVALID")
        starts.append(int(ts[0]))
        ticks.add(int(increments[0]))
    if len(ticks) != 1 or starts != sorted(starts) or len(set(starts)) != 6:
        raise ValueError("RECENT_BLOCK_ORDER_INVALID")
    means = np.asarray(score_detail.get("block_means"), dtype=float)
    counts = np.asarray(score_detail.get("counts"))
    if (means.ndim != 1 or len(means) != total_verification_blocks or
            counts.ndim != 1 or len(counts) != len(means) or
            not np.isfinite(means).all() or
            not np.issubdtype(counts.dtype, np.integer) or
            (counts <= 0).any() or
            not np.array_equal(counts[-6:], np.asarray(recent_window_counts))):
        raise ValueError("RECENT_BLOCK_INDEX_INVALID")
    return tuple(float(v) for v in means[-6:])


def stage4r_decision_details(a: Mapping, scores: Mapping,
                             score_detail: Mapping | None,
                             *, prior_history_block_count: int,
                             recent_segments: Sequence[object],
                             recent_window_counts: Sequence[int],
                             total_verification_blocks: int) -> dict:
    """Return original decision plus fixed cap/TCBO diagnostics and action.

    A legacy acceptance is returned unchanged before checking optional block
    diagnostics. Enhanced overrides require a recent pass, history veto,
    mature prior FIFO and valid chronological d1..d6.
    """
    method = a["method"]
    variant = a.get("stage4r_variant", "LEGACY")
    if method not in ("RAVM", "DualMean") or variant not in ("LEGACY", *ENHANCED):
        raise ValueError("Stage4-R gate accepts only RAVM/DualMean fixed variants")
    if (isinstance(prior_history_block_count, bool) or
            not isinstance(prior_history_block_count, int) or
            prior_history_block_count < 0):
        raise ValueError("Invalid predecision history count")
    blank = dict(stage4r_variant=variant, stage4r_legacy_commit=0,
                 stage4r_override=0, stage4r_reason="",
                 stage4r_prior_history_blocks=prior_history_block_count,
                 stage4r_history_cap="", stage4r_guard_nonpositive_count="",
                 stage4r_latest_three_nonpositive="", stage4r_guard_pass="",
                 stage4r_guard_used=variant == "TEMPORAL_GUARD",
                 **{f"stage4r_d{i}": "" for i in range(1, 7)})
    if frozenset(scores) != SCORE_KEYS:
        return dict(commit=0, recent_pass=False, history_pass=False,
                    reject_reason="INVALID_SCORES_RETAIN", **{**blank,
                    "stage4r_reason": "INVALID_SCORES_RETAIN"})
    try:
        from stage3_engine import decision_details  # original, byte-identical routine
        legacy = decision_details(dict(a), dict(scores))
        original = bool(legacy["commit"])
        recent_pass = bool(legacy["recent_pass"])
        history_pass = bool(legacy["history_pass"])
    except (TypeError, ValueError, KeyError, FloatingPointError):
        return dict(commit=0, recent_pass=False, history_pass=False,
                    reject_reason="INVALID_SCORES_RETAIN", **{**blank,
                    "stage4r_reason": "INVALID_SCORES_RETAIN"})
    blank["stage4r_legacy_commit"] = int(original)
    if a.get("fallback", False):
        return dict(**legacy, **{**blank,
                    "stage4r_reason": "FROZEN_FALLBACK"})
    base = dict(**legacy, **{**blank,
        "stage4r_reason": ("LEGACY_ACCEPT" if original else
                            "RECENT_FAIL" if not recent_pass else "HISTORY_VETO")})
    # For all six gate arms, log actual ordered d1..d6 when the score detail
    # exists. Its validity affects only a possible *new* override.
    try:
        deltas = _recent_order_and_deltas(score_detail, recent_segments,
                                          recent_window_counts,
                                          total_verification_blocks)
        diagnostics = {f"stage4r_d{i}": x for i, x in enumerate(deltas, 1)}
        count = sum(x <= 0. for x in deltas)
        latest = all(x <= 0. for x in deltas[3:])
        diagnostics.update(stage4r_guard_nonpositive_count=count,
                           stage4r_latest_three_nonpositive=latest,
                           stage4r_guard_pass=bool(count >= 5 and latest))
        base.update(diagnostics)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        if variant in ENHANCED and recent_pass and not history_pass:
            base.update(commit=0, reject_reason="RECENT_BLOCK_INDEX_INVALID",
                        stage4r_reason=f"RECENT_BLOCK_INDEX_INVALID:{exc}")
        return base
    if variant == "LEGACY" or original or not recent_pass:
        return base
    if prior_history_block_count < 12:
        base.update(reject_reason="HISTORY_SUPPORT_INSUFFICIENT",
                    stage4r_reason="HISTORY_SUPPORT_INSUFFICIENT")
        return base
    try:
        verdict = reference_decide(method, variant, scores, deltas,
                                   prior_history_block_count, fallback=False)
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        base.update(commit=0, reject_reason="INVALID_EVIDENCE_RETAIN",
                    stage4r_reason=f"INVALID_EVIDENCE_RETAIN:{exc}")
        return base
    base.update(commit=int(verdict.commit), reject_reason=verdict.reason,
                stage4r_reason=verdict.reason,
                stage4r_override=int(verdict.commit and not original),
                stage4r_history_cap=(verdict.history_cap if verdict.history_cap is not None else ""))
    return base
