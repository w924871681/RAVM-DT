"""Pinned original CMC scorer/gate for future Stage5 fair comparisons.

This adapter performs inference only.  It deliberately does not accept a
Stage3 ``score_models`` result because the original CMC ``_paired_score``
infers each block separately and its two-score admissibility contract is
distinct.  A corrected read-only GPU probe found a finite score difference
between the per-block and concatenated paths on prior saved nodes, while the
Stage3 recomputation matched its archive.  Calling this adapter on any real
Stage5 lifecycle remains NOT_RUN until an approved experiment stage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence


ORIGINAL_ENGINE_SHA256 = "f27d0abb6fabb50453cb4e04684b8e7b5f99d27aca7ae1f1755990640f3ce024"
ORIGINAL_CONFIG_SHA256 = "7275a287ee4a1f4602ed0a10a67362a1a61ffa398848d4529669b0584a127d13"
ORIGINAL_FAMILY = "V2_SP_ShortDual"
HISTORY_CAPACITY = 12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class OriginalCMCDecision:
    """Scores in native squared-error units, before any Rdev normalization."""

    recent_score: float
    history_score: float
    recent_median: float
    recent_mad: float
    history_median: float
    history_mad: float
    recent_block_count: int
    history_block_count: int
    commit: bool
    reason: str


class OriginalCMCGate:
    """Bind and call the authenticated CMC V2 per-block scoring function.

    ``source_root`` is the directory containing ``core/v2_engine.py``.  The
    caller must put this exact root on Python's import path before binding.
    An already-loaded ``core`` package from another root is rejected rather
    than silently mixing provenance.  The frozen JSON-formatted YAML config
    is read only after its byte hash matches the archived source.
    """

    def __init__(self, source_root: Path, frozen_config: Path):
        root = Path(source_root).resolve(strict=True)
        source = (root / "core" / "v2_engine.py").resolve(strict=True)
        config_path = Path(frozen_config).resolve(strict=True)
        if _sha256(source) != ORIGINAL_ENGINE_SHA256:
            raise ValueError("Original CMC engine SHA-256 mismatch")
        if _sha256(config_path) != ORIGINAL_CONFIG_SHA256:
            raise ValueError("Original frozen CMC config SHA-256 mismatch")
        spec = importlib.util.find_spec("core.v2_engine")
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != source:
            raise ImportError("core.v2_engine is not loaded from the authenticated source root")
        module = importlib.import_module("core.v2_engine")
        paired_score = getattr(module, "_paired_score", None)
        if not callable(paired_score) or Path(inspect.getsourcefile(paired_score) or "").resolve() != source:
            raise ImportError("Bound _paired_score is not the authenticated original callable")
        if _sha256(source) != ORIGINAL_ENGINE_SHA256:
            raise ValueError("Original CMC engine changed during import")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config.get("schema") != "ravm-dt-v2-final-commit-config-v1"
                or config.get("status") != "FINAL_CONFIG_FROZEN"):
            raise ValueError("Unexpected original CMC frozen config identity")
        expected = {"kappa_rec": 0.0, "kappa_hist": 0.5,
                    "delta_imp": 2.8018237766243602e-05, "delta_ret": 0.0}
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("Original CMC gate parameters do not match the frozen run")
        self._paired_score: Callable[..., Any] = paired_score
        self._config = {k: float(config[k]) for k in expected}
        self.source_sha256 = ORIGINAL_ENGINE_SHA256
        self.config_sha256 = ORIGINAL_CONFIG_SHA256

    def evaluate(self, candidate: Any, parent: Any, recent_blocks: Sequence[Any],
                 prior_history_blocks: Sequence[Any], horizon: int = 1) -> OriginalCMCDecision:
        """Score the same candidate/parent per block, then apply the V2 gate.

        Evidence is supplied as blocks; a precomputed Stage3 score dictionary
        cannot be used.  The bounded historical buffer is the last 12 *prior*
        blocks, before current recent evidence enters history.  No Service
        labels are an input to the decision.
        """
        if candidate is None or parent is None or isinstance(candidate, dict) or isinstance(parent, dict):
            raise TypeError("Candidate and parent must be model objects, not a matched score dictionary")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        recent = tuple(recent_blocks)
        history = tuple(prior_history_blocks)[-HISTORY_CAPACITY:]
        if not recent or not history:
            raise ValueError("Original CMC comparison requires recent and prior historical blocks")
        u_rec, m_rec, mad_rec, _, _ = self._paired_score(
            candidate, parent, recent, horizon, self._config["kappa_rec"])
        u_hist, m_hist, mad_hist, _, _ = self._paired_score(
            candidate, parent, history, horizon, self._config["kappa_hist"])
        values = (u_rec, u_hist, m_rec, mad_rec, m_hist, mad_hist)
        finite = all(math.isfinite(float(x)) for x in values)
        recent_pass = finite and u_rec <= -self._config["delta_imp"]
        history_pass = finite and u_hist <= self._config["delta_ret"]
        commit = bool(recent_pass and history_pass)
        reason = ("NONFINITE_RETAIN" if not finite else "ACCEPT" if commit else
                  "RECENT_AND_HISTORY_FAIL" if not recent_pass and not history_pass else
                  "RECENT_FAIL" if not recent_pass else "HISTORY_VETO")
        return OriginalCMCDecision(
            recent_score=float(u_rec), history_score=float(u_hist),
            recent_median=float(m_rec), recent_mad=float(mad_rec),
            history_median=float(m_hist), history_mad=float(mad_hist),
            recent_block_count=len(recent), history_block_count=len(history),
            commit=commit, reason=reason)
