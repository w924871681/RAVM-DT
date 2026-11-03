"""Frozen Stage4-R software policy and lock identity checks.

This module only validates the one-shot pilot contract. It never opens task
data, model weights, outcomes, service, or historical experiment results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from temporal_guard_reference import MARGIN, R_DEV

METHOD_VERSION = "TCBO_FIXED_ONE_SHOT_20261015"
REFERENCE_SHA256 = "cc26e5a1feae14905061ce8655a509f49d0c1dcaeb6f263023116d52f12df76b"
FIXED_NAMES = ("RAVM_LEGACY", "DUAL_LEGACY", "RAVM_CAP_ONLY", "DUAL_CAP_ONLY",
               "RAVM_TCBO", "DUAL_TCBO", "AlwaysFT", "Frozen")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fixed_stage4r_policies() -> list[dict]:
    """Exact eight-arm order; no search grid or method-specific bonus input."""
    result = []
    for name, method, variant in (
        ("RAVM_LEGACY", "RAVM", "LEGACY"),
        ("DUAL_LEGACY", "DualMean", "LEGACY"),
        ("RAVM_CAP_ONLY", "RAVM", "CAP_ONLY"),
        ("DUAL_CAP_ONLY", "DualMean", "CAP_ONLY"),
        ("RAVM_TCBO", "RAVM", "TEMPORAL_GUARD"),
        ("DUAL_TCBO", "DualMean", "TEMPORAL_GUARD")):
        result.append(dict(id=name, method=method, threshold=MARGIN[method],
                           retention_threshold=0.0, learning_rate=.01,
                           fallback=False, stage4r_variant=variant))
    result.extend((dict(id="AlwaysFT", method="AlwaysFT", threshold=0.,
                        retention_threshold=0., learning_rate=.01, fallback=False),
                   dict(id="Frozen", method="Frozen", threshold=0.,
                        retention_threshold=0., learning_rate=.01, fallback=False)))
    return result


def validate_fixed_stage4r_policies(policies: list[dict]) -> None:
    from stage3_engine import normalize_arm  # deferred to avoid import cycle
    expected = [normalize_arm(x) for x in fixed_stage4r_policies()]
    if policies != expected:
        raise ValueError("STAGE4R_LOCK_INVALID: policies differ from the frozen eight-arm order")


def verify_stage4r_lock(path: str | Path, *, expected_sha256: str,
                        code_dir: str | Path) -> str:
    """Bind the pre-outcome lock to exact code and the unchanged rule source."""
    path = Path(path)
    code_dir = Path(code_dir)
    if not path.is_file() or len(expected_sha256) != 64:
        raise ValueError("STAGE4R_LOCK_INVALID: frozen lock or expected SHA missing")
    digest = sha256_file(path)
    if digest != expected_sha256.lower():
        raise ValueError("STAGE4R_LOCK_INVALID: frozen lock SHA-256 mismatch")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if (obj.get("status") != "FROZEN_BEFORE_OUTCOMES" or
            obj.get("method_version") != METHOD_VERSION or
            obj.get("approved_method_version_sha256") != REFERENCE_SHA256):
        raise ValueError("STAGE4R_LOCK_INVALID: status or method reference changed")
    mechanism = obj.get("mechanism", {})
    pilot = obj.get("pilot", {})
    if (mechanism.get("recent_thresholds") != MARGIN or
            mechanism.get("reference_mse") != R_DEV or
            mechanism.get("history_original_tolerance") != 0.0 or
            mechanism.get("history_cap") !=
                "min(0.01*Rdev,max(0,-original_method_recent_score))" or
            mechanism.get("min_prior_history_blocks_for_both_enhanced_variants") != 12 or
            mechanism.get("recent_block_count") != 6 or
            mechanism.get("temporal_guard") !=
                "at least 5 of 6 chronological block mean deltas <= 0 AND latest three all <= 0" or
            mechanism.get("override_only_after_recent_pass_and_history_veto") is not True or
            mechanism.get("legacy_accepted_commits_remain_accepted") is not True or
            mechanism.get("risk_guarantee") is not False or
            pilot.get("methods_in_fixed_order") != list(FIXED_NAMES) or
            pilot.get("centers") != [43000,43001,43002,43003] or
            pilot.get("model_seeds") != [7311] or
            pilot.get("streams") != ["clean","mixed"] or
            pilot.get("mixed_corruption_alpha") != .10 or
            pilot.get("stages_per_trajectory") != 24 or
            pilot.get("logical_stage_records_expected") != 1536 or
            pilot.get("candidate_steps") != 50 or
            pilot.get("candidate_learning_rate") != .01):
        raise ValueError("STAGE4R_LOCK_INVALID: fixed scientific settings changed")
    expected_centers = {str(i) for i in range(43000, 43004)}
    arrays = obj.get("generated_raw_array_sha256_by_center")
    schedules = obj.get("all_stream_schedules_sha256")
    valid_digest = lambda x: isinstance(x, str) and len(x) == 64 and all(
        c in "0123456789abcdef" for c in x.lower())
    if (not isinstance(arrays, dict) or set(map(str, arrays)) != expected_centers or
            any(not valid_digest(v) for v in arrays.values()) or
            not valid_digest(obj.get("source_model_checkpoint_sha256")) or
            not isinstance(obj.get("exact_generator_and_rng_parameters"), dict) or
            not obj["exact_generator_and_rng_parameters"] or
            not isinstance(obj.get("data_generation_and_lock_time_utc"), str) or
            not obj["data_generation_and_lock_time_utc"]):
        raise ValueError("STAGE4R_LOCK_INVALID: complete pre-outcome input identities required")
    if isinstance(schedules, dict):
        expected_schedule_keys = {f"{center}:{stream}" for center in range(43000, 43004)
                                  for stream in ("clean", "mixed")}
        if set(schedules) != expected_schedule_keys or any(not valid_digest(v) for v in schedules.values()):
            raise ValueError("STAGE4R_LOCK_INVALID: eight stream schedule SHA identities required")
    elif not valid_digest(schedules):
        raise ValueError("STAGE4R_LOCK_INVALID: stream schedule SHA identity required")
    declared = obj.get("code_sha256")
    required = ("stage3_engine.py", "stage4r_gate.py", "stage4r_protocol.py",
                "temporal_guard_reference.py", "evidence_store.py",
                "core/engine.py", "core/models.py", "core/protocol.py")
    if not isinstance(declared, dict) or any(name not in declared for name in required):
        raise ValueError("STAGE4R_LOCK_INVALID: complete code SHA map required")
    for name in required:
        if sha256_file(code_dir / name) != declared[name]:
            raise ValueError(f"STAGE4R_LOCK_INVALID: code SHA mismatch for {name}")
    if sha256_file(code_dir / "temporal_guard_reference.py") != REFERENCE_SHA256:
        raise ValueError("STAGE4R_LOCK_INVALID: scientific reference source changed")
    return digest
