"""Development/calibration-only RAVM-DT V2 executor.

This module is deliberately isolated from the V1 engine.  It rejects every
center outside the existing method-development and decision-calibration pools.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from core.engine import evidence_blocks
from core.models import DEVICE, Forecast, adapt, errors, robust_candidate, state_hash, sync, tensors
from core.protocol import ATTACKS, ROOT, Segment, Timeline, digest, windows


ALLOWED_CENTERS = frozenset(range(11000, 11020)) | frozenset(range(12000, 12020))
V1_CANDIDATE = {"B": 5, "beta": 0.1}
V1_GATE = {
    "kappa_rec": 0.5,
    "kappa_hist": 0.0,
    "delta_imp": 2.8018237766243602e-05,
    "delta_ret": 0.0,
}


def _load_initial_model(center: int, seed: int, backbone: str, horizon: int) -> Forecast:
    path = ROOT / "checkpoints" / f"M0_{backbone}_H{horizon}_{seed}_{center}.pt"
    if not path.exists():
        raise FileNotFoundError(f"V2 refuses to create or alter V1 checkpoints: {path}")
    model = Forecast(backbone, horizon).to(DEVICE)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=DEVICE))
    return model


def _v2_attack(
    seg: Segment,
    center: int,
    stage: int,
    kind: str,
    alpha: float,
    location: str,
    seed: int = 9417,
) -> tuple[Segment, np.ndarray, np.ndarray, str]:
    """Exact V1 attack logic with audit artifacts redirected to scratch_v2."""
    if seg.role not in ("update", "verification"):
        raise AssertionError("attack may only touch maintenance telemetry")
    out = seg.copy()
    n = len(seg.y)
    mask = np.zeros(n, bool)
    source_ts = np.full(n, -1, dtype=np.int64)
    active = (
        location == "both"
        or (location == "update_only" and seg.role == "update")
        or (location == "verification_only" and seg.role == "verification")
    )
    actual = kind
    if kind == "mixed":
        order = np.random.default_rng(np.random.SeedSequence([seed, center, (stage - 1) // 4])).permutation(4)
        actual = ATTACKS[order[(stage - 1) % 4]]
    rng = np.random.default_rng(np.random.SeedSequence([seed, center, stage, 0 if seg.role == "update" else 1]))
    if active and alpha > 0:
        for block in np.array_split(np.arange(n), 10):
            count = int(np.floor(len(block) * alpha + 1e-9))
            if count:
                ids = (
                    block[int(rng.integers(0, len(block) - count + 1)) :][:count]
                    if actual == "burst"
                    else rng.choice(block, count, replace=False)
                )
                mask[ids] = True
        dims = out.x.shape[1] // 2
        observed = out.x[:, :dims]
        observed_mask = out.x[:, dims:]
        sign = np.random.default_rng(np.random.SeedSequence([seed, center, stage])).choice([-1, 1])
        if actual == "replay":
            old = Timeline(center).past_replay(stage, n)
            out.x[mask] = old.x[mask]
            out.y[mask] = old.y[mask]
            source_ts[mask] = old.ts[mask]
            assert np.all(source_ts[mask] < seg.ts[0])
        else:
            selected = rng.choice(dims, max(1, dims // 2), replace=False)
            change = np.zeros_like(observed)
            if actual == "random":
                change[mask] = np.clip(rng.normal(0, 0.15, (int(mask.sum()), dims)), -0.2, 0.2)
                out.y[mask] = np.clip(
                    out.y[mask] + np.clip(rng.normal(0, 0.15, int(mask.sum())), -0.2, 0.2), 0, 1
                )
            else:
                change[np.ix_(mask, selected)] = sign * 0.2
                out.y[mask] = np.clip(out.y[mask] + sign * 0.2, 0, 1)
            observed[:] = np.clip(observed + change * observed_mask, 0, 1) * observed_mask
    assert mask.sum() <= np.floor(n * alpha + 1e-8) if active else not mask.any()

    audit_dir = ROOT / "scratch_v2" / "attack_masks"
    audit_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{center}:{stage}:{seg.role}:{actual}:{alpha}:{location}:{seed}".encode()).hexdigest()
    path = audit_dir / (key + ".npz")
    if not path.exists():
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(
                stream,
                mask=mask,
                source_timestamps=source_ts,
                target_timestamps=seg.ts,
                center=center,
                stage=stage,
                role=seg.role,
                kind=actual,
                alpha=alpha,
                location=location,
            )
        try:
            if not path.exists():
                os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    return out, mask, source_ts, actual


def _paired_score(candidate, parent, blocks, horizon: int, kappa: float):
    values: list[float] = []
    calls = 0
    for block in blocks:
        batch = tensors(windows(block, H=horizon))
        candidate_loss, _ = errors(candidate, batch)
        parent_loss, _ = errors(parent, batch)
        values.append(float((candidate_loss - parent_loss).mean()))
        calls += 2 * ((len(candidate_loss) + 255) // 256)
    if not values:
        return float("inf"), float("nan"), float("nan"), np.empty(0), calls
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median + kappa * mad, median, mad, array, calls


def _family(structure: dict) -> str:
    value = structure["family"]
    allowed = {
        "V0_LegacyFull",
        "V1_SP_StaticDual",
        "V2_SP_ShortDual",
        "V3_SP_AdaptiveDual",
        "V4_SP_AdaptiveDual_NoRealign",
        "D0_SP_RecentOnly",
    }
    if value not in allowed:
        raise ValueError(f"unknown V2 structure family: {value}")
    return value


def run_v2(
    center: int,
    seed: int,
    structure: dict,
    kind: str = "clean",
    alpha: float = 0.0,
    location: str = "both",
    gate: dict | None = None,
    stages: int = 24,
    backbone: str = "GRU",
    horizon: int = 1,
    run_id: str = "",
) -> list[dict]:
    if center not in ALLOWED_CENTERS:
        raise PermissionError(f"V2 development executor refuses sealed/heldout center {center}")
    family = _family(structure)
    gate = {**V1_GATE, **(gate or {})}
    long_capacity = 12 if family == "V2_SP_ShortDual" else 24
    h_short = int(structure.get("h_short", 12))
    support_min = float(structure.get("support_min", 5 / 6))
    r_persist = int(structure.get("r_persist", 2))
    realign = family == "V3_SP_AdaptiveDual"

    timeline = Timeline(center)
    parent = _load_initial_model(center, seed, backbone, horizon)
    initial_hash = state_hash(parent)
    history = evidence_blocks(timeline.initial_history(), horizon)
    rows: list[dict] = []
    conflict_count = 0

    for stage in range(1, stages + 1):
        raw_update, raw_verification = timeline.maintenance(stage)
        update, update_mask, update_source, actual = _v2_attack(
            raw_update, center, stage, kind, alpha, location
        )
        verification, verification_mask, verification_source, _ = _v2_attack(
            raw_verification, center, stage, kind, alpha, location
        )
        before = state_hash(parent)
        sync()
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        maintenance_start = time.perf_counter()
        update_batch = tensors(windows(update, H=horizon))
        candidate_start = time.perf_counter()
        if family == "V0_LegacyFull":
            candidate = robust_candidate(
                parent,
                update_batch,
                V1_CANDIDATE["B"],
                V1_CANDIDATE["beta"],
                seed + center + stage * 100,
                False,
                horizon,
            )
        else:
            candidate = adapt(parent, update_batch, steps=50, seed=seed + center + stage * 100, loss="mse")
        sync()
        update_time = time.perf_counter() - candidate_start
        assert state_hash(parent) == before, "candidate modified deployed parent"

        verification_start = time.perf_counter()
        recent = evidence_blocks(verification, horizon)
        u_rec, median_rec, mad_rec, rec_values, rec_calls = _paired_score(
            candidate, parent, recent, horizon, gate["kappa_rec"]
        )
        if family == "D0_SP_RecentOnly":
            u_long = median_long = mad_long = float("nan")
            u_short = median_short = mad_short = float("nan")
            long_calls = short_calls = 0
        else:
            u_long, median_long, mad_long, _, long_calls = _paired_score(
                candidate, parent, history, horizon, gate["kappa_hist"]
            )
            if family in {"V3_SP_AdaptiveDual", "V4_SP_AdaptiveDual_NoRealign"}:
                u_short, median_short, mad_short, _, short_calls = _paired_score(
                    candidate, parent, history[-h_short:], horizon, gate["kappa_hist"]
                )
            else:
                u_short = median_short = mad_short = float("nan")
                short_calls = 0

        support_ratio = float(np.mean(rec_values <= -gate["delta_imp"]))
        conflict = bool(
            family in {"V3_SP_AdaptiveDual", "V4_SP_AdaptiveDual_NoRealign"}
            and u_rec <= -gate["delta_imp"]
            and support_ratio >= support_min
            and u_long > gate["delta_ret"]
        )
        conflict_count = conflict_count + 1 if conflict else 0
        normal = bool(
            family != "D0_SP_RecentOnly"
            and u_rec <= -gate["delta_imp"]
            and u_long <= gate["delta_ret"]
        )
        recovery = bool(
            family in {"V3_SP_AdaptiveDual", "V4_SP_AdaptiveDual_NoRealign"}
            and conflict_count >= r_persist
            and u_rec <= -gate["delta_imp"]
            and support_ratio >= support_min
            and u_short <= gate["delta_ret"]
        )
        if family == "D0_SP_RecentOnly":
            commit = bool(u_rec <= -gate["delta_imp"])
            branch = "recent_only" if commit else "retain"
        else:
            commit = normal or recovery
            branch = "normal" if normal else "recovery" if recovery else "retain"
        sync()
        verification_time = time.perf_counter() - verification_start

        # The deployable decision is sealed before any service label is reachable.
        timeline.record_decision(stage, commit)
        realigned = bool(recovery and realign)
        if realigned:
            history = history[-h_short:]
        history = (history + recent)[-long_capacity:]
        sync()
        maintenance_time = time.perf_counter() - maintenance_start
        peak = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else 0

        service = timeline.service(stage)
        service_hash = service.hash
        service_batch = tensors(windows(service, H=horizon))
        current_mse, _ = errors(parent, service_batch)
        candidate_mse, _ = errors(candidate, service_batch)
        deployed = candidate if commit else parent
        deployed_mse, deployed_mae = errors(deployed, service_batch)
        delta = float(candidate_mse.mean() - current_mse.mean())
        assert service.hash == service_hash
        if not commit:
            assert state_hash(deployed) == before

        memory = sum(block.x.nbytes + block.y.nbytes + block.ts.nbytes for block in history)
        rows.append(
            {
                "run_id": run_id,
                "structure_id": structure["structure_id"],
                "variant_family": family,
                "center_id": center,
                "center_type": timeline.meta["center_type"],
                "model_seed": seed,
                "horizon": horizon,
                "backbone": backbone,
                "stage": stage,
                "drift_regime": timeline.meta["drift_regime"],
                "attack_type": kind,
                "actual_attack_type": actual,
                "attack_alpha": alpha,
                "attack_location": location,
                "parent_state_hash": before,
                "candidate_state_hash": state_hash(candidate),
                "initial_state_hash": initial_hash,
                "commit": int(commit),
                "decision_branch": branch,
                "normal_branch_commit": int(normal and commit),
                "recovery_branch_commit": int(recovery and commit),
                "current_service_mse": float(current_mse.mean()),
                "candidate_service_mse": float(candidate_mse.mean()),
                "deployed_service_mse": float(deployed_mse.mean()),
                "deployed_service_mae": float(deployed_mae.mean()),
                "delta_srv": delta,
                "harmful_commit": int(commit and delta > 0),
                "downside": float(commit) * max(0.0, delta),
                "beneficial_candidate": int(delta < 0),
                "U_rec": u_rec,
                "U_hist_long": u_long,
                "U_hist_short": u_short,
                "median_rec": median_rec,
                "mad_rec": mad_rec,
                "median_hist_long": median_long,
                "mad_hist_long": mad_long,
                "median_hist_short": median_short,
                "mad_hist_short": mad_short,
                "support_ratio": support_ratio,
                "conflict_trigger": int(conflict),
                "conflict_persistence_length": conflict_count,
                "history_realign": int(realigned),
                "h_short": h_short if family.startswith("V3_") or family.startswith("V4_") else 0,
                "support_min": support_min if family.startswith("V3_") or family.startswith("V4_") else 0.0,
                "r_persist": r_persist if family.startswith("V3_") or family.startswith("V4_") else 0,
                "history_capacity": long_capacity,
                "candidate_update_time_s": update_time,
                "verification_time_s": verification_time,
                "maintenance_time_s": maintenance_time,
                "peak_gpu_memory_bytes": peak,
                "persistent_buffer_bytes": memory,
                "number_of_forward_passes": 50 + rec_calls + long_calls + short_calls,
                "number_of_backward_updates": 50,
                "update_start_ts": int(update.ts[0]),
                "update_end_ts": int(update.ts[-1]),
                "verification_start_ts": int(verification.ts[0]),
                "verification_end_ts": int(verification.ts[-1]),
                "decision_ts": int(verification.ts[-1]),
                "service_start_ts": int(service.ts[0]),
                "service_end_ts": int(service.ts[-1]),
                "attack_mask_hash": digest(update_mask, verification_mask, update_source, verification_source),
                "update_data_hash": update.hash,
                "verification_data_hash": verification.hash,
                "service_data_hash": service_hash,
                "kappa_rec": gate["kappa_rec"],
                "kappa_hist": gate["kappa_hist"],
                "delta_imp": gate["delta_imp"],
                "delta_ret": gate["delta_ret"],
            }
        )
        parent = deployed
    return rows
