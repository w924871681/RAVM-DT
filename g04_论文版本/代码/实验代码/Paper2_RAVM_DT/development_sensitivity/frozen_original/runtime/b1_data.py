"""B-1 proposed development assets and causal workload provider.

No Service values are passed to a controller before a durable decision seal.
This file is isolated from, and does not modify, any earlier experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from core.protocol import ROOT, REGIMES, Timeline, attack, digest, save_json, spans, windows
from core.models import DEVICE, Forecast, adapt, state_hash, tensors, sync
from vendor.main_cfg import MainCfg
from vendor.sim import simulate_centers


CENTERS = (46000, 46001, 46002, 46003)
REGIME_BY_CENTER = dict(zip(CENTERS, REGIMES))
STREAMS = ("clean", "mixed", "coherent_sticky")
SEED = 7311
ATTACK_SEED = 9417
ALPHA = 0.10
STICKY_MAGNITUDE = 0.20
STICKY_ACTIVE_STAGES = (1, 24)
SOURCE_SHA256 = "f73b2cad3da6f0ad52933f3b28178398be69826b6d3ca72bf8f00c9634aa3910"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _logit(a):
    a = np.clip(a, 1e-5, 1 - 1e-5)
    return np.log(a / (1 - a))


def _sigmoid(a):
    return 1 / (1 + np.exp(-a))


def generate_new_development_inputs() -> list[dict]:
    """Create raw external sequences, never score a model or expose Service."""
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    targets = [data_dir / f"center_{c}.npz" for c in CENTERS]
    if any(p.exists() for p in targets):
        raise FileExistsError("B-1 raw identity already exists; refuse overwrite or partial generation")
    cfg = MainCfg()
    cfg.sim.seed = SEED
    cfg.sim.n_history = 17000
    for kind in "ABC":
        getattr(cfg.sim, f"type_{kind}").drift_prob = 0.
    raw = simulate_centers(cfg, len(CENTERS), seed_offset=CENTERS[0] * 17,
                           start_center_id=CENTERS[0])
    rows = []
    for center, source in zip(CENTERS, raw):
        regime = REGIME_BY_CENTER[center]
        t = np.arange(len(source.y))
        shift = np.zeros(len(t))
        onset = 1400 + 8 * 640
        if regime == "abrupt":
            shift[t >= onset] = .8
        elif regime == "gradual":
            shift = .8 * np.clip((t - onset) / (8 * 640), 0, 1)
        elif regime == "recurring":
            shift = .8 * (((t >= 1400 + 6 * 640) & (t < 1400 + 12 * 640)) |
                          (t >= 1400 + 18 * 640))
        y = _sigmoid(_logit(source.y) + shift).astype("float32")
        x = np.zeros_like(source.X)
        for k, gain in enumerate(source.metadata["metric_gains"]):
            delay = source.metadata["metric_delays"][k] // source.dt
            dshift = (np.concatenate([np.repeat(shift[0], delay), shift[:len(shift) - delay]])
                      if delay else shift)
            x[:, k] = (_sigmoid(_logit(source.X[:, k]) + gain * dshift) * source.M[:, k]).astype("float32")
        x = np.concatenate([x, source.M], 1)
        if len(y) < spans(24)[2][1] or x.shape[1] != 24:
            raise ValueError("Insufficient 24-stage data or feature identity mismatch")
        meta = {"center_id": center, "center_type": source.center_type,
                "drift_regime": regime, "pool": "B1_DEV", "dt": source.dt,
                "data_seed": SEED, "clean_hash": digest(x, y, source.timestamps),
                "base_generator": source.metadata}
        path = data_dir / f"center_{center}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, x=x, y=y, ts=source.timestamps,
                                meta=json.dumps(meta))
        rows.append({"center_id": center, "regime_audit_only": regime,
                     "relative_path": f"data/{path.name}", "bytes": path.stat().st_size,
                     "sha256": sha256(path), "raw_array_digest": digest(x, y, source.timestamps),
                     "n_samples": len(y), "features": x.shape[1],
                     "first_ts": int(source.timestamps[0]),
                     "last_ts": int(source.timestamps[-1])})
    return rows


def sticky_spec(center: int) -> dict:
    if center not in CENTERS:
        raise ValueError("unlocked center")
    h = hashlib.sha256(f"B1-STICKY:{ATTACK_SEED}:{center}".encode()).digest()
    return {"center_id": center, "channel": int.from_bytes(h[:4], "little") % 12,
            "direction": 1 if h[4] % 2 == 0 else -1,
            "magnitude_normalized": STICKY_MAGNITUDE,
            "alpha_nominal": ALPHA, "stages_inclusive": STICKY_ACTIVE_STAGES,
            "mask_seed": ATTACK_SEED, "roles": ["update", "verification"],
            "positions_per_segment": "floor(alpha*n) uniform without replacement; independent center/stage/role seed",
            "target": "observed y and one metric x channel; observation mask channels untouched"}


def coherent_sticky_attack(seg, center: int, stage: int):
    if seg.role not in ("update", "verification") or not 1 <= stage <= 24:
        raise ValueError("unauthorized corruption role/stage")
    spec = sticky_spec(center)
    out = seg.copy()
    mask = np.zeros(len(seg.y), dtype=bool)
    if STICKY_ACTIVE_STAGES[0] <= stage <= STICKY_ACTIVE_STAGES[1]:
        role_code = 0 if seg.role == "update" else 1
        rng = np.random.default_rng(np.random.SeedSequence(
            [ATTACK_SEED, center, stage, role_code, 51]))
        ids = rng.choice(len(mask), int(np.floor(len(mask) * ALPHA + 1e-9)), replace=False)
        mask[ids] = True
        j = spec["channel"]
        sign = spec["direction"]
        observed = out.x[:, j]
        presence = out.x[:, 12 + j]
        observed[mask] = np.clip(observed[mask] + sign * STICKY_MAGNITUDE * presence[mask], 0, 1) * presence[mask]
        out.y[mask] = np.clip(out.y[mask] + sign * STICKY_MAGNITUDE, 0, 1)
    return out, mask


class Provider:
    """One independently advancing strategy trajectory on a frozen B-1 center/stream."""

    def __init__(self, center: int, stream: str, seed: int = SEED):
        if center not in CENTERS or stream not in STREAMS or seed != SEED:
            raise ValueError("B-1 center, stream or seed outside proposed matrix")
        self.center, self.stream, self.seed, self.stages = center, stream, seed, 24
        self.raw_path = ROOT / "data" / f"center_{center}.npz"
        self.timeline = Timeline(center)
        if self.timeline.meta.get("pool") != "B1_DEV":
            raise ValueError("new center has wrong data role")
        self._prepared: dict[int, dict] = {}
        self._sealed: dict[int, str] = {}
        self._serviced: set[int] = set()

    def initial(self):
        src = ROOT / "checkpoints" / f"source_GRU_H1_{self.seed}.pt"
        if sha256(src) != SOURCE_SHA256:
            raise ValueError("source checkpoint changed")
        path = ROOT / "checkpoints" / f"B1_M0_GRU_H1_{self.seed}_{self.center}.pt"
        sidecar = path.with_suffix(".json")
        model = Forecast().to(DEVICE)
        if path.exists() or sidecar.exists():
            if not path.is_file() or not sidecar.is_file():
                raise ValueError("partial M0; forensic review needed")
            info = json.loads(sidecar.read_text(encoding="utf-8"))
            if info.get("file_sha256") != sha256(path):
                raise ValueError("M0 bytes differ from sidecar")
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True), strict=True)
            if state_hash(model) != info.get("state_hash"):
                raise ValueError("M0 state differs from sidecar")
        else:
            model.load_state_dict(torch.load(src, map_location=DEVICE, weights_only=True), strict=True)
            warm = self.timeline.warmup()
            model = adapt(model, tensors(windows(warm)), steps=50,
                          seed=self.seed + self.center)
            sync()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
            torch.save(model.state_dict(), tmp)
            if path.exists():
                tmp.unlink(missing_ok=True)
                raise FileExistsError("concurrent M0 write")
            os.replace(tmp, path)
            info = {"state_hash": state_hash(model), "file_sha256": sha256(path),
                    "source_sha256": SOURCE_SHA256, "warmup_hash": warm.hash,
                    "warmup_steps": 50, "warmup_seed": self.seed + self.center,
                    "warmup_last_ts": int(warm.ts[-1]),
                    "optimizer_state_saved": False}
            save_json(sidecar, info)
        return model

    def initial_history(self):
        return self.timeline.initial_history()

    def xy(self, segment):
        return tensors(windows(segment))

    def seed_for_adaptation(self, stage):
        return self.seed + self.center + stage * 100

    def maintenance(self, stage):
        u, v = self.timeline.maintenance(stage)
        if self.stream == "clean":
            cu, cv = u, v
            um = np.zeros(len(u.y), bool)
            vm = np.zeros(len(v.y), bool)
            actual = "none"
        elif self.stream == "mixed":
            cu, um, _, actual = attack(u, self.center, stage, "mixed", ALPHA, "both", ATTACK_SEED)
            cv, vm, _, _ = attack(v, self.center, stage, "mixed", ALPHA, "both", ATTACK_SEED)
        else:
            cu, um = coherent_sticky_attack(u, self.center, stage)
            cv, vm = coherent_sticky_attack(v, self.center, stage)
            actual = "coherent_sticky"
        self._prepared[stage] = {"actual_attack_type": actual,
                                 "mask_hash": digest(um, vm),
                                 "update_changed_y": int(np.count_nonzero(cu.y != u.y)),
                                 "verification_changed_y": int(np.count_nonzero(cv.y != v.y))}
        return cu, cv

    def register_durable_seal(self, stage: int, seal_path: Path, seal_sha256: str):
        if stage in self._sealed or not seal_path.is_file() or sha256(seal_path) != seal_sha256:
            raise ValueError("durable decision seal absent or different")
        self._sealed[stage] = seal_sha256

    def service(self, stage: int):
        if stage not in self._sealed or stage in self._serviced:
            raise PermissionError("Service requires prior durable unique seal")
        self.timeline.record_decision(stage, False)
        segment = self.timeline.service(stage)
        self._serviced.add(stage)
        return segment

    def offline_evidence(self, stage: int):
        if stage not in self._serviced:
            raise PermissionError("attack evidence is offline only")
        return dict(self._prepared[stage], audit_only_after_service=True)

    def restore_completed_prefix(self, stages: int):
        """Reopen prior observations, never their Service, solely to rebuild history."""
        recent = []
        for stage in range(1, stages + 1):
            _, v = self.maintenance(stage)
            from core.engine import evidence_blocks
            recent.extend(evidence_blocks(v))
            self.timeline.record_decision(stage, False)
        return recent
