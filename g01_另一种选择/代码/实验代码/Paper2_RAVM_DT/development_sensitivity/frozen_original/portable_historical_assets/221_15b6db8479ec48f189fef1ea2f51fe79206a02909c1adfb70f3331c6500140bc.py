"""Author-gated B-1 full-lifecycle executor; preflight import never opens Service.

Each center/stream/method follows its own true deployed parent and pool.  The
science entry point refuses to run without a separate author budget approval
bound to the immutable pre-Service lock.  This file is implemented in B-1
preflight but no formal B-1 stage is executed by that preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from b1_data import CENTERS, SEED, SOURCE_SHA256, STREAMS, Provider, sha256
from b1_policy import (ADAPT, METHODS, POOL_METHODS, RESTORE, RETAIN, Choice,
                       Pool, PreDecisionEvidence, StateEntry, choose)
from b1_store import Journal, ModelVault, canonical
from cmc_original_gate_adapter import OriginalCMCGate
from core.engine import evidence_blocks
from core.models import Batch, DEVICE, adapt, state_hash, sync
from core.protocol import ROOT, windows
import stage3_engine as original_score
from b1_process_identity import current_identity, identity_is_live


LOCK = ROOT / "protocol" / "B1_RUNNER_CONSOLE_HARDENING_v5.lock"
RAW_IDENTITY = ROOT / "data" / "B1_RAW_IDENTITY.json"
ATTACK_PLAN = ROOT / "data" / "B1_ATTACK_PLAN.json"
SCIENCE_ROOT = ROOT / "science_v5"
RUNS_ROOT = SCIENCE_ROOT / "runs"
WEIGHTS_ROOT = SCIENCE_ROOT / "weights"
EXPECTED_ROWS = len(CENTERS) * len(STREAMS) * len(METHODS) * 24


def _load_lock():
    if not LOCK.is_file():
        raise PermissionError("pre-Service protocol lock absent")
    doc = json.loads(LOCK.read_text(encoding="utf-8"))
    if doc.get("status") != "FROZEN_PENDING_AUTHOR_BUDGET" or doc.get("expected_stage_rows") != EXPECTED_ROWS:
        raise ValueError("B-1 protocol identity differs")
    for rel, expected in doc["frozen_files_sha256"].items():
        p = ROOT / rel
        if not p.is_file() or sha256(p) != expected:
            raise ValueError(f"frozen file mismatch: {rel}")
    return doc, sha256(LOCK)


def authorize_science(approval_file: Path):
    lock, lock_sha = _load_lock()
    if not approval_file.is_file():
        raise PermissionError("separate author approval and resource cap absent")
    approval = json.loads(approval_file.read_text(encoding="utf-8"))
    required = {"stage": "B1", "approved": True, "pre_service_lock_sha256": lock_sha,
                "approved_scientific_stage_rows": EXPECTED_ROWS}
    if any(approval.get(k) != v for k, v in required.items()):
        raise PermissionError("approval does not bind frozen B-1 matrix and lock")
    frozen_budget = lock.get("resource_budget")
    if not isinstance(frozen_budget, dict):
        raise ValueError("pre-Service lock lacks frozen resource budget")
    if (not isinstance(approval.get("max_process_seconds"), (int, float))
            or approval["max_process_seconds"] <= 0
            or not isinstance(approval.get("max_new_bytes"), int)
            or approval["max_new_bytes"] <= 0):
        raise PermissionError("explicit positive author resource limits absent")
    if (approval["max_process_seconds"] != frozen_budget.get("max_process_seconds")
            or approval["max_new_bytes"] != frozen_budget.get("max_new_bytes")):
        raise PermissionError("approval resource caps differ from the frozen B-1 proposal")
    return lock, lock_sha, approval


def authorize_preledger_diagnostic(approval_file: Path):
    """Authorize only the QA path that exits before BudgetLedger/Provider."""
    lock, lock_sha = _load_lock()
    if not approval_file.is_file():
        raise PermissionError("diagnostic authorization absent")
    approval = json.loads(approval_file.read_text(encoding="utf-8"))
    required = {"stage": "B1", "approved": True, "authorization_scope": "DIAGNOSTIC_PRELEDGER_ONLY",
                "pre_service_lock_sha256": lock_sha, "approved_scientific_stage_rows": 0}
    if any(approval.get(k) != v for k, v in required.items()):
        raise PermissionError("diagnostic authorization does not bind the hardened lock")
    return lock, lock_sha, approval


def _process_alive(pid: int, expected_identity: str) -> bool:
    """Cross-check that the separately launched wall-clock watchdog remains alive."""
    return identity_is_live(pid, expected_identity)[0]


def verify_supervisor_lease(lease_path: Path, lock_sha: str, approval_file: Path,
                            resource_budget: dict) -> dict:
    """Require an external watchdog lease before any B-1 Service-capable run.

    The lease is written by ``b1_science_supervisor.py`` after it starts a
    detached watchdog.  This check is deliberately in the science runner so a
    direct ``--run-science`` call cannot silently bypass the 12-hour limit.
    """
    path = Path(lease_path)
    # The supervisor learns the child PID only after Popen.  Permit that small
    # bootstrap race, but never begin a stage or construct Provider/M0 first.
    deadline_wait = time.monotonic() + 10.
    last_error = "external wall-clock supervisor lease absent"
    while time.monotonic() < deadline_wait:
        if not path.is_file():
            time.sleep(.05)
            continue
        try:
            lease = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(.05)
            continue
        if lease.get("child_pid") != os.getpid():
            last_error = "external supervisor lease child PID mismatch"
            time.sleep(.05)
            continue
        break
    else:
        raise PermissionError(last_error)
    required = {
        "schema": "RAVM_DT_B1_WALLCLOCK_LEASE_v2",
        "pre_service_lock_sha256": lock_sha,
        "approval_sha256": sha256(Path(approval_file)),
        "child_pid": os.getpid(),
        "child_identity": current_identity(),
        "external_watchdog": "b1_wallclock_watchdog.py",
    }
    if any(lease.get(key) != value for key, value in required.items()):
        raise PermissionError("external supervisor lease identity mismatch")
    if not isinstance(lease.get("watchdog_identity"), str) or not isinstance(lease.get("child_identity"), str):
        raise PermissionError("external supervisor lease lacks process identities")
    started = lease.get("started_epoch_ns")
    deadline = lease.get("deadline_epoch_ns")
    wall_ns = int(resource_budget["max_wall_hours"]) * 3600 * 1_000_000_000
    diagnostic_lease = lease.get("mode") == "DIAGNOSTIC_PRELEDGER_ONLY"
    deadline_ok = (diagnostic_lease and 0 < deadline - started <= 20 * 1_000_000_000) or (not diagnostic_lease and deadline - started == wall_ns)
    if (not isinstance(started, int) or not isinstance(deadline, int)
            or not deadline_ok or deadline <= time.time_ns()):
        raise PermissionError("external wall-clock deadline differs from frozen budget")
    if not _process_alive(lease.get("watchdog_pid"), lease.get("watchdog_identity")):
        raise PermissionError("external wall-clock watchdog is not alive")
    return lease


def verify_task_identity(center: int, stream: str):
    raw = json.loads(RAW_IDENTITY.read_text(encoding="utf-8"))
    plan = json.loads(ATTACK_PLAN.read_text(encoding="utf-8"))
    if raw.get("role") != "NEW_B1_DEVELOPMENT_ONLY_NOT_FINAL" or raw.get("seed") != SEED:
        raise ValueError("raw role/seed mismatch")
    rows = [r for r in raw["rows"] if r["center_id"] == center]
    if len(rows) != 1 or stream not in plan["streams"]:
        raise ValueError("center or stream not frozen")
    path = ROOT / rows[0]["relative_path"]
    if path.stat().st_size != rows[0]["bytes"] or sha256(path) != rows[0]["sha256"]:
        raise ValueError("raw NPZ bytes changed")
    if sha256(ROOT / "checkpoints" / f"source_GRU_H1_{SEED}.pt") != SOURCE_SHA256:
        raise ValueError("source checkpoint changed")
    return rows[0]


def _scores(vault: ModelVault, pool: Pool, candidate, candidate_hash: str,
            recent_blocks: list, prior_blocks: list, method: str,
            cmc_gate: OriginalCMCGate | None):
    from core.models import tensors
    blocks = list(prior_blocks[-12:]) + recent_blocks
    pairs = [tensors(windows(b)) for b in blocks]
    counts = [len(pair[0]) for pair in pairs]
    batch = Batch(torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs]),
                  "verification")
    model_hashes = (set(pool.entries) if method in POOL_METHODS else {pool.current_hash}) | {candidate_hash}
    models = {h: vault.load(pool.entries[h]) for h in model_hashes if h in pool.entries}
    models[candidate_hash] = candidate
    losses = {}
    times = {}
    for h in sorted(model_hashes):
        started = time.perf_counter()
        _, per_window, _ = original_score.predictions_losses(models[h], batch)
        sync()
        losses[h] = per_window
        times[h] = time.perf_counter() - started
    offsets = np.cumsum([0] + counts)
    recent_first = len(counts) - len(recent_blocks)
    recent_mse = {h: tuple(float(losses[h][offsets[i]:offsets[i + 1]].mean())
                           for i in range(recent_first, len(counts)))
                  for h in model_hashes}
    parent_loss = losses[pool.current_hash]
    pair_scores = {h: original_score.score_arrays(parent_loss, losses[h], counts,
                                                   len(recent_blocks))[0]
                   for h in model_hashes if h != pool.current_hash}
    cmc = None
    if method == "CMC_RAVM_ORIGINAL":
        if cmc_gate is None:
            raise ValueError("authenticated original CMC gate absent")
        cmc = bool(cmc_gate.evaluate(candidate, models[pool.current_hash],
                                      recent_blocks, prior_blocks).commit)
    ev = PreDecisionEvidence(pool.last_stage + 1, pool.current_hash, candidate_hash,
                             recent_mse, pair_scores, cmc,
                             pool_scored=method in POOL_METHODS)
    return ev, times


def _service_mse(model, xy) -> float:
    return float(original_score.predictions_losses(model, xy)[1].mean())


def _ensemble_mse(vault: ModelVault, pool: Pool, xy) -> tuple[float, int]:
    x, y = xy
    outputs = []
    for entry in pool.entries.values():
        model = vault.load(entry)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(x[i:i + 256]) for i in range(0, len(x), 256)])
            outputs.append(pred)
    forecast = torch.stack(outputs).mean(0)
    mse = (forecast - y).square().mean(1)
    return float(mse.mean().item()), len(outputs)


def _evaluate_sealed_stage(provider: Provider, journal: Journal, vault: ModelVault,
                           stage: int, seal: dict, seal_sha: str, service,
                           *, recovered: bool = False):
    decision = seal["decision"]
    before = Pool.from_dict(decision["pool_before"])
    after = Pool.from_dict(decision["pool_after"])
    choice = Choice(**decision["choice"])
    if choice.selected_hash != after.current_hash or before.current_hash != decision["parent_state_hash"]:
        raise ValueError("sealed pool/action identity mismatch")
    if not int(decision["verification_last_ts"]) < int(service.ts[0]):
        raise ValueError("Service timestamp not after decision evidence")
    service_xy = provider.xy(service)
    start = time.perf_counter()
    if journal.task["method"] == "POOL_SIMPLE_ENSEMBLE":
        before_mse, before_models = _ensemble_mse(vault, before, service_xy)
        after_mse, after_models = _ensemble_mse(vault, after, service_xy)
        predictor = "equal_weight_pool"
    else:
        before_model = vault.load(before.entries[before.current_hash])
        after_model = vault.load(after.entries[after.current_hash])
        before_mse = _service_mse(before_model, service_xy)
        after_mse = _service_mse(after_model, service_xy)
        before_models = after_models = 1
        predictor = "single_state"
    sync()
    scoring_seconds = time.perf_counter() - start
    delta = after_mse - before_mse if choice.action != RETAIN else 0.
    if choice.action == RETAIN and not np.isclose(before_mse, after_mse, rtol=0, atol=1e-12):
        raise ValueError("retained predictor changed")
    original_commit_harm = 0
    original_commit_downside = 0.
    if choice.action == ADAPT:
        parent = vault.load(before.entries[before.current_hash])
        cand_entry = StateEntry(**decision["candidate_entry"])
        candidate = vault.load(cand_entry)
        candidate_delta = _service_mse(candidate, service_xy) - _service_mse(parent, service_xy)
        original_commit_harm = int(candidate_delta > 0)
        original_commit_downside = max(0., candidate_delta)
    result = {
        "center": journal.task["center"], "stream": journal.task["stream"],
        "method": journal.task["method"], "stage": stage,
        "action": choice.action, "reason": choice.reason,
        "parent_state_hash": before.current_hash, "deployed_state_hash": after.current_hash,
        "candidate_state_hash": decision["candidate_state_hash"],
        "predictor": predictor, "predictor_models_before": before_models,
        "predictor_models_after": after_models,
        "parent_service_mse": before_mse, "deployed_service_mse": after_mse,
        "delta_service_mse": delta,
        "harmful_action": int(choice.action != RETAIN and delta > 0),
        "downside_absolute": max(0., delta) if choice.action != RETAIN else 0.,
        "original_candidate_commit_harm": original_commit_harm,
        "original_candidate_commit_downside": original_commit_downside,
        "service_segment_hash": service.hash,
        "service_first_ts": int(service.ts[0]),
        "offline_attack_evidence": provider.offline_evidence(stage),
        "pool_after": after.to_dict(),
        "cost_seconds": {**decision["predecision_cost_seconds"],
                         "offline_service_scoring": scoring_seconds},
        "sealed_decision_recovered_without_redeciding": recovered,
    }
    journal.complete(stage, seal_sha, result)
    return result


def run_task(center: int, stream: str, method: str, lock_sha: str,
             budget_guard=None) -> int:
    raw_identity = verify_task_identity(center, stream)
    task = {"center": center, "stream": stream, "method": method, "seed": SEED,
            "raw_sha256": raw_identity["sha256"]}
    journal = Journal(RUNS_ROOT / str(center) / stream / method, task, lock_sha)
    vault = ModelVault(WEIGHTS_ROOT / str(center) / stream / method,
                       center, stream, SOURCE_SHA256)
    provider = Provider(center, stream)
    completed, pending = journal.inspect_prefix()
    if completed or pending is not None:
        # Resume from the durable stage-1 identity. Never silently retrain M0.
        first = json.loads(journal.seal_path(1).read_text(encoding="utf-8"))
        pool = Pool.from_dict(first["decision"]["pool_before"])
        pool.validate(1)
        vault.load(pool.entries[pool.m0_hash])
    else:
        initial = provider.initial()
        m0 = vault.put(initial, created_stage=0, first_deployed_stage=0,
                       available_stage=1, training_max_ts=int(provider.timeline.warmup().ts[-1]))
        pool = Pool.initial(m0)
    history = evidence_blocks(provider.initial_history())
    for stage in range(1, completed + 1):
        u, v = provider.maintenance(stage)
        seal = json.loads(journal.seal_path(stage).read_text(encoding="utf-8"))
        if seal["decision"]["update_hash"] != u.hash or seal["decision"]["verification_hash"] != v.hash:
            raise ValueError("resumed raw maintenance differs from seal")
        record = json.loads(journal.result_path(stage).read_text(encoding="utf-8"))["result"]
        pool = Pool.from_dict(record["pool_after"])
        pool.validate(stage + 1)
        history.extend(evidence_blocks(v))
        provider.timeline.record_decision(stage, False)
    if pending is not None:
        stage = completed + 1
        if budget_guard:
            budget_guard()
        u, v = provider.maintenance(stage)
        if pending["decision"]["update_hash"] != u.hash or pending["decision"]["verification_hash"] != v.hash:
            raise ValueError("pending stage observation differs from sealed identity")
        if pending["decision"]["pool_before"] != pool.to_dict():
            raise ValueError("pending stage parent pool differs")
        seal_path = journal.seal_path(stage)
        seal_sha = sha256(seal_path)
        provider.register_durable_seal(stage, seal_path, seal_sha)
        journal.open_service(stage, seal_sha)
        service = provider.service(stage)
        _evaluate_sealed_stage(provider, journal, vault, stage, pending, seal_sha,
                               service, recovered=True)
        pool = Pool.from_dict(pending["decision"]["pool_after"])
        history.extend(evidence_blocks(v))
        completed = stage
    cmc_gate = (OriginalCMCGate(ROOT / "code", ROOT / "protocol" / "frozen_v2_final_commit_config.yaml")
                if method == "CMC_RAVM_ORIGINAL" else None)
    for stage in range(completed + 1, 25):
        if budget_guard:
            budget_guard()
        started = time.perf_counter()
        update, verification = provider.maintenance(stage)
        recent = evidence_blocks(verification)
        if len(recent) != 6 or not history:
            raise ValueError("missing six recent or historical blocks")
        parent_entry = pool.entries[pool.current_hash]
        start = time.perf_counter()
        parent = vault.load(parent_entry)
        parent_load_s = time.perf_counter() - start
        before_hash = state_hash(parent)
        start = time.perf_counter()
        if method == "Frozen":
            candidate = parent
            training_s = 0.
        else:
            candidate = adapt(parent, provider.xy(update), steps=50,
                              seed=provider.seed_for_adaptation(stage))
            sync()
            training_s = time.perf_counter() - start
        if state_hash(parent) != before_hash:
            raise ValueError("adaptation mutated deployed parent")
        candidate_hash = state_hash(candidate)
        candidate_entry = vault.put(candidate, created_stage=stage,
                                    first_deployed_stage=stage,
                                    available_stage=stage + 1,
                                    training_max_ts=int(update.ts[-1]))
        start = time.perf_counter()
        try:
            ev, forward_times = _scores(vault, pool, candidate, candidate_hash,
                                        recent, history, method, cmc_gate)
            choice = choose(method, pool, ev)
            force_error = None
        except (ValueError, FloatingPointError, PermissionError, KeyError, TypeError) as exc:
            force_error = f"FAIL_CLOSED:{type(exc).__name__}:{exc}"
            choice = Choice(RETAIN, pool.current_hash, force_error, False, (),
                            0.0, 0.0)
            forward_times = {}
        sync()
        verification_s = time.perf_counter() - start
        after = pool.transition(stage, choice.action, choice.selected_hash,
                                candidate_entry if choice.action == ADAPT else None)
        decision = {
            "action": choice.action, "choice": asdict(choice),
            "parent_state_hash": pool.current_hash,
            "candidate_state_hash": candidate_hash,
            "candidate_entry": asdict(candidate_entry),
            "pool_before": pool.to_dict(), "pool_after": after.to_dict(),
            "update_hash": update.hash, "verification_hash": verification.hash,
            "update_last_ts": int(update.ts[-1]),
            "verification_last_ts": int(verification.ts[-1]),
            "predecision_cost_seconds": {"parent_load": parent_load_s,
                                         "candidate_train": training_s,
                                         "verification_and_decision": verification_s,
                                         "per_model_verification_forward": forward_times},
            "forced_retain_error": force_error,
            "source_checkpoint_sha256": SOURCE_SHA256,
        }
        seal_path, seal_sha = journal.seal(stage, decision)
        provider.register_durable_seal(stage, seal_path, seal_sha)
        journal.open_service(stage, seal_sha)
        service = provider.service(stage)
        _evaluate_sealed_stage(provider, journal, vault, stage,
                               {"decision": decision}, seal_sha, service)
        pool = after
        history.extend(recent)
        if time.perf_counter() - started < 0:
            raise RuntimeError("nonmonotonic clock")
    return 24


def new_science_bytes(root: Path):
    """Count only outputs created by the author-approved science invocation."""
    output_root = root / "science_v5"
    total = sum(p.stat().st_size for base in (output_root,) if base.exists()
                for p in base.rglob("*") if p.is_file())
    ledger = root / "logs" / "B1_PROCESS_BUDGET.jsonl"
    return total + (ledger.stat().st_size if ledger.exists() else 0)


class BudgetLedger:
    """Conservatively charge an interrupted invocation until its next restart."""
    def __init__(self, path: Path, cap_seconds: float):
        self.path = path
        self.cap_seconds = cap_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        prior = 0.0
        opened = None
        for row in rows:
            if row["kind"] == "START":
                if opened is not None:
                    raise ValueError("unclosed budget invocation before a second START")
                opened = row["epoch_ns"]
            elif row["kind"] == "END":
                if opened is None or row["epoch_ns"] < opened:
                    raise ValueError("invalid budget ledger ordering")
                prior += (row["epoch_ns"] - opened) / 1e9
                opened = None
            else:
                raise ValueError("unknown budget ledger record")
        now = time.time_ns()
        if opened is not None:
            # Includes downtime, so an external kill cannot reset the cap.
            prior += max(0., (now - opened) / 1e9)
            self._append({"kind": "END", "epoch_ns": now,
                          "recovery_charge_includes_downtime": True})
        if prior >= cap_seconds:
            raise TimeoutError("cumulative approved B-1 process budget exhausted")
        self.prior = prior
        self.started_monotonic = time.monotonic()
        self._append({"kind": "START", "epoch_ns": now, "pid": os.getpid()})

    def _append(self, row: dict):
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(canonical(row).decode("utf-8") + "\n")
            f.flush()
            os.fsync(f.fileno())

    def charged_seconds(self):
        return self.prior + time.monotonic() - self.started_monotonic

    def close(self):
        self._append({"kind": "END", "epoch_ns": time.time_ns()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-science", action="store_true")
    p.add_argument("--approval-file", type=Path)
    p.add_argument("--supervisor-lease", type=Path)
    p.add_argument("--diagnostic-preledger", choices=("preledger-stop", "forced-exception", "forced-exit", "abrupt-exit-sim", "hold"))
    args = p.parse_args()
    if args.approval_file is None or (not args.run_science and args.diagnostic_preledger is None):
        raise PermissionError("B-1 science is author-gated; preflight never runs stages")
    diagnostic = args.diagnostic_preledger is not None
    lock, lock_sha, approval = (authorize_preledger_diagnostic(args.approval_file) if diagnostic
                                else authorize_science(args.approval_file))
    if args.supervisor_lease is None:
        raise PermissionError("B-1 science requires a live external wall-clock supervisor lease")
    lease = verify_supervisor_lease(args.supervisor_lease, lock_sha, args.approval_file,
                                    ({"max_wall_hours": 0} if diagnostic else lock["resource_budget"]))
    if diagnostic:
        mode = args.diagnostic_preledger
        def forbidden(name):
            def trap(*_args, **_kwargs):
                raise AssertionError(f"PRELEDGER_FORBIDDEN_CALL:{name}")
            return trap
        Provider.service = forbidden("Provider.service")
        globals()["adapt"] = forbidden("candidate_adapt")
        globals()["run_task"] = forbidden("formal_task_entry")
        record = ROOT / "logs" / "process_hardening" / Path(args.supervisor_lease).parent.name / "runner.preledger.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"status": "PRELEDGER_GATE_PASSED", "mode": mode,
                                      "service_calls": 0, "m0_created": False,
                                      "official_budget_ledger_created": False,
                                      "active_forbidden_traps": ["Provider.service", "candidate_adapt", "formal_task_entry"]}, sort_keys=True), encoding="utf-8")
        if mode == "forced-exception":
            raise RuntimeError("DIAGNOSTIC_FORCED_RUNNER_EXCEPTION_SENTINEL")
        if mode == "forced-exit":
            raise SystemExit(1)
        if mode == "abrupt-exit-sim":
            os._exit(139)
        if mode == "hold":
            while True:
                time.sleep(.1)
        print(json.dumps({"status": "DIAGNOSTIC_PRELEDGER_STOP_PASS"}, ensure_ascii=False), flush=True)
        return
    ledger = BudgetLedger(ROOT / "logs" / "B1_PROCESS_BUDGET.jsonl",
                          approval["max_process_seconds"])

    def budget_guard():
        if not _process_alive(lease["watchdog_pid"], lease["watchdog_identity"]):
            raise RuntimeError("external wall-clock watchdog disappeared; refuse next B-1 stage")
        if time.time_ns() >= lease["deadline_epoch_ns"]:
            raise TimeoutError("external B-1 wall-clock deadline reached")
        if ledger.charged_seconds() >= approval["max_process_seconds"]:
            raise TimeoutError("author process-time budget reached")
        if new_science_bytes(ROOT) >= approval["max_new_bytes"]:
            raise OSError("author new-output storage budget reached")
    try:
        for center in CENTERS:
            for stream in STREAMS:
                for method in METHODS:
                    budget_guard()
                    run_task(center, stream, method, lock_sha, budget_guard)
    finally:
        ledger.close()
    print(json.dumps({"status": "ALL_TASKS_COMPLETED_PENDING_INDEPENDENT_AUDIT",
                      "logical_stage_rows_expected": EXPECTED_ROWS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
