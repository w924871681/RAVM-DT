"""Independent post-run verifier for the frozen B-1 result protocol.

The auditor does not import the B-1 controller or reuse its choice functions.
It checks sealed on-disk records, denominators and aggregates after a complete
run.  ``--allow-toy`` is solely for the pre-science QA fixture and can never
produce a scientific conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str):
    raise ValueError(message)


def _event_order(rows: list[dict], stage: int, seal_sha: str):
    indices = {kind: [i for i, row in enumerate(rows)
                      if row["stage"] == stage and row["kind"] == kind
                      and row.get("detail", {}).get("seal_sha256") == seal_sha]
               for kind in ("DECISION_DURABLY_SEALED", "SERVICE_ACCESS_OPEN", "STAGE_COMPLETED")}
    if any(len(indices[kind]) != 1 for kind in indices):
        fail(f"missing/duplicate event ordering evidence at stage {stage}")
    if not (indices["DECISION_DURABLY_SEALED"][0] < indices["SERVICE_ACCESS_OPEN"][0]
            < indices["STAGE_COMPLETED"][0]):
        fail(f"seal→Service→completion order invalid at stage {stage}")


def _cost(value) -> float:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, dict):
        return sum(_cost(v) for v in value.values())
    return 0.0


def _check_pool(pool: dict, *, center: int, stream: str, source_sha: str,
                stage: int, label: str) -> None:
    """Validate serialized state identities without importing controller code."""
    entries = pool.get("entries")
    if not isinstance(entries, dict) or len(entries) > 4:
        fail(f"K=4 pool capacity/type invalid at {label}")
    if pool.get("center") != center or pool.get("stream") != stream:
        fail(f"pool task identity differs at {label}")
    if pool.get("source_sha256") != source_sha:
        fail(f"pool source checkpoint identity differs at {label}")
    if pool.get("m0_hash") not in entries or pool.get("current_hash") not in entries:
        fail(f"M0/current missing from pool at {label}")
    pred = pool.get("predecessor_hash")
    provisional = pool.get("provisional_since_stage")
    if (pred is None) != (provisional is None):
        fail(f"predecessor/provisional pairing differs at {label}")
    if pred is not None and pred not in entries:
        fail(f"protected predecessor missing at {label}")
    for state_hash, entry in entries.items():
        if state_hash != entry.get("state_hash"):
            fail(f"pool key/state hash mismatch at {label}")
        if (entry.get("center"), entry.get("stream"), entry.get("source_sha256")) != (center, stream, source_sha):
            fail(f"model source/task lineage differs at {label}")
        for key in ("state_hash", "file_sha256", "source_sha256"):
            value = entry.get(key, "")
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                fail(f"invalid {key} at {label}")
        if int(entry.get("file_bytes", 0)) <= 0:
            fail(f"invalid model file size at {label}")
        rel = Path(entry.get("relative_path", ""))
        if not entry.get("relative_path") or rel.is_absolute() or ".." in rel.parts:
            fail(f"unsafe model weight path at {label}")
        if int(entry.get("available_stage", stage + 1)) > stage:
            # after-pools use the next-stage availability boundary; a seal's
            # before-pool only contains models available at the current stage.
            if not (label.endswith("pool_after") and int(entry["available_stage"]) == stage + 1):
                fail(f"future model state in pool at {label}")
    m0 = entries[pool["m0_hash"]]
    if (m0.get("created_stage"), m0.get("first_deployed_stage"), m0.get("available_stage")) != (0, 0, 1):
        fail(f"M0 lineage/availability differs at {label}")


def _verify_outcome(result: dict, action: str, label: str) -> None:
    """Recompute primary risk fields from sealed parent/deployed Service MSEs."""
    parent_mse = float(result["parent_service_mse"])
    deployed_mse = float(result["deployed_service_mse"])
    recorded_delta = float(result["delta_service_mse"])
    recorded_downside = float(result["downside_absolute"])
    if (not all(math.isfinite(v) and v >= 0 for v in (parent_mse, deployed_mse))
            or not math.isfinite(recorded_delta) or not math.isfinite(recorded_downside)
            or recorded_downside < 0):
        fail(f"nonfinite/negative service outcome: {label}")
    expected_delta = deployed_mse - parent_mse if action != "RETAIN" else 0.
    expected_harm = int(action != "RETAIN" and expected_delta > 0.)
    expected_downside = max(0., expected_delta) if action != "RETAIN" else 0.
    if (not math.isclose(recorded_delta, expected_delta, rel_tol=0., abs_tol=1e-12)
            or int(result["harmful_action"]) != expected_harm
            or not math.isclose(recorded_downside, expected_downside, rel_tol=0., abs_tol=1e-12)):
        fail(f"harm/downside accounting differs from service outcomes: {label}")
    candidate_harm = int(result.get("original_candidate_commit_harm", 0))
    candidate_downside = float(result.get("original_candidate_commit_downside", 0.))
    if candidate_harm not in (0, 1) or not math.isfinite(candidate_downside) or candidate_downside < 0:
        fail(f"invalid separate candidate-only harm fields: {label}")
    if action != "ADAPT_AND_COMMIT" and (candidate_harm != 0 or candidate_downside != 0.):
        fail(f"candidate-only harm recorded for an uncommitted candidate: {label}")


def audit(run_root: Path, lock_path: Path, *, allow_toy: bool = False,
          verify_weights: bool = False) -> dict:
    run_root, lock_path = Path(run_root), Path(lock_path)
    output_root = run_root / "science_v5"
    lock = load(lock_path)
    if lock.get("schema") != "RAVM_DT_B1_PRE_SERVICE_LOCK_v1":
        fail("unsupported B-1 lock schema")
    lock_sha = sha256(lock_path)
    project_root = lock_path.parent.parent
    for rel, digest in lock.get("frozen_files_sha256", {}).items():
        frozen = project_root / rel
        if not frozen.is_file() or sha256(frozen) != digest:
            fail(f"frozen source/data identity mismatch: {rel}")
    matrix = lock["matrix"]
    centers, streams, methods, stages = matrix["centers"], matrix["streams"], matrix["methods"], matrix["stages"]
    expected = {(c, s, m, t) for c in centers for s in streams for m in methods for t in range(1, stages + 1)}
    raw = load(run_root / "data" / "B1_RAW_IDENTITY.json")
    if sha256(run_root / "data" / "B1_RAW_IDENTITY.json") != lock["frozen_files_sha256"].get("data/B1_RAW_IDENTITY.json"):
        fail("run raw identity file differs from the frozen data identity")
    raw_by_center = {int(row["center_id"]): row["sha256"] for row in raw["rows"]}
    if set(raw_by_center) != set(centers):
        fail("raw center identity does not cover locked matrix")
    found, toy = set(), False
    by_method_stream = defaultdict(list)
    by_center_method_stream = defaultdict(list)
    costs = defaultdict(float)
    actions = {}
    weight_refs = []
    source_checkpoint_sha = lock["frozen_files_sha256"].get("checkpoints/source_GRU_H1_7311.pt")
    if not source_checkpoint_sha:
        fail("lock lacks immutable source checkpoint identity")
    for center in centers:
        for stream in streams:
            for method in methods:
                task_root = output_root / "runs" / str(center) / stream / method
                events = task_root / "events.jsonl"
                if not events.is_file():
                    fail(f"missing event log: {center}/{stream}/{method}")
                event_rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
                if any(event_rows[i]["event_sequence"] >= event_rows[i + 1]["event_sequence"]
                       or event_rows[i]["monotonic_ns"] > event_rows[i + 1]["monotonic_ns"]
                       for i in range(len(event_rows) - 1)):
                    fail("event sequence or monotonic time regression")
                previous_pool_after = None
                for stage in range(1, stages + 1):
                    seal_path = task_root / "seals" / f"stage_{stage:03d}.json"
                    result_path = task_root / "results" / f"stage_{stage:03d}.json"
                    if not seal_path.is_file() or not result_path.is_file():
                        fail(f"missing stage artifact: {center}/{stream}/{method}/{stage}")
                    seal, record = load(seal_path), load(result_path)
                    task = seal.get("task")
                    key = (center, stream, method, stage)
                    if task != {"center": center, "stream": stream, "method": method,
                                "seed": matrix["seed"], "raw_sha256": raw_by_center[center]}:
                        fail(f"task identity mismatch: {key}")
                    if seal.get("pre_service_lock_sha256") != lock_sha or seal.get("stage") != stage:
                        fail(f"seal lock/stage mismatch: {key}")
                    seal_sha = sha256(seal_path)
                    if record.get("decision_seal_sha256") != seal_sha or record.get("task") != task:
                        fail(f"result lacks matching immutable seal: {key}")
                    _event_order(event_rows, stage, seal_sha)
                    result = record.get("result", {})
                    action = result.get("action")
                    if action not in ("RETAIN", "ADAPT_AND_COMMIT", "RESTORE_ARCHIVED"):
                        fail(f"unknown deployed action: {key}")
                    _verify_outcome(result, action, str(key))
                    if not int(seal["decision"]["verification_last_ts"]) < int(result["service_first_ts"]):
                        fail(f"Service timestamp precedes sealed verification: {key}")
                    decision = seal.get("decision", {})
                    before, sealed_after = decision.get("pool_before"), decision.get("pool_after")
                    if not isinstance(before, dict) or not isinstance(sealed_after, dict):
                        fail(f"seal lacks complete before/after pool lineage: {key}")
                    _check_pool(before, center=center, stream=stream, source_sha=source_checkpoint_sha,
                                stage=stage, label=f"{key}/pool_before")
                    _check_pool(sealed_after, center=center, stream=stream, source_sha=source_checkpoint_sha,
                                stage=stage + 1, label=f"{key}/pool_after")
                    if previous_pool_after is not None and before != previous_pool_after:
                        fail(f"cross-stage pool recovery lineage discontinuity: {key}")
                    if sealed_after != result.get("pool_after"):
                        fail(f"sealed and completed pool state differ: {key}")
                    choice = decision.get("choice", {})
                    if (choice.get("action") != action
                            or choice.get("selected_hash") != result.get("deployed_state_hash")
                            or decision.get("parent_state_hash") != before.get("current_hash")
                            or result.get("parent_state_hash") != before.get("current_hash")):
                        fail(f"sealed choice/parent/deployed identity differs: {key}")
                    selected = choice.get("selected_hash")
                    if selected != sealed_after.get("current_hash"):
                        fail(f"selected state is not deployed current state: {key}")
                    candidate_entry = decision.get("candidate_entry", {})
                    if (candidate_entry.get("state_hash") != decision.get("candidate_state_hash")
                            or candidate_entry.get("center") != center
                            or candidate_entry.get("stream") != stream
                            or candidate_entry.get("source_sha256") != source_checkpoint_sha
                            or int(candidate_entry.get("created_stage", -1)) != stage
                            or int(candidate_entry.get("available_stage", -1)) != stage + 1):
                        fail(f"candidate model lineage differs: {key}")
                    if (len(candidate_entry.get("file_sha256", "")) != 64
                            or int(candidate_entry.get("file_bytes", 0)) <= 0
                            or candidate_entry.get("optimizer_state_saved") is not False):
                        fail(f"candidate content/optimizer-state identity differs: {key}")
                    candidate_rel = Path(candidate_entry.get("relative_path", ""))
                    if not candidate_entry.get("relative_path") or candidate_rel.is_absolute() or ".." in candidate_rel.parts:
                        fail(f"unsafe candidate weight path: {key}")
                    weight_refs.append((center, stream, method, candidate_entry))
                    if action == "RETAIN" and selected != before.get("current_hash"):
                        fail(f"RETAIN changed deployed state: {key}")
                    if action == "ADAPT_AND_COMMIT" and (selected != candidate_entry.get("state_hash")
                                                          or selected in before.get("entries", {})):
                        fail(f"ADAPT did not deploy the newly trained candidate: {key}")
                    if action == "RESTORE_ARCHIVED":
                        archive = before.get("entries", {}).get(selected)
                        if (archive is None or selected == before.get("current_hash")
                                or int(archive.get("first_deployed_stage", stage)) >= stage
                                or selected not in choice.get("archived_eligible", ())):
                            fail(f"RESTORE target is not a previously deployed eligible archive: {key}")
                    if sealed_after.get("m0_hash") != before.get("m0_hash"):
                        fail(f"M0 identity changed across transition: {key}")
                    previous_pool_after = sealed_after
                    if action == "RETAIN" and (float(result["delta_service_mse"]) != 0.
                                                or int(result["harmful_action"]) != 0):
                        fail(f"retain risk accounting differs: {key}")
                    pool = result["pool_after"]
                    if result.get("deployed_state_hash") != pool.get("current_hash"):
                        fail(f"deployed pool state differs: {key}")
                    for entry in pool["entries"].values():
                        weight_refs.append((center, stream, method, entry))
                    if task.get("toy_fixture") or result.get("toy_fixture"):
                        toy = True
                    found.add(key)
                    by_method_stream[(method, stream)].append(result)
                    by_center_method_stream[(center, method, stream)].append(result)
                    costs[(method, stream)] += _cost(result.get("cost_seconds", {}))
                    actions[(center, stream, method, stage)] = action
    if found != expected:
        fail(f"matrix coverage differs; missing={len(expected - found)} extra={len(found - expected)}")
    if toy and not allow_toy:
        fail("TOY fixture supplied without --allow-toy")
    if verify_weights and not toy:
        for center, stream, method, entry in weight_refs:
            weight = output_root / "weights" / str(center) / stream / method / entry["relative_path"]
            if not weight.is_file() or weight.stat().st_size != entry["file_bytes"] or sha256(weight) != entry["file_sha256"]:
                fail(f"weight lineage bytes differ: {weight}")
    risk = {}
    mse = {}
    for method in methods:
        for stream in streams:
            rows = by_method_stream[(method, stream)]
            if len(rows) != len(centers) * stages:
                fail(f"risk denominator differs: {method}/{stream}")
            harm = sum(int(r["harmful_action"]) for r in rows)
            downside = sum(float(r["downside_absolute"]) for r in rows) / len(rows)
            risk[(method, stream)] = {"denominator": len(rows), "harmful_actions": harm,
                                      "frequency": harm / len(rows), "downside": downside,
                                      "qualified": harm <= 4 and downside <= lock["risk_and_outcomes"]["downside_budget"]}
            center_means = []
            for center in centers:
                center_rows = by_center_method_stream[(center, method, stream)]
                center_means.append(sum(float(r["deployed_service_mse"]) for r in center_rows) / stages)
            mse[(method, stream)] = sum(center_means) / len(center_means)
    clean_mixed = {method: (mse[(method, "clean")] + mse[(method, "mixed")]) / 2 for method in methods}
    controls = ["POOL_GREEDY_RECENT", "POOL_DUAL_MEAN", "POOL_SIMPLE_ENSEMBLE"]
    qualified_controls = [method for method in controls if all(risk[(method, stream)]["qualified"] for stream in streams)]
    h1 = "H1_PROVISIONAL_REVERSIBLE"
    recurring = centers[3]
    restore_divergence = any(actions[(recurring, stream, h1, stage)] == "RESTORE_ARCHIVED"
                             and actions[(recurring, stream, "POOL_GREEDY_RECENT", stage)]
                             != "RESTORE_ARCHIVED"
                             for stream in streams for stage in range(1, stages + 1))
    best_control = min(qualified_controls, key=lambda name: clean_mixed[name]) if qualified_controls else None
    h1_budget = all(risk[(h1, stream)]["qualified"] for stream in streams)
    pilot = bool(not toy and h1_budget and restore_divergence and best_control is not None
                 and clean_mixed[h1] <= .99 * clean_mixed[best_control])
    return {
        "schema": "RAVM_DT_B1_INDEPENDENT_AUDIT_v1", "lock_sha256": lock_sha,
        "toy_fixture": toy, "matrix_rows": len(found), "risk": {f"{m}/{s}": v for (m, s), v in risk.items()},
        "center_equal_mse": {f"{m}/{s}": v for (m, s), v in mse.items()},
        "clean_mixed_equal_mse": clean_mixed,
        "cost_seconds": {f"{m}/{s}": v for (m, s), v in costs.items()},
        "same_pool_qualified_controls": qualified_controls, "best_same_pool_control": best_control,
        "h1_recurrent_restore_diverges_from_greedy": restore_divergence,
        "status": "TOY_AUDIT_PASS_NOT_SCIENCE" if toy else ("PILOT_PROMISING_ONLY" if pilot else "B1_SCIENCE_NO_GO_OR_DUPLICATE"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-toy", action="store_true")
    parser.add_argument("--verify-weights", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_root, args.lock, allow_toy=args.allow_toy, verify_weights=args.verify_weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "matrix_rows": result["matrix_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
