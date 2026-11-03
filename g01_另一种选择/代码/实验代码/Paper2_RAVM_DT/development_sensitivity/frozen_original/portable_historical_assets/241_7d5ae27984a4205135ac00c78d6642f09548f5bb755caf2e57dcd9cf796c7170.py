from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(ROOT_CODE))
from b1_process_identity import identity_is_live


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def alive(pid, identity) -> bool:
    if not isinstance(identity, str):
        return False
    return identity_is_live(pid, identity)[0]


def files_below(path: Path):
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file()]


def load_ledger(path: Path, now_ns: int) -> float:
    if not path.is_file():
        return 0.0
    total, opened = 0.0, None
    raw = path.read_bytes()
    complete = raw.rsplit(b"\n", 1)[0] + b"\n" if raw.endswith(b"\n") else raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
    for line in complete.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") == "START":
            if opened is not None:
                return total + max(0.0, (now_ns - opened) / 1e9)
            opened = int(row["epoch_ns"])
        elif row.get("kind") == "END" and opened is not None:
            total += max(0.0, (int(row["epoch_ns"]) - opened) / 1e9)
            opened = None
    if opened is not None:
        total += max(0.0, (now_ns - opened) / 1e9)
    return total


def snapshot(root: Path, pids_path: Path, started_epoch: float) -> dict:
    now = time.time()
    state = json.loads(pids_path.read_text(encoding="utf-8")) if pids_path.is_file() else {}
    results = files_below(root / "runs")
    result_files = [p for p in results if p.parent.name == "results" and p.name.startswith("stage_") and p.suffix == ".json"]
    seal_files = [p for p in results if p.parent.name == "seals" and p.name.startswith("stage_") and p.suffix == ".json"]
    task_dirs = [p for p in (root / "runs").glob("*/*/*") if p.is_dir()]
    complete_tasks = sum(1 for p in task_dirs if sum(1 for x in (p / "results").glob("stage_*.json")) == 24)
    latest = max(result_files, key=lambda p: p.stat().st_mtime_ns, default=None)
    lease_files = sorted((root / "logs" / "wallclock").glob("*/lease.json"), key=lambda p: p.stat().st_mtime_ns) if (root / "logs" / "wallclock").exists() else []
    lease = json.loads(lease_files[-1].read_text(encoding="utf-8")) if lease_files else {}
    now_ns = time.time_ns()
    ledger_seconds = load_ledger(root / "logs" / "B1_PROCESS_BUDGET.jsonl", now_ns)
    budget_bytes = (root / "logs" / "B1_PROCESS_BUDGET.jsonl").stat().st_size if (root / "logs" / "B1_PROCESS_BUDGET.jsonl").is_file() else 0
    science_paths = files_below(root / "runs") + files_below(root / "weights")
    science_paths += [p for p in (root / "checkpoints").glob("B1_M0_*") if p.is_file()]
    science_bytes = sum(p.stat().st_size for p in science_paths) + budget_bytes
    supervisor_pid = state.get("supervisor_pid")
    child_pid = lease.get("child_pid")
    watchdog_pid = lease.get("watchdog_pid")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "orchestrator_pid": state.get("orchestrator_pid"),
        "progress_sidecar_pid": os.getpid(),
        "supervisor_pid": supervisor_pid,
        "supervisor_alive": alive(supervisor_pid, state.get("supervisor_identity")),
        "science_child_pid": child_pid,
        "science_child_alive": alive(child_pid, lease.get("child_identity")),
        "watchdog_pid": watchdog_pid,
        "watchdog_alive": alive(watchdog_pid, lease.get("watchdog_identity")),
        "completed_result_files": len(result_files),
        "expected_stage_rows": 3168,
        "sealed_decision_files": len(seal_files),
        "completed_tasks_of_132": complete_tasks,
        "latest_completed_task_from_path_metadata": str(latest.relative_to(root)) if latest else None,
        "latest_result_mtime_utc": datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc).isoformat() if latest else None,
        "cumulative_charged_process_seconds": round(ledger_seconds, 3),
        "remaining_process_seconds": round(max(0.0, 14400.0 - ledger_seconds), 3),
        "new_science_bytes": science_bytes,
        "remaining_new_science_bytes": max(0, 4294967296 - science_bytes),
        "wall_elapsed_seconds": round(now - started_epoch, 3),
        "wall_budget_seconds": 43200,
        "non_sensitive_exception_summary": state.get("last_exception_summary"),
    }


def write_sample(root: Path, record: dict) -> None:
    log = root / "logs" / "B1_PROGRESS_15MIN.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    atomic_json(root / "logs" / "B1_PROGRESS_LATEST.json", record)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pids", type=Path, required=True)
    p.add_argument("--started-epoch", type=float, required=True)
    p.add_argument("--stop-file", type=Path, required=True)
    p.add_argument("--interval", type=int, default=900)
    args = p.parse_args()
    if args.interval != 900:
        raise ValueError("the authorized sidecar interval is fixed at 900 seconds")
    next_sample = time.monotonic()
    first = True
    while True:
        if args.stop_file.exists():
            final = snapshot(args.root, args.pids, args.started_epoch)
            final["sample_kind"] = "completion_or_abnormal_stop"
            write_sample(args.root, final)
            return 0
        if first or time.monotonic() >= next_sample:
            try:
                rec = snapshot(args.root, args.pids, args.started_epoch)
            except Exception as exc:
                rec = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "orchestrator_pid": None,
                    "progress_sidecar_pid": os.getpid(),
                    "supervisor_pid": None, "supervisor_alive": False,
                    "science_child_pid": None, "science_child_alive": False,
                    "watchdog_pid": None, "watchdog_alive": False,
                    "completed_result_files": None, "expected_stage_rows": 3168,
                    "sealed_decision_files": None, "completed_tasks_of_132": None,
                    "latest_completed_task_from_path_metadata": None,
                    "cumulative_charged_process_seconds": None,
                    "remaining_process_seconds": None, "new_science_bytes": None,
                    "remaining_new_science_bytes": None,
                    "wall_elapsed_seconds": round(time.time() - args.started_epoch, 3),
                    "wall_budget_seconds": 43200,
                    "non_sensitive_exception_summary": f"metadata sampling {type(exc).__name__}: {str(exc)[:240]}",
                }
            rec["sample_kind"] = "startup_or_interval"
            write_sample(args.root, rec)
            first = False
            next_sample = time.monotonic() + args.interval
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
