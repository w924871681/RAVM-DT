"""B-1 process supervisor with explicit Windows child evidence.

Normal operation retains the frozen scientific runner and its 12-hour lease.
Diagnostic modes stop before Provider, Service, M0, the official budget ledger,
or a lifecycle task is reachable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import b1_runner
from b1_process_identity import current_identity, process_snapshot

ROOT = Path(__file__).resolve().parents[1]
GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
# Both Service-capable children must be insulated from close events issued to
# the interactive launcher.  The supervisor retains OS handles and receipts,
# while the watchdog still owns the hard deadline and may taskkill the runner.
DETACHED_CHILD = GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def event(path: Path, phase: str, **fields) -> None:
    row = {"epoch_ns": time.time_ns(), "phase": phase, **fields}
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush(); os.fsync(f.fileno())


def _child_env() -> dict:
    env = dict(os.environ)
    env.update({"PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"})
    return env


def _start(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path,
           *, detached: bool = False):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    out, err = stdout_path.open("ab"), stderr_path.open("ab")
    try:
        flags = DETACHED_CHILD if detached else GROUP
        return subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                env=_child_env(), creationflags=flags)
    finally:
        out.close(); err.close()


def _watchdog_receipt(path: Path, timeout: float = 15.0) -> dict | None:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(.05)
    return None


def launch(approval_file: Path, diagnostic_mode: str | None = None, diagnostic_timeout_seconds: float = 0.0) -> int:
    if diagnostic_mode:
        lock, lock_sha, _ = b1_runner.authorize_preledger_diagnostic(approval_file)
        if diagnostic_mode not in {"preledger-stop", "forced-exception", "forced-exit", "abrupt-exit-sim", "hold"}:
            raise ValueError("unsupported diagnostic mode")
        wall_seconds = float(diagnostic_timeout_seconds) if diagnostic_mode == "hold" else 20.0
        if wall_seconds <= 0 or wall_seconds > 20:
            raise ValueError("diagnostic wall deadline must be in (0, 20] seconds")
    else:
        lock, lock_sha, _ = b1_runner.authorize_science(approval_file)
        wall_seconds = float(lock["resource_budget"]["max_wall_hours"]) * 3600
    start = time.time_ns()
    run_id = f"B1_{start}_{os.urandom(8).hex()}"
    root = ROOT / "logs" / "process_hardening" / run_id
    lease, receipt = root / "lease.json", root / "watchdog_receipt.json"
    events = root / "supervisor.events.jsonl"
    deadline = start + int(wall_seconds * 1_000_000_000)
    watchdog_cmd = [sys.executable, "-u", "-X", "faulthandler", str(ROOT / "code" / "b1_wallclock_watchdog.py"),
                    "--lease", str(lease), "--receipt", str(receipt), "--event-log", str(root / "watchdog.events.jsonl")]
    watchdog = _start(watchdog_cmd, ROOT, root / "watchdog.stdout.log", root / "watchdog.stderr.log",
                      detached=True)
    child_cmd = [sys.executable, "-u", "-X", "faulthandler", str(ROOT / "code" / "b1_runner.py"),
                 "--approval-file", str(Path(approval_file).resolve()), "--supervisor-lease", str(lease)]
    child_cmd += ["--diagnostic-preledger", diagnostic_mode] if diagnostic_mode else ["--run-science"]
    child = _start(child_cmd, ROOT, root / "runner.stdout.log", root / "runner.stderr.log", detached=True)
    child_snapshot, watchdog_snapshot = process_snapshot(child.pid), process_snapshot(watchdog.pid)
    if child_snapshot.get("state") != "alive" or watchdog_snapshot.get("state") != "alive":
        raise RuntimeError(f"child/watchdog failed before lease: {child_snapshot}; {watchdog_snapshot}")
    lease_doc = {
        "schema": "RAVM_DT_B1_WALLCLOCK_LEASE_v2", "mode": "DIAGNOSTIC_PRELEDGER_ONLY" if diagnostic_mode else "SCIENCE",
        "run_id": run_id, "pre_service_lock_sha256": lock_sha, "approval_sha256": sha256(Path(approval_file)),
        "child_pid": child.pid, "child_identity": child_snapshot["identity"],
        "watchdog_pid": watchdog.pid, "watchdog_identity": watchdog_snapshot["identity"],
        "supervisor_pid": os.getpid(), "supervisor_identity": current_identity(),
        "started_epoch_ns": start, "deadline_epoch_ns": deadline, "external_watchdog": "b1_wallclock_watchdog.py",
        "lease_token": os.urandom(32).hex(), "runner_command": child_cmd, "watchdog_command": watchdog_cmd,
        "watchdog_launch_mode": "DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP",
        "runner_launch_mode": "DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP",
    }
    atomic_json(lease, lease_doc)
    event(events, "lease_durable", child=child_snapshot, watchdog=watchdog_snapshot, cwd=str(ROOT),
          diagnostic_mode=diagnostic_mode, python=sys.executable)
    rc = child.wait()
    event(events, "runner_exited", returncode=rc, child_snapshot=process_snapshot(child.pid))
    watchdog_receipt = _watchdog_receipt(receipt)
    if watchdog.poll() is None:
        watchdog.wait(timeout=20)
    if watchdog_receipt is None:
        watchdog_receipt = {"schema": "RAVM_DT_B1_WALLCLOCK_RECEIPT_v2", "status": "WATCHDOG_RECEIPT_MISSING",
                            "child_pid": child.pid, "observed_epoch_ns": time.time_ns()}
        atomic_json(receipt, watchdog_receipt)
    normal = rc == 0 and watchdog_receipt.get("status") == "CHILD_EXITED_BEFORE_WALLCLOCK_DEADLINE"
    result = {"schema": "RAVM_DT_B1_SUPERVISOR_RECEIPT_v2", "run_id": run_id, "mode": lease_doc["mode"],
              "runner_exit_code": rc, "watchdog_exit_code": watchdog.returncode, "watchdog_receipt": str(receipt),
              "watchdog_status": watchdog_receipt.get("status"),
              "status": "RUNNER_NORMAL_EXIT" if normal else ("WALLCLOCK_HARD_STOP_ENFORCED" if watchdog_receipt.get("status") == "WALLCLOCK_HARD_STOP_ENFORCED" else "RUNNER_NONZERO_OR_SUPERVISION_FAILURE"),
              "runner_stdout": str(root / "runner.stdout.log"), "runner_stderr": str(root / "runner.stderr.log"),
              "watchdog_stdout": str(root / "watchdog.stdout.log"), "watchdog_stderr": str(root / "watchdog.stderr.log"),
              "events": str(events), "ended_epoch_ns": time.time_ns()}
    atomic_json(root / "supervisor_receipt.json", result)
    event(events, "supervisor_receipt_durable", result=result)
    return 0 if normal else (124 if watchdog_receipt.get("status") == "WALLCLOCK_HARD_STOP_ENFORCED" else 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument("--diagnostic-mode", choices=("preledger-stop", "forced-exception", "forced-exit", "abrupt-exit-sim", "hold"))
    parser.add_argument("--diagnostic-timeout-seconds", type=float, default=0.0)
    args = parser.parse_args()
    try:
        return launch(args.approval_file, args.diagnostic_mode, args.diagnostic_timeout_seconds)
    except BaseException:
        root = ROOT / "logs" / "process_hardening" / f"supervisor_failure_{time.time_ns()}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "supervisor.traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
