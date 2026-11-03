"""Author-gated B-1 launcher; a failed runner can never become package success."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocol" / "B1_RUNNER_CONSOLE_HARDENING_v5.lock"
DETACHED = getattr(subprocess, "DETACHED_PROCESS", 8) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512)


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def rows(root: Path) -> int:
    runs = root / "science_v5" / "runs"
    return sum(1 for p in runs.glob("*/*/*/results/stage_*.json")) if runs.exists() else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument("--diagnostic-mode", choices=("preledger-stop", "forced-exception", "forced-exit", "abrupt-exit-sim", "hold"))
    parser.add_argument("--diagnostic-timeout-seconds", type=float, default=0.0)
    args = parser.parse_args()
    logs = ROOT / "logs" / "process_hardening_orchestrator"
    state_path = logs / f"state_{time.time_ns()}.json"
    state = {"schema": "RAVM_DT_B1_LOCAL_ORCHESTRATION_v2", "started_utc": datetime.now(timezone.utc).isoformat(),
             "lock_path": str(LOCK), "lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
             "diagnostic_mode": args.diagnostic_mode, "actual_stage_rows": rows(ROOT), "phase": "PRELAUNCH",
             "science_exit_code": None, "package_exit_code": None, "audit_status": "NOT_RUN", "return_package_status": "NOT_STARTED"}
    atomic_json(state_path, state)
    command = [sys.executable, str(ROOT / "code" / "b1_science_supervisor.py"), "--approval-file", str(args.approval_file.resolve())]
    if args.diagnostic_mode:
        command += ["--diagnostic-mode", args.diagnostic_mode, "--diagnostic-timeout-seconds", str(args.diagnostic_timeout_seconds)]
    out, err = logs / f"supervisor_{state_path.stem}.stdout.log", logs / f"supervisor_{state_path.stem}.stderr.log"
    with out.open("ab") as fo, err.open("ab") as fe:
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe, creationflags=DETACHED)
    state.update({"phase": "SUPERVISOR_RUNNING", "supervisor_pid": proc.pid, "supervisor_stdout": str(out), "supervisor_stderr": str(err)})
    atomic_json(state_path, state)
    rc = proc.wait()
    state["science_exit_code"] = rc
    state["actual_stage_rows"] = rows(ROOT)
    state["ended_utc"] = datetime.now(timezone.utc).isoformat()
    if args.diagnostic_mode:
        state["phase"] = "DIAGNOSTIC_PRELEDGER_PASS" if rc == 0 else "ENGINEERING_BLOCKED_NO_SCIENCE"
        state["return_package_status"] = "NOT_APPLICABLE_DIAGNOSTIC_ONLY"
    elif rc != 0 or state["actual_stage_rows"] != 3168:
        state["phase"] = "ENGINEERING_BLOCKED_NO_SCIENCE"
        state["return_package_status"] = "NOT_BUILT_BECAUSE_SCIENCE_EXECUTION_FAILED"
    else:
        state["phase"] = "SCIENCE_COMPLETED_AWAITING_AUDIT"
        state["return_package_status"] = "NOT_BUILT_AWAITING_INDEPENDENT_AUDIT"
    atomic_json(state_path, state)
    print(json.dumps({"state": str(state_path), "phase": state["phase"], "science_exit_code": rc}, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
