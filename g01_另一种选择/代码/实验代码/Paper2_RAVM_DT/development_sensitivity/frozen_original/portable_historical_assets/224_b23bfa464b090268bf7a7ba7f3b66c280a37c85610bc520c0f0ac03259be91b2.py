"""Identity-bound external B-1 wall-clock watchdog."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from b1_process_identity import current_identity, identity_is_live, process_snapshot


def install_ctrl_c_immunity() -> dict:
    """Make the independent watchdog ignore console Ctrl+C on Windows.

    This is intentionally limited to the watchdog process.  It does not alter
    the runner's model, data, decisions, resource limits, or timeout action.
    """
    if os.name != "nt":
        return {"platform": os.name, "ctrl_c_handler": "NOT_APPLICABLE"}
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.SetConsoleCtrlHandler(None, True):
        raise OSError(ctypes.get_last_error(), "SetConsoleCtrlHandler failed")
    return {"platform": "nt", "ctrl_c_handler": "IGNORE_REGISTERED"}


def canonical(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def durable_json(path: Path, obj: dict, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as f:
            f.write(canonical(obj)); f.flush(); os.fsync(f.fileno())
    else:
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with tmp.open("xb") as f:
            f.write(canonical(obj)); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)


def event(path: Path | None, phase: str, **fields) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"epoch_ns": time.time_ns(), "phase": phase, **fields}, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush(); os.fsync(f.fileno())


def load_lease(path: Path) -> dict:
    lease = json.loads(path.read_text(encoding="utf-8"))
    required = ("child_pid", "child_identity", "watchdog_pid", "watchdog_identity", "deadline_epoch_ns")
    if lease.get("schema") != "RAVM_DT_B1_WALLCLOCK_LEASE_v2" or any(k not in lease for k in required):
        raise ValueError("watchdog lease identity/schema mismatch")
    if not isinstance(lease["child_pid"], int) or not isinstance(lease["deadline_epoch_ns"], int):
        raise ValueError("watchdog lease type mismatch")
    return lease


def supervise(lease_path: Path, receipt_path: Path, event_log: Path | None, poll_seconds: float = .05) -> dict:
    until = time.monotonic() + 20
    while not lease_path.is_file():
        if time.monotonic() >= until:
            raise TimeoutError("lease absent for watchdog")
        time.sleep(poll_seconds)
    lease = load_lease(lease_path)
    if lease["watchdog_pid"] != os.getpid() or lease["watchdog_identity"] != current_identity():
        raise PermissionError("watchdog lease identity mismatch")
    event(event_log, "lease_verified", lease=str(lease_path), child=process_snapshot(lease["child_pid"]))
    while time.time_ns() < lease["deadline_epoch_ns"]:
        alive, snapshot = identity_is_live(lease["child_pid"], lease["child_identity"])
        if not alive:
            status = "CHILD_EXITED_BEFORE_WALLCLOCK_DEADLINE" if snapshot.get("state") == "exited" else "CHILD_IDENTITY_UNAVAILABLE_OR_MISMATCH"
            receipt = {"schema": "RAVM_DT_B1_WALLCLOCK_RECEIPT_v2", "status": status, "lease": str(lease_path),
                       "child_pid": lease["child_pid"], "child_identity": lease["child_identity"],
                       "deadline_epoch_ns": lease["deadline_epoch_ns"], "observed_epoch_ns": time.time_ns(), "snapshot": snapshot}
            durable_json(receipt_path, receipt, exclusive=True); event(event_log, "child_not_live", receipt=receipt); return receipt
        time.sleep(poll_seconds)
    alive, snapshot = identity_is_live(lease["child_pid"], lease["child_identity"])
    if not alive:
        receipt = {"schema": "RAVM_DT_B1_WALLCLOCK_RECEIPT_v2", "status": "CHILD_IDENTITY_UNAVAILABLE_OR_MISMATCH", "lease": str(lease_path),
                   "child_pid": lease["child_pid"], "child_identity": lease["child_identity"], "observed_epoch_ns": time.time_ns(), "snapshot": snapshot}
        durable_json(receipt_path, receipt, exclusive=True); event(event_log, "deadline_no_kill_identity_invalid", receipt=receipt); return receipt
    completed = subprocess.run(["taskkill", "/PID", str(lease["child_pid"]), "/T", "/F"], text=True, capture_output=True, check=False)
    time.sleep(.1)
    post = process_snapshot(lease["child_pid"])
    receipt = {"schema": "RAVM_DT_B1_WALLCLOCK_RECEIPT_v2", "status": "WALLCLOCK_HARD_STOP_ENFORCED", "lease": str(lease_path),
               "child_pid": lease["child_pid"], "child_identity": lease["child_identity"], "deadline_epoch_ns": lease["deadline_epoch_ns"],
               "observed_epoch_ns": time.time_ns(), "taskkill_returncode": completed.returncode,
               "taskkill_stdout": completed.stdout[-1000:], "taskkill_stderr": completed.stderr[-1000:], "post_termination_snapshot": post}
    durable_json(receipt_path, receipt, exclusive=True); event(event_log, "deadline_taskkill", receipt=receipt); return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=.05)
    args = parser.parse_args()
    if not 0 < args.poll_seconds <= 5:
        raise ValueError("poll seconds outside safe range")
    try:
        event(args.event_log, "ctrl_c_immunity", **install_ctrl_c_immunity())
        print(json.dumps(supervise(args.lease, args.receipt, args.event_log, args.poll_seconds), ensure_ascii=False), flush=True)
        return 0
    except BaseException:
        event(args.event_log, "watchdog_exception", traceback=traceback.format_exc())
        traceback.print_exc()
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
