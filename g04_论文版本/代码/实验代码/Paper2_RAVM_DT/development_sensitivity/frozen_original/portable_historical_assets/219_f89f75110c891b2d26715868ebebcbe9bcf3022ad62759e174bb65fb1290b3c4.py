"""Fail-closed Windows process identity helpers for B-1 supervision.

This module deliberately never uses ``os.kill(pid, 0)``.  A numerical PID is
not an identity: on Windows an exited process object can remain open and a PID
can subsequently be reused.  The lease therefore records the process creation
FILETIME and independent observers inspect a synchronizable process handle.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value


def _filetime_int(value) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def _windows_snapshot(pid: int) -> dict:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return {"pid": pid, "state": "unavailable", "winerror": ctypes.get_last_error()}
    try:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                        ctypes.byref(kernel), ctypes.byref(user)):
            return {"pid": pid, "state": "unavailable", "winerror": ctypes.get_last_error()}
        waited = kernel32.WaitForSingleObject(handle, 0)
        if waited == WAIT_TIMEOUT:
            state = "alive"
        elif waited == WAIT_OBJECT_0:
            state = "exited"
        else:
            return {"pid": pid, "state": "unavailable", "winerror": ctypes.get_last_error(),
                    "wait_result": int(waited)}
        return {"pid": pid, "state": state, "creation_filetime": _filetime_int(creation),
                "identity": f"win:{pid}:{_filetime_int(creation)}"}
    finally:
        kernel32.CloseHandle(handle)


def process_snapshot(pid: int) -> dict:
    """Return a non-destructive, auditable process state.

    Unknown/unavailable results are intentionally not upgraded to alive.
    """
    if not isinstance(pid, int) or pid <= 0:
        return {"pid": pid, "state": "invalid"}
    if os.name == "nt":
        return _windows_snapshot(pid)
    proc = f"/proc/{pid}"
    if not os.path.exists(proc):
        return {"pid": pid, "state": "unavailable"}
    try:
        stat = open(f"{proc}/stat", "r", encoding="utf-8").read().split()
        created = stat[21]
        return {"pid": pid, "state": "alive", "creation_filetime": created,
                "identity": f"proc:{pid}:{created}"}
    except OSError:
        return {"pid": pid, "state": "unavailable"}


def current_identity() -> str:
    snap = process_snapshot(os.getpid())
    if snap.get("state") != "alive" or not isinstance(snap.get("identity"), str):
        raise RuntimeError(f"cannot establish current process identity: {snap}")
    return snap["identity"]


def identity_is_live(pid: int, expected_identity: str) -> tuple[bool, dict]:
    snap = process_snapshot(pid)
    return (snap.get("state") == "alive" and snap.get("identity") == expected_identity), snap
