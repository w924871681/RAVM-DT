"""Independent lightweight progress reporter for an already running pilot.

Writes one snapshot immediately and then at most every 900 seconds by default.
DOES NOT launch the experiment, control the model, access Service, or fabricate ETA.
Runner must atomically update progress_state.json after real stage/trajectory seals.

Usage:
  python -u progress_monitor_15min.py OUTPUT_DIR --interval 900
State schema (runner created):
  run_id, status (RUNNING|DONE|STOPPED|FAILED), phase, phase_started_unix,
  trajectories_done, trajectories_total, stage_rows_done, stage_rows_total,
  active_unit, last_stage_commit_unix, gpu_mem_mb?, gpu_name?,
  auto_gate_decision?, error?, calibration_state?

On restart, monitor only appends a timestamped monitoring entry; it does not alter the experiment.
"""
import argparse
import datetime as dt
import json
import math
import os
import tempfile
import time
from pathlib import Path

TERMINAL = {"DONE", "STOPPED", "FAILED", "BLOCKED"}


def utc_stamp(now):
    return dt.datetime.fromtimestamp(now, dt.timezone.utc).isoformat(timespec="seconds")


def show_dur(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "暂无法估计（有效完工样本不足）"
    seconds = max(0, int(round(seconds)))
    d, s = divmod(seconds, 86400)
    h, s = divmod(s, 3600)
    m = int(round(s/60))
    if m == 60:
        m = 0; h += 1
    if h == 24:
        d += 1; h = 0
    return f"{d}天{h}小时{m}分钟" if d else f"{h}小时{m}分钟"


def as_nonnegative_int(value):
    try:
        v = int(value)
        return v if v >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def create_snapshot(state, now):
    phase = str(state.get("phase", "UNKNOWN"))
    started = state.get("phase_started_unix")
    done = as_nonnegative_int(state.get("trajectories_done"))
    total = as_nonnegative_int(state.get("trajectories_total"))
    sd = as_nonnegative_int(state.get("stage_rows_done"))
    st = as_nonnegative_int(state.get("stage_rows_total"))
    ok = done is not None and total is not None and total >= done and total > 0
    frac = done / total if ok else None
    elapsed = float(now - started) if isinstance(started, (float, int)) and 0 < now - started < 5*365*86400 else None
    # A trajectory is a complete 24-stage sequence. Do not project while insufficient samples.
    remain = (elapsed/done * (total-done)) if ok and done >= 3 and elapsed is not None else None
    last = state.get("last_stage_commit_unix")
    stale_sec = int(now-last) if isinstance(last,(int,float)) and now >= last else None
    stalled = state.get("status") == "RUNNING" and stale_sec is not None and stale_sec >= 1800
    snapshot = {
        "snapshot_at_utc": utc_stamp(now), "run_id": state.get("run_id"),
        "phase": phase, "status": state.get("status", "UNKNOWN"),
        "trajectories_done":done, "trajectories_total":total,
        "stage_rows_done":sd, "stage_rows_total":st,
        "phase_progress_percent":round(frac*100,2) if frac is not None else None,
        "elapsed_phase_seconds":round(elapsed,1) if elapsed is not None else None,
        "eta_phase_seconds":round(remain,1) if remain is not None else None,
        "eta_valid_for_expanded_phase":False,
        "stale_seconds":stale_sec, "potentially_stalled":stalled,
        "active_unit":state.get("active_unit", ""),
        "gpu_mem_mb":state.get("gpu_mem_mb"), "gpu_name":state.get("gpu_name"),
        "auto_gate_decision":state.get("auto_gate_decision"),
        "calibration_state":state.get("calibration_state"),
        "error":state.get("error"),
        "future_phases_eta": "未知；未用本阶段耗时冒充未来阶段准确ETA"
    }
    return snapshot


def atomic_write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def format_text(z):
    frac = f"{z['phase_progress_percent']:.2f}%" if z['phase_progress_percent'] is not None else "未知"
    t = f"{z['trajectories_done']}/{z['trajectories_total']}" if z['trajectories_total'] is not None else "未知"
    s = f"{z['stage_rows_done']}/{z['stage_rows_total']}" if z['stage_rows_total'] is not None else "未知"
    elapsed=show_dur(z['elapsed_phase_seconds'])
    eta=show_dur(z['eta_phase_seconds'])
    warn=("\n- **警告**：30分钟无已记录阶段提交；需检查本地Runner进程，不能据此断言程序已死。" if z['potentially_stalled'] else "")
    return (f"# RAVM-D3c 15分钟进度快照\n\n"
            f"- 采样时间（UTC）：{z['snapshot_at_utc']}\n"
            f"- 运行ID：`{z['run_id']}`\n"
            f"- 状态/阶段：`{z['status']}` / `{z['phase']}`\n"
            f"- 当前任务：{z['active_unit']}\n"
            f"- **完成完整24阶段轨迹：{t}（{frac}）**\n"
            f"- **已封存阶段记录：{s}**\n"
            f"- 本阶段实测已耗时：{elapsed}\n"
            f"- **本阶段预计还需：{eta}**（至少完成3条独立轨迹后按实测平均速度估算；非承诺）\n"
            f"- 后续阶段预计时间：暂无法从当前阶段推断，开始执行后独立估计\n"
            f"- D3c自动扩展裁决：{z['auto_gate_decision'] or '未到决策点'}\n"
            f"- GPU：{z['gpu_name'] or '未记录'} / 显存 {z['gpu_mem_mb'] or '未记录'} MB\n"
            f"- 最后阶段提交距今：{z['stale_seconds'] if z['stale_seconds'] is not None else '未知'}秒\n"
            f"- 错误：{z['error'] or '无已报告错误'}"
            f"{warn}\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--interval",type=int,default=900)
    ap.add_argument("--once",action="store_true")
    args=ap.parse_args()
    if args.interval < 60:
        ap.error("interval must be at least 60 seconds")
    d=Path(args.directory)
    state_path=d/"progress_state.json"
    d.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            s=json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(s,dict):
                raise ValueError("expected JSON object")
        except (OSError,ValueError,json.JSONDecodeError) as e:
            s={"status":"WAITING_FOR_ATOMIC_PROGRESS_STATE", "phase":"NO_STATE", "error":str(e)}
        z=create_snapshot(s,time.time())
        with (d/"PROGRESS_HISTORY.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(z,ensure_ascii=False)+"\n")
        atomic_write(d/"LATEST_PROGRESS_ZH.md",format_text(z))
        print(format_text(z),flush=True)
        if args.once or s.get("status") in TERMINAL:
            break
        time.sleep(args.interval)

if __name__=="__main__":main()
