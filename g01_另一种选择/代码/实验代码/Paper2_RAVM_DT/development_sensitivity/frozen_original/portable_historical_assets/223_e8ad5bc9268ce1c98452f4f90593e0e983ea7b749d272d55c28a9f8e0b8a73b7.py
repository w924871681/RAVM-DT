"""Content-addressed B-1 weights and durable pre-Service decision journal."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from core.models import DEVICE, Forecast, state_hash
from b1_policy import StateEntry


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def exclusive_json(path: Path, obj: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(obj)
    with path.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(payload).hexdigest()


class ModelVault:
    def __init__(self, root: Path, center: int, stream: str, source_sha256: str):
        self.root = Path(root).resolve()
        self.center, self.stream, self.source_sha256 = center, stream, source_sha256
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, model, *, created_stage: int, first_deployed_stage: int,
            available_stage: int, training_max_ts: int) -> StateEntry:
        identity = state_hash(model)
        path = self.root / "states" / identity[:2] / f"{identity}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_name(f"{identity}.{os.getpid()}.tmp")
            with tmp.open("xb") as f:
                torch.save({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, f)
                f.flush()
                os.fsync(f.fileno())
            if path.exists():
                tmp.unlink(missing_ok=True)
            else:
                os.replace(tmp, path)
        stored = torch.load(path, map_location="cpu", weights_only=True)
        probe = Forecast().cpu()
        probe.load_state_dict(stored, strict=True)
        if state_hash(probe) != identity:
            raise ValueError("CAS state does not match filename/content")
        entry = StateEntry(identity, sha256(path), path.stat().st_size,
                           path.relative_to(self.root).as_posix(),
                           self.center, self.stream, self.source_sha256,
                           created_stage, first_deployed_stage, available_stage,
                           training_max_ts)
        entry.validate()
        return entry

    def load(self, entry: StateEntry):
        entry.validate()
        if (entry.center, entry.stream, entry.source_sha256) != (self.center, self.stream, self.source_sha256):
            raise ValueError("cross-task historical state")
        path = (self.root / entry.relative_path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("weight path outside vault or absent")
        if path.stat().st_size != entry.file_bytes or sha256(path) != entry.file_sha256:
            raise ValueError("weight bytes changed")
        model = Forecast().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True), strict=True)
        if state_hash(model) != entry.state_hash:
            raise ValueError("logical weight hash differs")
        return model


class Journal:
    def __init__(self, root: Path, task: dict, pre_service_lock_sha256: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.lock_sha = pre_service_lock_sha256
        self.events = self.root / "events.jsonl"

    def seal_path(self, stage: int) -> Path:
        return self.root / "seals" / f"stage_{stage:03d}.json"

    def result_path(self, stage: int) -> Path:
        return self.root / "results" / f"stage_{stage:03d}.json"

    def event(self, kind: str, stage: int, **detail) -> dict:
        rows = self.events.read_text(encoding="utf-8").splitlines() if self.events.exists() else []
        number = len(rows) + 1
        record = {"event_sequence": number, "kind": kind, "stage": stage,
                  "monotonic_ns": time.monotonic_ns(), "detail": detail}
        with self.events.open("a", encoding="utf-8", newline="\n") as f:
            f.write(canonical(record).decode("utf-8") + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def seal(self, stage: int, decision: dict) -> tuple[Path, str]:
        content = {"schema": "B1_PRE_SERVICE_DECISION_v1", "task": self.task,
                   "pre_service_lock_sha256": self.lock_sha, "stage": stage,
                   "decision": decision}
        path = self.seal_path(stage)
        identity = exclusive_json(path, content)
        self.event("DECISION_DURABLY_SEALED", stage, seal_sha256=identity)
        return path, identity

    def open_service(self, stage: int, seal_sha: str):
        path = self.seal_path(stage)
        if not path.is_file() or sha256(path) != seal_sha:
            raise PermissionError("Service before verified durable decision")
        return self.event("SERVICE_ACCESS_OPEN", stage, seal_sha256=seal_sha)

    def complete(self, stage: int, seal_sha: str, result: dict):
        if sha256(self.seal_path(stage)) != seal_sha:
            raise ValueError("result lacks matching pre-Service seal")
        record = {"schema": "B1_STAGE_RESULT_v1", "task": self.task,
                  "stage": stage, "decision_seal_sha256": seal_sha,
                  "result": result}
        identity = exclusive_json(self.result_path(stage), record)
        self.event("STAGE_COMPLETED", stage, result_sha256=identity, seal_sha256=seal_sha)
        return identity

    def inspect_prefix(self) -> tuple[int, dict | None]:
        """Return completed prefix and at most one sealed but incomplete stage.

        A pending stage must resume its *sealed* decision; the policy is never
        called again. This also detects holes or changed lock/task identities.
        """
        completed = 0
        pending = None
        for stage in range(1, 25):
            seal_path, result_path = self.seal_path(stage), self.result_path(stage)
            if result_path.exists() and not seal_path.exists():
                raise ValueError("result without seal")
            if not seal_path.exists():
                if any(self.seal_path(t).exists() or self.result_path(t).exists() for t in range(stage + 1, 25)):
                    raise ValueError("non-contiguous stage files")
                break
            sealed = json.loads(seal_path.read_text(encoding="utf-8"))
            if sealed.get("task") != self.task or sealed.get("pre_service_lock_sha256") != self.lock_sha or sealed.get("stage") != stage:
                raise ValueError("seal identity differs")
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("decision_seal_sha256") != sha256(seal_path) or result.get("task") != self.task or result.get("stage") != stage:
                    raise ValueError("result identity differs")
                completed = stage
            else:
                if any(self.seal_path(t).exists() or self.result_path(t).exists() for t in range(stage + 1, 25)):
                    raise ValueError("pending stage cannot have later artifacts")
                pending = sealed
                break
        return completed, pending
