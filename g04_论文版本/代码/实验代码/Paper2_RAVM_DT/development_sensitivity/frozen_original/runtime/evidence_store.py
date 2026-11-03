"""Isolated, content-addressed evidence entities for Stage 3A.

Entities are never overwritten. An index contains actual bytes and file SHA-256,
not only abstract model/content hashes. All paths stay in the caller's run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


class EvidenceStore:
    def __init__(self, root: str | Path, run_key: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_key = str(run_key)
        self.index = self.root / 'entities.jsonl'
        self.seen: dict[str, dict] = {}
        self.new_bytes = 0
        self.new_entities = 0
        self.task_dir = self.root / 'runs' / object_digest({'run_key': run_key})
        self.task_dir.mkdir(parents=True, exist_ok=True)

    def append(self, name: str, record: dict) -> None:
        if '/' in name or '\\' in name or '..' in name:
            raise ValueError('JSONL basename required')
        with (self.task_dir / name).open('a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            f.flush()
            if name == 'decision_seals.jsonl':
                # The deployment decision must reach durable storage before
                # the provider is allowed to expose future service data.
                os.fsync(f.fileno())

    def _record(self, path: Path, content_id: str, kind: str) -> dict:
        key = str(path)
        if key in self.seen:
            return self.seen[key]
        record = dict(kind=kind, content_id=content_id,
                      path=str(path), relative_path=path.relative_to(self.root).as_posix(),
                      bytes=path.stat().st_size, sha256=file_sha(path))
        self.seen[key] = record
        with self.index.open('a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps({**record, 'first_indexed_run': self.run_key}, sort_keys=True) + '\n')
        return record

    def _path(self, kind: str, content_id: str, suffix: str) -> Path:
        if not kind.replace('_', '').isalnum() or len(content_id) != 64:
            raise ValueError('Invalid entity key')
        path = self.root / kind / content_id[:2] / (content_id + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def model(self, model, state_id: str) -> dict:
        path = self._path('weights', state_id, '.pt')
        if not path.exists():
            tmp = path.with_suffix(f'.{os.getpid()}.tmp')
            torch.save({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, tmp)
            if not path.exists():
                os.replace(tmp, path)
                self.new_bytes += path.stat().st_size
                self.new_entities += 1
            tmp.unlink(missing_ok=True)
        return self._record(path, state_id, 'weights')

    def arrays(self, kind: str, content_id: str, arrays: dict) -> dict:
        path = self._path(kind, content_id, '.npz')
        if not path.exists():
            tmp = path.with_suffix(f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, **arrays)
            if not path.exists():
                os.replace(tmp, path)
                self.new_bytes += path.stat().st_size
                self.new_entities += 1
            tmp.unlink(missing_ok=True)
        return self._record(path, content_id, kind)

    def json(self, kind: str, content_id: str, obj: dict) -> dict:
        path = self._path(kind, content_id, '.json')
        payload = json.dumps(obj, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f'Content-address collision: {path}')
        else:
            tmp = path.with_suffix(f'.{os.getpid()}.tmp')
            tmp.write_bytes(payload)
            if not path.exists():
                os.replace(tmp, path)
                self.new_bytes += path.stat().st_size
                self.new_entities += 1
            tmp.unlink(missing_ok=True)
        return self._record(path, content_id, kind)
