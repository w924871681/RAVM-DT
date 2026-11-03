"""Stage 3A evidence-preserving shared-parent lifecycle executor.

The only changed workload decision input is a fixed nonnegative history margin.
Candidate training and scalar score arithmetic follow the audited gate_graph.
Providers expose maintenance before service, and service only after all decisions
are durably sealed. Regimes, attack masks and service outcomes never reach gates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from core.models import adapt, state_hash, Batch, DEVICE, sync
from core.engine import evidence_blocks
from core.protocol import digest, windows
from evidence_store import EvidenceStore, object_digest
from stage4r_gate import stage4r_decision_details
from stage4r_protocol import validate_fixed_stage4r_policies, verify_stage4r_lock

SCORE_KEYS = ('recent_mean', 'recent_median', 'history_mean', 'history_robust',
              'cumulative_mean', 'mwu_p')
METHODS = ('RecentMean', 'RecentMedian', 'DualMean', 'RAVM', 'CumulativeMean', 'MWUMean')
OPTIMIZER_SPEC = dict(optimizer='SGD', loss='MSE', steps=50, learning_rate=.01,
                      batch='min(64,n), replacement sampling', gradient_clip=1.)
SCORE_VERSION = 'audited-gate_graph-arithmetic-v1-krec0-khist0.5-bounded12'


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_arm(a: dict) -> dict:
    """Accept old graph dictionaries and reference-like dictionary fields."""
    out = dict(a)
    out['id'] = str(a.get('id', a.get('arm_id', '')))
    out['threshold'] = float(a.get('threshold', a.get('recent_margin', 0.)))
    out['scale'] = float(a.get('scale', a.get('recent_scale', 0.)))
    out['retention_threshold'] = float(a.get('retention_threshold', a.get('retention_tolerance', a.get('delta_ret', 0.))))
    out['retention_scale'] = float(a.get('retention_scale', 0.))
    out['fallback'] = bool(a.get('fallback', False))
    out['learning_rate'] = float(a.get('learning_rate', .01))
    if out['learning_rate'] not in (.01, .003, .001):
        raise ValueError('Candidate learning rate outside the frozen thermal grid')
    if not out['id'] or out.get('method') not in METHODS + ('AlwaysFT', 'Frozen'):
        raise ValueError('Missing arm identity or unknown method')
    for key in ('threshold', 'scale', 'retention_threshold', 'retention_scale'):
        if not np.isfinite(out[key]) or out[key] < 0:
            raise FloatingPointError(f'Invalid {key}')
    if out['method'] not in ('RAVM', 'DualMean') and out['retention_threshold'] != 0:
        raise ValueError('History tolerance cannot be applied to a gate without history')
    if 'stage4r_variant' in out:
        if out['method'] not in ('RAVM', 'DualMean') or out['stage4r_variant'] not in ('LEGACY', 'CAP_ONLY', 'TEMPORAL_GUARD'):
            raise ValueError('Stage4-R variant requires one of the two fixed history gates')
        if out['retention_threshold'] != 0:
            raise ValueError('Stage4-R preserves original zero history tolerance')
    return out


def decision(method: str, scores: dict, threshold: float, fallback: bool = False,
             retention_threshold: float = 0.) -> bool:
    """Online controller: scores only, no optional future/regime/mask fields."""
    a = normalize_arm(dict(id='controller', method=method, threshold=threshold,
                           retention_threshold=retention_threshold, fallback=fallback))
    if a['fallback'] or method == 'Frozen':
        return False
    if method == 'AlwaysFT':
        return True
    if set(scores) != set(SCORE_KEYS):
        raise ValueError('Controller accepts exactly the six verification scores')
    if not np.isfinite([scores[k] for k in SCORE_KEYS]).all():
        raise FloatingPointError('Nonfinite evidence; never fail open')
    if not 0 <= scores['mwu_p'] <= 1:
        raise ValueError('Invalid Mann-Whitney p value')
    key = {'RecentMean': 'recent_mean', 'RecentMedian': 'recent_median',
           'DualMean': 'recent_mean', 'RAVM': 'recent_median',
           'CumulativeMean': 'cumulative_mean', 'MWUMean': 'recent_mean'}[method]
    if scores[key] > -a['threshold']:
        return False
    if method == 'RAVM':
        return bool(scores['history_robust'] <= a['retention_threshold'])
    if method == 'DualMean':
        return bool(scores['history_mean'] <= a['retention_threshold'])
    if method == 'MWUMean':
        return bool(scores['mwu_p'] < .2)
    return True


def decision_details(a: dict, scores: dict) -> dict:
    commit = decision(a['method'], scores, a['threshold'], a['fallback'], a['retention_threshold'])
    if a['fallback']:
        return dict(commit=int(commit), recent_pass=None, history_pass=None, reject_reason='FROZEN_FALLBACK')
    if a['method'] in ('AlwaysFT', 'Frozen'):
        return dict(commit=int(commit), recent_pass=None, history_pass=None,
                    reject_reason='ALWAYS_UPDATE_ENDPOINT' if commit else 'FROZEN_ENDPOINT')
    key = {'RecentMean': 'recent_mean', 'RecentMedian': 'recent_median',
           'DualMean': 'recent_mean', 'RAVM': 'recent_median',
           'CumulativeMean': 'cumulative_mean', 'MWUMean': 'recent_mean'}[a['method']]
    rp = bool(scores[key] <= -a['threshold'])
    hp = None
    if a['method'] in ('RAVM', 'DualMean'):
        hk = 'history_robust' if a['method'] == 'RAVM' else 'history_mean'
        hp = bool(scores[hk] <= a['retention_threshold'])
    if not rp:
        reason = 'RECENT_FAIL' if hp is not False else 'RECENT_AND_HISTORY_FAIL'
    elif hp is False:
        reason = 'HISTORY_VETO'
    elif a['method'] == 'MWUMean' and not scores['mwu_p'] < .2:
        reason = 'MWU_FAIL'
    else:
        reason = 'ACCEPT'
    return dict(commit=int(commit), recent_pass=rp, history_pass=hp, reject_reason=reason)


def stage_metrics(old: float, new: float, commit: bool) -> dict:
    if not np.isfinite([old, new]).all() or min(old, new) < 0:
        raise FloatingPointError('Nonfinite/negative service loss')
    delta = float(new) - float(old)
    return dict(current_service_mse=float(old), candidate_service_mse=float(new), delta_srv=delta,
                deployed_service_mse=float(new if commit else old),
                harmful_commit=int(bool(commit) and delta > 0), downside=max(delta, 0.) * int(commit),
                beneficial_candidate=int(delta < 0))


def adapt_at_learning_rate(parent, xy, seed: int, learning_rate: float):
    """Thermal-only LR branch; .01 literally reuses the audited optimizer."""
    if learning_rate == .01:
        return adapt(parent, xy, steps=50, seed=seed)
    if learning_rate not in (.003, .001):
        raise ValueError('Unauthorized candidate learning rate')
    if getattr(xy, 'role', None) != 'update':
        raise AssertionError('Candidate adapter accepts only matured update role')
    model = copy.deepcopy(parent)
    model.train()
    x, y = xy
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        idx = torch.as_tensor(rng.integers(len(x), size=min(64, len(x))), device=DEVICE)
        opt.zero_grad(set_to_none=True)
        pred = model(x[idx])
        loss = torch.nn.functional.mse_loss(pred, y[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite candidate training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
    return model


@torch.no_grad()
def predictions_losses(model, xy):
    """Same 256-window inference batches and float32 errors as original errors()."""
    model.eval()
    x, y = xy
    if not len(x) or len(x) != len(y):
        raise ValueError('Empty or mismatched evidence batch')
    pred = torch.cat([model(x[i:i + 256]) for i in range(0, len(x), 256)])
    mse = (pred - y).square().mean(1).cpu().numpy()
    mae = (pred - y).abs().mean(1).cpu().numpy()
    arr = pred.cpu().numpy()
    if not np.isfinite(arr).all() or not np.isfinite(mse).all() or not np.isfinite(mae).all():
        raise FloatingPointError('Nonfinite predictions or losses')
    return arr, mse, mae


def score_arrays(old: np.ndarray, new: np.ndarray, counts: list[int], n_recent: int):
    """Original sample-weighted means, block medians and history-only .5 MAD."""
    if not counts or not all(counts) or sum(counts) != len(old) or len(new) != len(old):
        raise ValueError('Empty/mismatched verification blocks')
    if n_recent < 1 or len(counts) <= n_recent:
        raise ValueError('Recent and prior history evidence required')
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise FloatingPointError('Nonfinite verification loss')
    ds = new - old
    offset = np.cumsum([0] + counts)
    blockmeans = np.array([float(ds[offset[i]:offset[i + 1]].mean()) for i in range(len(counts))])
    start = int(offset[len(counts) - n_recent])
    hfirst = max(0, len(counts) - n_recent - 12)
    hstart = int(offset[hfirst])
    histmeans = blockmeans[hfirst:len(counts) - n_recent]
    recentmeans = blockmeans[-n_recent:]
    histmedian = float(np.median(histmeans))
    histmad = float(np.median(abs(histmeans - histmedian)))
    recentmedian = float(np.median(recentmeans))
    recentmad = float(np.median(abs(recentmeans - recentmedian)))
    scores = dict(recent_mean=float(ds[start:].mean()), recent_median=recentmedian,
                  history_mean=float(ds[hstart:start].mean()), history_robust=histmedian + .5 * histmad,
                  cumulative_mean=float(ds.mean()),
                  mwu_p=float(mannwhitneyu(new[start:], old[start:], alternative='less', method='asymptotic').pvalue))
    if not np.isfinite(list(scores.values())).all():
        raise FloatingPointError('Nonfinite verification score')
    detail = dict(block_means=blockmeans, counts=np.array(counts, dtype=np.int64), offsets=offset,
                  recent_median=recentmedian, recent_mad=recentmad, history_median=histmedian,
                  history_mad=histmad, kappa_recent=0., kappa_history=.5,
                  bounded_history_first_block=hfirst, recent_first_window=start,
                  bounded_history_first_window=hstart)
    return scores, detail


def score_models(parent, candidate, blocks, n_recent, xy):
    """Public compatibility helper; full evidence is recorded by run_graph."""
    batches = [xy(b) for b in blocks]
    batch = Batch(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches]), 'verification')
    _, old, _ = predictions_losses(parent, batch)
    _, new, _ = predictions_losses(candidate, batch)
    return score_arrays(old, new, [len(b[0]) for b in batches], n_recent)[0]


def _segment_record(provider, seg, xy, store: EvidenceStore | None):
    if not len(xy[0]):
        raise ValueError('Empty maintenance evidence')
    w = windows(seg)
    # Workload has no excluded labels. Providers with exclusions must supply the
    # original retained indices explicitly; positional guessing is forbidden.
    if hasattr(provider, 'window_metadata'):
        wm = provider.window_metadata(seg)
        origin = np.asarray(wm['origin_indices'], dtype=np.int64)
        target_ts = np.asarray(wm['target_timestamps'], dtype=np.int64)
    else:
        if len(w[0]) != len(xy[0]):
            raise ValueError('Provider must expose retained window metadata')
        origin = np.arange(len(w[0]), dtype=np.int64)
        target_ts = w[2]
    if len(origin) != len(xy[0]) or len(target_ts) != len(xy[0]):
        raise ValueError('Invalid retained window metadata')
    wx = xy[0].detach().cpu().numpy()
    wy = xy[1].detach().cpu().numpy()
    record = dict(segment_hash=seg.hash, role=seg.role, raw_count=len(seg.y),
                  first_ts=int(seg.ts[0]), last_ts=int(seg.ts[-1]),
                  valid_window_count=len(wx), window_hash=digest(wx, wy, target_ts, origin),
                  origin_indices_role='segment-relative raw origin; target timestamps identify absolute raw records')
    if store is not None:
        entity = store.arrays('segments', seg.hash,
                              dict(x=seg.x, y=seg.y, timestamps=seg.ts,
                                   valid_window_origin_indices=origin, valid_window_target_timestamps=target_ts))
        record['entity'] = entity
    return record, origin, target_ts


def _memory_bytes():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Incomplete evidence JSONL {path}:{line_no}; preserve and diagnose') from exc
    return records


def _load_state(initial, store, wanted_hash):
    path = store.root / 'weights' / wanted_hash[:2] / (wanted_hash + '.pt')
    if not path.exists():
        raise FileNotFoundError(f'Missing resumable state entity: {path}')
    model = copy.deepcopy(initial)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    if state_hash(model) != wanted_hash:
        raise ValueError(f'Resumable state content mismatch: {path}')
    return model


def run_graph(provider, policies, independent=False, evidence_dir=None, run_key='', resume=True,
              stage4r_lock_path: str | Path | None = None,
              expected_stage4r_lock_sha256: str | None = None,
              allow_synthetic_stage4r_lock_for_tests: bool = False):
    """Run each arm's causal deployment chain with exact same-state sharing.

    Optional provider hooks: initial_provenance; window_metadata(seg);
    offline_audit(stage) (called only after the sealed service access);
    budget_guard(stage) (called before a stage, no policy outcomes supplied).
    All state and window entities are retained when evidence_dir is supplied.
    """
    started = time.perf_counter()
    policies = [normalize_arm(a) for a in policies]
    if not policies or len({a['id'] for a in policies}) != len(policies):
        raise ValueError('Missing/duplicated arms')
    stage4r_mode = (any('stage4r_variant' in a for a in policies) or
                    stage4r_lock_path is not None or expected_stage4r_lock_sha256 is not None or
                    allow_synthetic_stage4r_lock_for_tests)
    stage4r_lock_sha256 = None
    if stage4r_mode:
        validate_fixed_stage4r_policies(policies)
        if evidence_dir is None:
            raise ValueError('Stage4-R requires durable evidence_dir before service access')
        if allow_synthetic_stage4r_lock_for_tests:
            if (provider.metadata.get('task_type') != 'SYNTHETIC_SOFTWARE_TEST' or
                    stage4r_lock_path is not None or expected_stage4r_lock_sha256 is not None):
                raise ValueError('Stage4-R synthetic escape requires a labelled software-test provider')
            stage4r_lock_sha256 = 'SYNTHETIC_SOFTWARE_TEST_ONLY'
        else:
            if stage4r_lock_path is None or expected_stage4r_lock_sha256 is None:
                raise ValueError('STAGE4R_LOCK_INVALID: frozen lock path and expected SHA required')
            stage4r_lock_sha256 = verify_stage4r_lock(stage4r_lock_path,
                expected_sha256=expected_stage4r_lock_sha256,
                code_dir=Path(__file__).resolve().parent)
    if provider.metadata.get('task_type', 'workload') == 'workload' and any(a['learning_rate'] != .01 for a in policies):
        raise ValueError('Workload candidate LR must remain exactly .01; thermal branch only')
    task_identity = {k: v for k, v in provider.metadata.items() if k not in ('drift_regime', 'true_regime')}
    key_identity = dict(task=task_identity, arms=policies, independent=independent)
    if stage4r_mode:
        key_identity['stage4r_lock_sha256'] = stage4r_lock_sha256
    actual_key = run_key or object_digest(key_identity)
    store = EvidenceStore(evidence_dir, actual_key) if evidence_dir is not None else None
    if store:
        # A caller may resume the same named task, never silently replace its
        # scientific policies, inputs or precision setup with a different run.
        manifest = dict(run_key=actual_key, task_identity=task_identity, policies=policies,
                        stages=int(provider.stages), independent=bool(independent),
                        optimizer_base=OPTIMIZER_SPEC, score_version=SCORE_VERSION)
        if stage4r_mode:
            manifest['stage4r_lock_sha256'] = stage4r_lock_sha256
        manifest_path = store.task_dir / 'run_manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
                raise ValueError('Named task cannot resume with changed policies, inputs or stage count')
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    rows = []
    training_nodes = naive_nodes = 0
    timings = []
    completed_stages = 0
    prior_nodes = {}
    prior_seconds = 0.
    # Engineering-only causal trace. Sequence numbers persist across resumption;
    # monotonic clock readings are compared only within their logging session.
    prior_events = _read_jsonl(store.task_dir / 'events.jsonl') if store else []
    event_sequence = max((int(e['event_sequence']) for e in prior_events), default=0)
    logging_session_id = object_digest(dict(pid=os.getpid(), process_counter_ns=time.perf_counter_ns(),
                                           actual_wall_time_ns=time.time_ns(), run_key=actual_key))
    last_monotonic_ns = 0
    def record_event(kind, stage, **fields):
        nonlocal event_sequence, last_monotonic_ns
        actual_monotonic_ns = time.perf_counter_ns()
        if actual_monotonic_ns <= last_monotonic_ns:
            raise RuntimeError('Non-increasing actual monotonic clock; do not backfill causal timing')
        event_sequence += 1
        last_monotonic_ns = actual_monotonic_ns
        event = dict(run_key=actual_key, event_sequence=event_sequence,
                     logging_session_id=logging_session_id, monotonic_ns=actual_monotonic_ns,
                     event_type=kind, stage=stage, actual_utc=utc_now(), **fields)
        if store:
            store.append('events.jsonl', event)
        return event
    try:
        record_event('TASK_STARTED_OR_RESUMED', 0, resume=bool(prior_events))
        initial = provider.initial()
        initial_hash = state_hash(initial)
        if store:
            initial_entity = store.model(initial, initial_hash)
            store.append('initial.jsonl', dict(run_key=actual_key, initial_state_hash=initial_hash,
                          entity=initial_entity, source_provenance=getattr(provider, 'initial_provenance', {}),
                          metadata=provider.metadata, utc=utc_now()))
        parents = {a['id']: initial for a in policies}
        full = evidence_blocks(provider.initial_history())
        full_xy = [provider.xy(b) for b in full]
        full_records = [_segment_record(provider, b, xy, store)[0] for b, xy in zip(full, full_xy)]
        if store and resume:
            completions = _read_jsonl(store.task_dir / 'stage_completion.jsonl')
            finished = {int(r['stage']): r for r in completions if r.get('status') == 'COMPLETE'}
            if finished and sorted(finished) != list(range(1, max(finished) + 1)):
                raise ValueError('Resumption requires a contiguous complete stage prefix')
            saved_rows = _read_jsonl(store.task_dir / 'chains.jsonl')
            seals = {int(r['stage']): r for r in _read_jsonl(store.task_dir / 'decision_seals.jsonl')}
            for oldstage in sorted(finished):
                if oldstage > provider.stages:
                    raise ValueError('Resuming a shorter lifecycle would drop completed stages')
                # A previously completed stage is reconstructed from saved chains;
                # its service capability is restored only using its durable seal.
                oldrows = {r['arm']: r for r in saved_rows if int(r['stage']) == oldstage}
                if set(oldrows) != {a['id'] for a in policies} or oldstage not in seals:
                    raise ValueError('Missing complete arm chains or original decision seal')
                if any(r['initial_state_hash'] != initial_hash for r in oldrows.values()):
                    raise ValueError('M0 changed during resumption')
                update, verification, _ = provider.maintenance(oldstage)
                if any(r['update_data_hash'] != update.hash or r['verification_data_hash'] != verification.hash for r in oldrows.values()):
                    raise ValueError('Maintenance input changed during resumption')
                old_decisions = {a['id']: int(oldrows[a['id']]['commit']) for a in policies}
                original_seal = seals[oldstage]
                if any(r['decision_seal'] != original_seal['seal_sha256'] for r in oldrows.values()):
                    raise ValueError('Saved decision seal mismatch')
                restore_event = record_event('RESTORE_PRIOR_SERVICE_CAPABILITY', oldstage,
                                              original_decision_seal=original_seal['seal_sha256'],
                                              no_new_scoring_or_adaptation=True)
                service = provider.service(oldstage, old_decisions)
                record_event('PRIOR_SERVICE_CAPABILITY_RESTORED', oldstage,
                              source_restore_event_sequence=restore_event['event_sequence'],
                              no_new_scoring_or_adaptation=True)
                if any(r['service_data_hash'] != service.hash for r in oldrows.values()):
                    raise ValueError('Service input changed during resumption')
                for a in policies:
                    r = oldrows[a['id']]
                    for field in ('method', 'threshold', 'scale', 'retention_threshold', 'retention_scale', 'fallback', 'learning_rate'):
                        if r[field] != a[field]:
                            raise ValueError('Saved arm settings changed during resumption')
                    if r['parent_state_hash'] != state_hash(parents[a['id']]):
                        raise ValueError('Saved deployment chain is discontinuous')
                    parents[a['id']] = _load_state(initial, store, r['deployed_state_hash'])
                    rows.append(r)
                recent = evidence_blocks(verification)
                rx = [provider.xy(b) for b in recent]
                rr = [_segment_record(provider, b, xy, store)[0] for b, xy in zip(recent, rx)]
                full += recent
                full_xy += rx
                full_records += rr
                completed_stages += 1
                naive_nodes += sum(a['method'] != 'Frozen' and not a['fallback'] for a in policies)
                prior_seconds += float(finished[oldstage]['stage_seconds'])
            all_old_nodes = _read_jsonl(store.task_dir / 'nodes.jsonl')
            prior_nodes = {r['computational_key']: r for r in all_old_nodes}
            for old in prior_nodes.values():
                if int(old['stage']) <= completed_stages:
                    training_nodes += int(old['optimizer_steps'] > 0)
                    timings.append(dict(stage=old['stage'], node_id=old['node_id'], training_needed=old['optimizer_steps'] > 0,
                      training_seconds=old['training_seconds'], scoring_seconds=old['scoring_seconds'],
                      serialization_seconds=old['serialization_seconds'],
                      node_seconds=old['training_seconds'] + old['scoring_seconds'] + old['serialization_seconds'],
                      optimizer_steps=old['optimizer_steps'], restored_from_prior_execution=True))
        precision = dict(device=DEVICE, torch=torch.__version__, dtype=str(next(initial.parameters()).dtype),
                         cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                         cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                         matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                         cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
        for stage in range(completed_stages + 1, provider.stages + 1):
            if hasattr(provider, 'budget_guard'):
                provider.budget_guard(stage)
            stage_start = time.perf_counter()
            update, verification, attack_metadata = provider.maintenance(stage)
            update_xy = provider.xy(update)
            update_record, _, _ = _segment_record(provider, update, update_xy, store)
            recent = evidence_blocks(verification)
            recent_xy = [provider.xy(b) for b in recent]
            recent_metadata = [_segment_record(provider, b, xy, store) for b, xy in zip(recent, recent_xy)]
            recent_records = [r[0] for r in recent_metadata]
            blocks = full + recent
            block_records = full_records + recent_records
            batches = full_xy + recent_xy
            counts = [len(b[0]) for b in batches]
            if not all(counts):
                raise ValueError('Empty verification block')
            batch = Batch(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches]), 'verification')
            block_key = object_digest(dict(blocks=block_records, n_recent=len(recent), score_version=SCORE_VERSION))
            node_cache = {}
            chosen = {}
            pending = []
            for a in policies:
                parent = parents[a['id']]
                parent_hash = state_hash(parent)
                train_needed = a['method'] != 'Frozen' and not a['fallback']
                naive_nodes += int(train_needed)
                training_key = dict(task=task_identity, initial_state_hash=initial_hash,
                                    parent_state_hash=parent_hash, update_window_hash=update_record['window_hash'],
                                    update_segment_hash=update.hash,
                                    optimizer={**OPTIMIZER_SPEC, 'learning_rate': a['learning_rate']},
                                    actual_seed=int(provider.seed_for_adaptation(stage)), precision=precision,
                                    train_needed=train_needed)
                cache_key = object_digest({**training_key, 'independent_arm': a['id'] if independent else ''})
                cached = prior_nodes.get(cache_key)
                if cache_key not in node_cache and cached is not None:
                    if cached['training_key'] != training_key or cached['scoring_block_identity'] != block_key or int(cached['stage']) != stage:
                        raise ValueError('Persistent node identity conflict')
                    candidate_hash = cached['candidate_entity']['content_id']
                    candidate = _load_state(initial, store, candidate_hash)
                    scores = cached['scores']
                    score_detail = cached['score_detail'] or None
                    if score_detail is not None:
                        ev_path = Path(cached['verification_entity']['path'])
                        with np.load(ev_path, allow_pickle=False) as z:
                            recomputed, arrays_detail = score_arrays(z['parent_loss'], z['candidate_loss'], counts, len(recent))
                        if recomputed != scores:
                            raise ValueError('Cached score no longer reproduces stored window losses')
                        score_detail = arrays_detail
                    timing = dict(stage=stage, node_id=cached['node_id'], training_needed=train_needed,
                        training_seconds=cached['training_seconds'], scoring_seconds=cached['scoring_seconds'],
                        serialization_seconds=cached['serialization_seconds'],
                        node_seconds=cached['training_seconds'] + cached['scoring_seconds'] + cached['serialization_seconds'],
                        optimizer_steps=cached['optimizer_steps'], restored_from_prior_execution=True)
                    node_cache[cache_key] = candidate, scores, candidate_hash, cached['node_id'], timing, score_detail
                    training_nodes += int(train_needed)
                    timings.append(timing)
                if cache_key not in node_cache:
                    node_start = time.perf_counter()
                    if DEVICE == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    sync()
                    tr_start = time.perf_counter()
                    if train_needed:
                        candidate = adapt_at_learning_rate(parent, update_xy,
                          seed=provider.seed_for_adaptation(stage), learning_rate=a['learning_rate'])
                        training_nodes += 1
                    else:
                        candidate = parent
                    sync()
                    train_seconds = time.perf_counter() - tr_start
                    if state_hash(parent) != parent_hash:
                        raise AssertionError('Candidate modified deployed parent')
                    candidate_hash = state_hash(candidate)
                    sc_start = time.perf_counter()
                    if train_needed:
                        oldpred, oldloss, _ = predictions_losses(parent, batch)
                        newpred, newloss, _ = predictions_losses(candidate, batch)
                        scores, score_detail = score_arrays(oldloss, newloss, counts, len(recent))
                    else:
                        scores = {k: 0. for k in SCORE_KEYS}
                        score_detail = None
                    sync()
                    score_seconds = time.perf_counter() - sc_start
                    score_id = object_digest(dict(training_key=training_key, candidate_state_hash=candidate_hash,
                                                  verification_identity=block_key))
                    ser_start = time.perf_counter()
                    evidence_entity = None
                    if store:
                        parent_entity = store.model(parent, parent_hash)
                        candidate_entity = store.model(candidate, candidate_hash)
                        if train_needed:
                            evidence_entity = store.arrays('verification', score_id,
                              dict(parent_prediction=oldpred, candidate_prediction=newpred,
                                   target=batch[1].detach().cpu().numpy(), parent_loss=oldloss,
                                   candidate_loss=newloss, paired_loss_delta=newloss - oldloss,
                                   block_means=score_detail['block_means'], block_counts=score_detail['counts'],
                                   block_offsets=score_detail['offsets']))
                        node_record = dict(node_id=score_id, computational_key=cache_key, run_key=actual_key,
                          stage=stage, training_key=training_key, scoring_block_identity=block_key,
                          parent_entity=parent_entity, candidate_entity=candidate_entity,
                          verification_entity=evidence_entity, update_identity=update_record,
                          block_identities=block_records, scores=scores,
                          score_detail={k: v for k, v in (score_detail or {}).items() if not isinstance(v, np.ndarray)},
                          optimizer_steps=50 if train_needed else 0,
                          verification_status='COMPUTED' if train_needed else 'NOT_EVALUATED_FROZEN_ENDPOINT',
                          actual_seed=int(provider.seed_for_adaptation(stage)), training_seconds=train_seconds,
                          scoring_seconds=score_seconds, serialization_seconds=time.perf_counter() - ser_start,
                          gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if DEVICE == 'cuda' else 0,
                          host_rss_bytes=_memory_bytes(), recorded_utc=utc_now())
                        store.append('nodes.jsonl', node_record)
                    ser_seconds = time.perf_counter() - ser_start
                    timing = dict(stage=stage, node_id=score_id, training_needed=train_needed,
                                  training_seconds=train_seconds, scoring_seconds=score_seconds,
                                  serialization_seconds=ser_seconds,
                                  node_seconds=time.perf_counter() - node_start,
                                  optimizer_steps=50 if train_needed else 0)
                    timings.append(timing)
                    node_cache[cache_key] = candidate, scores, candidate_hash, score_id, timing, score_detail
                candidate, scores, candidate_hash, score_id, timing, score_detail = node_cache[cache_key]
                if stage4r_mode and a['method'] in ('RAVM', 'DualMean'):
                    details = stage4r_decision_details(a, scores, score_detail,
                        prior_history_block_count=min(12, len(full)),
                        recent_segments=recent,
                        recent_window_counts=[len(x[0]) for x in recent_xy],
                        total_verification_blocks=len(blocks))
                else:
                    details = decision_details(a, scores)
                r = dict(arm=a['id'], method=a['method'], threshold=a['threshold'], scale=a['scale'],
                         learning_rate=a['learning_rate'],
                         retention_threshold=a['retention_threshold'], retention_scale=a['retention_scale'],
                         fallback=a['fallback'], stage=stage, parent_state_hash=parent_hash,
                         candidate_state_hash=candidate_hash, initial_state_hash=initial_hash,
                         update_data_hash=update.hash, verification_data_hash=verification.hash,
                         decision_ts=int(verification.ts[-1]), node_id=score_id,
                         candidate_update_time_s=timing['training_seconds'],
                         verification_time_s=timing['scoring_seconds'],
                         number_of_backward_updates=50 if train_needed else 0,
                         **details, **scores)
                if stage4r_mode:
                    r['stage4r_lock_sha256'] = stage4r_lock_sha256
                if score_detail is not None:
                    r.update(median_rec=score_detail['recent_median'], mad_rec=score_detail['recent_mad'],
                             median_hist=score_detail['history_median'], mad_hist=score_detail['history_mad'],
                             kappa_recent=0., kappa_history=.5)
                chosen[a['id']] = r['commit']
                pending.append((a, parent, candidate, r))
            # Seal inputs and ALL online decisions before asking for service.
            sealed_object = dict(run_key=actual_key, stage=stage, update_identity=update_record,
                evidence_identity=block_key,
                decisions=[dict(arm=r['arm'], commit=r['commit'], parent=r['parent_state_hash'],
                                candidate=r['candidate_state_hash'], threshold=r['threshold'],
                                retention_threshold=r['retention_threshold'], node_id=r['node_id'],
                                **({k: v for k, v in r.items() if k.startswith('stage4r_')}
                                   if stage4r_mode else {}))
                           for _, _, _, r in pending])
            seal = object_digest(sealed_object)
            seal_time_ns = time.time_ns()
            sealed_utc = utc_now()
            if store:
                store.append('decision_seals.jsonl', dict(**sealed_object, seal_sha256=seal, sealed_utc=sealed_utc,
                             decision_seal_time_ns=seal_time_ns))
            sealed_event = record_event('DECISIONS_DURABLY_SEALED', stage, decision_seal=seal,
                                         decision_count=len(pending))
            service_first_access_time_ns = time.time_ns()
            service_first_access_utc = utc_now()
            service_event = record_event('SERVICE_ACCESS_OPEN', stage, decision_seal=seal,
                                          source_decision_event_sequence=sealed_event['event_sequence'])
            service = provider.service(stage, chosen)
            service_return_event = record_event('SERVICE_DATA_RETURNED', stage, decision_seal=seal,
                                                 source_service_event_sequence=service_event['event_sequence'])
            if not all(r['decision_ts'] < int(service.ts[0]) for _, _, _, r in pending):
                raise AssertionError('Future service timestamps overlap decision inputs')
            service_xy = provider.xy(service)
            service_record, _, _ = _segment_record(provider, service, service_xy, store)
            loss_cache = {}
            for _, parent, candidate, r in pending:
                for model, h in ((parent, r['parent_state_hash']), (candidate, r['candidate_state_hash'])):
                    if h not in loss_cache:
                        pred, loss, mae = predictions_losses(model, service_xy)
                        loss_cache[h] = float(loss.mean()), pred, loss, mae
                        if store:
                            eval_id = object_digest(dict(state_hash=h, service_window_hash=service_record['window_hash']))
                            store.arrays('service', eval_id,
                              dict(prediction=pred, target=service_xy[1].detach().cpu().numpy(), loss=loss, mae=mae))
            # Simulator-only audit is deliberately invoked after service access.
            if hasattr(provider, 'offline_stage_evidence'):
                offline = provider.offline_stage_evidence(stage)
            elif hasattr(provider, 'offline_audit'):
                offline = provider.offline_audit(stage)
            elif hasattr(provider, 'offline_diagnostic'):
                offline = provider.offline_diagnostic(stage)
            else:
                offline = {}
            for a, parent, candidate, r in pending:
                old = loss_cache[r['parent_state_hash']][0]
                new = loss_cache[r['candidate_state_hash']][0]
                r.update(stage_metrics(old, new, bool(r['commit'])))
                deployed_hash = r['candidate_state_hash'] if r['commit'] else r['parent_state_hash']
                r.update(deployed_state_hash=deployed_hash, service_data_hash=service.hash,
                         service_start_ts=int(service.ts[0]), decision_seal=seal,
                         decision_sealed_utc=sealed_utc, service_first_access_utc=service_first_access_utc,
                         decision_seal_time_ns=seal_time_ns,
                         service_first_access_time_ns=service_first_access_time_ns,
                         decision_event_sequence=sealed_event['event_sequence'],
                         service_event_sequence=service_event['event_sequence'],
                         service_return_event_sequence=service_return_event['event_sequence'],
                         decision_monotonic_ns=sealed_event['monotonic_ns'],
                         service_first_access_monotonic_ns=service_event['monotonic_ns'],
                         service_return_monotonic_ns=service_return_event['monotonic_ns'],
                         causal_logging_session_id=logging_session_id,
                         **attack_metadata, **provider.metadata)
                r['deployed_service_mae'] = float(loss_cache[deployed_hash][3].mean())
                rows.append(r)
                parents[a['id']] = candidate if r['commit'] else parent
                if store:
                    store.append('chains.jsonl', r)
            before_bounded = full_records[-12:]
            after_bounded = block_records[-12:]
            after_ids = {r['segment_hash'] for r in after_bounded}
            history_event = dict(stage=stage, decision_seal=seal, service_first_access_utc=service_first_access_utc,
                appended=recent_records, expired_from_bounded_history=[r for r in before_bounded if r['segment_hash'] not in after_ids],
                bounded_history=after_bounded, full_cumulative_block_count=len(blocks),
                cumulative_history_not_expired=True, event_utc=utc_now(), offline_audit=offline)
            if store:
                store.append('history_and_masks.jsonl', history_event)
                store.append('stage_completion.jsonl', dict(stage=stage, rows=len(pending),
                  parent_chains={a: state_hash(m) for a, m in parents.items()}, decision_seal=seal,
                  stage_seconds=time.perf_counter() - stage_start, status='COMPLETE', recorded_utc=utc_now()))
            full, full_xy, full_records = blocks, batches, block_records
            completed_stages += 1
            complete_event = record_event('STAGE_COMPLETED', stage, decision_seal=seal,
                                           recordcount=len(pending), completed_stages=completed_stages)
            if hasattr(provider, 'stage_completed'):
                # Scheduling/resource callback after the complete durable stage;
                # no verification scores, losses, masks or regime labels supplied.
                stage_timings = [t for t in timings if t['stage'] == stage]
                provider.stage_completed(stage=stage, recordcount=len(pending),
                  elapsed_seconds=time.perf_counter() - started,
                  training_nodes=training_nodes, naive_training_nodes=naive_nodes,
                  stats=dict(completed_stages=completed_stages, logical_rows=len(rows),
                    stage_seconds=time.perf_counter() - stage_start,
                    stage_training_nodes=sum(t['training_needed'] for t in stage_timings),
                    new_node_timings=[t for t in stage_timings if not t.get('restored_from_prior_execution', False)],
                    evidence_task_dir=str(store.task_dir) if store else None,
                    evidence_new_entity_bytes=store.new_bytes if store else 0,
                    event_sequence=complete_event['event_sequence']))
    except Exception as exc:
        if store:
            if rows:
                pd.DataFrame(rows).to_csv(store.task_dir / f'partial_rows_{time.time_ns()}.csv', index=False)
            store.append('failures.jsonl', dict(run_key=actual_key, exception_type=type(exc).__name__,
              message=str(exc), traceback=traceback.format_exc(), completed_stages=completed_stages,
              completed_rows=len(rows), elapsed_seconds=time.perf_counter() - started, recorded_utc=utc_now()))
        raise
    stats = dict(training_nodes=training_nodes, naive_training_nodes=naive_nodes,
                 optimizer_steps=training_nodes * 50, seconds=time.perf_counter() - started,
                 completed_stages=completed_stages, logical_rows=len(rows), independent=bool(independent),
                 resumed_prior_completed_stage_seconds=prior_seconds,
                 node_timings=timings, evidence_new_entity_bytes=store.new_bytes if store else 0,
                 evidence_new_entity_count=store.new_entities if store else 0,
                 evidence_task_dir=str(store.task_dir) if store else None,
                 clean_corrupt_four_way_inference='NOT_EVALUATED')
    if store:
        store.append('completion.jsonl', dict(status='COMPLETE', run_key=actual_key, stats=stats, utc=utc_now()))
    return pd.DataFrame(rows), stats
