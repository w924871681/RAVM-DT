"""Pure B-1 proposed three-action controller and equally informed baselines.

Only predecision verification evidence may enter this module.  No Service,
regime, corruption mask, or future target is accepted by its public contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Mapping

RETAIN = "RETAIN"
ADAPT = "ADAPT_AND_COMMIT"
RESTORE = "RESTORE_ARCHIVED"
K = 4
RDEV = 0.00560364755324872
DUAL_THRESHOLD = 0.0001681094265974
RECENT_THRESHOLD = 0.0004202735664936
MWU_THRESHOLD = 8.40547132987308e-05
METHODS = (
    "CMC_RAVM_ORIGINAL", "DUAL_MEAN_FIXED", "RECENT_MEAN_FIXED", "MWU_MEAN_FIXED",
    "POOL_GREEDY_RECENT", "POOL_DUAL_MEAN", "POOL_SIMPLE_ENSEMBLE",
    "H1_PROVISIONAL_REVERSIBLE", "H1_NO_RESTORE", "Frozen", "AlwaysFT",
)
POOL_METHODS = frozenset({"POOL_GREEDY_RECENT", "POOL_DUAL_MEAN",
                          "POOL_SIMPLE_ENSEMBLE", "H1_PROVISIONAL_REVERSIBLE",
                          "H1_NO_RESTORE"})


@dataclass(frozen=True)
class StateEntry:
    state_hash: str
    file_sha256: str
    file_bytes: int
    relative_path: str
    center: int
    stream: str
    source_sha256: str
    created_stage: int
    first_deployed_stage: int
    available_stage: int
    training_max_ts: int
    last_archived_stage: int | None = None
    optimizer_state_saved: bool = False

    def validate(self):
        for h in (self.state_hash, self.file_sha256, self.source_sha256):
            if len(h) != 64 or any(c not in "0123456789abcdef" for c in h.lower()):
                raise ValueError("invalid weight/source hash")
        if self.file_bytes <= 0 or self.created_stage < 0 or self.available_stage < 1:
            raise ValueError("invalid state age/size")
        if self.optimizer_state_saved:
            raise ValueError("B-1 full optimizer resume is not implemented")


@dataclass
class Pool:
    center: int
    stream: str
    source_sha256: str
    m0_hash: str
    current_hash: str
    entries: dict[str, StateEntry]
    predecessor_hash: str | None = None
    provisional_since_stage: int | None = None
    last_stage: int = 0

    @classmethod
    def initial(cls, entry: StateEntry) -> "Pool":
        if entry.created_stage != 0 or entry.first_deployed_stage != 0 or entry.available_stage != 1:
            raise ValueError("M0 identity invalid")
        obj = cls(entry.center, entry.stream, entry.source_sha256,
                  entry.state_hash, entry.state_hash, {entry.state_hash: entry})
        obj.validate(1)
        return obj

    @classmethod
    def from_dict(cls, raw: dict) -> "Pool":
        data = dict(raw)
        data["entries"] = {h: StateEntry(**e) for h, e in data["entries"].items()}
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self, stage: int):
        if len(self.entries) > K or self.m0_hash not in self.entries or self.current_hash not in self.entries:
            raise ValueError("K/M0/current invariant failed")
        if self.predecessor_hash is not None and self.predecessor_hash not in self.entries:
            raise ValueError("missing protected predecessor")
        if (self.predecessor_hash is None) != (self.provisional_since_stage is None):
            raise ValueError("invalid provisional pairing")
        for h, entry in self.entries.items():
            entry.validate()
            if h != entry.state_hash or entry.center != self.center or entry.stream != self.stream:
                raise ValueError("cross-center/stream or key mismatch")
            if entry.source_sha256 != self.source_sha256 or entry.available_stage > stage:
                raise ValueError("future or cross-source weight")

    def eligible_archive(self, stage: int) -> dict[str, StateEntry]:
        self.validate(stage)
        return {h: e for h, e in self.entries.items()
                if h != self.current_hash and e.first_deployed_stage < stage
                and e.available_stage <= stage}

    def transition(self, stage: int, action: str, selected_hash: str,
                   candidate: StateEntry | None = None) -> "Pool":
        self.validate(stage)
        if stage != self.last_stage + 1:
            raise ValueError("non-contiguous stage transition")
        before = self.current_hash
        if action == RETAIN:
            if selected_hash != before:
                raise ValueError("retain must keep current state")
        elif action == ADAPT:
            if candidate is None or selected_hash != candidate.state_hash:
                raise ValueError("adapt action requires exact trained candidate")
            candidate.validate()
            if candidate.center != self.center or candidate.stream != self.stream or candidate.source_sha256 != self.source_sha256:
                raise ValueError("candidate identity mismatch")
            if candidate.created_stage != stage or candidate.first_deployed_stage != stage or candidate.available_stage != stage + 1:
                raise ValueError("candidate age/availability mismatch")
            if selected_hash == before:
                raise ValueError("identity adaptation is retain")
        elif action == RESTORE:
            if selected_hash not in self.eligible_archive(stage):
                raise ValueError("restore requires previously deployed eligible archive")
        else:
            raise ValueError("unknown action")
        entries = dict(self.entries)
        if action == ADAPT:
            entries[selected_hash] = candidate
        if action != RETAIN:
            old = entries[before]
            entries[before] = StateEntry(**{**asdict(old), "last_archived_stage": stage})
        predecessor = before if action != RETAIN else None
        provisional = stage if action != RETAIN else None
        protected = {self.m0_hash, selected_hash}
        if predecessor is not None:
            protected.add(predecessor)
        while len(entries) > K:
            victims = [e for h, e in entries.items() if h not in protected]
            if not victims:
                raise ValueError("K cannot accommodate protected states")
            victim = min(victims, key=lambda e: (e.last_archived_stage if e.last_archived_stage is not None else e.first_deployed_stage,
                                                 e.state_hash))
            del entries[victim.state_hash]
        new = Pool(self.center, self.stream, self.source_sha256, self.m0_hash,
                   selected_hash, entries, predecessor, provisional, stage)
        new.validate(stage + 1)
        return new


@dataclass(frozen=True)
class PreDecisionEvidence:
    stage: int
    current_hash: str
    candidate_hash: str
    recent_block_mse: Mapping[str, tuple[float, ...]]
    pair_scores: Mapping[str, Mapping[str, float]]
    cmc_original_commit: bool | None = None
    expected_recent_blocks: int = 6
    pool_scored: bool = True

    def validate(self, pool: Pool):
        if self.stage != pool.last_stage + 1 or self.current_hash != pool.current_hash:
            raise ValueError("evidence stage/parent mismatch")
        expected = (set(pool.entries) if self.pool_scored else {pool.current_hash}) | {self.candidate_hash}
        if set(self.recent_block_mse) != expected:
            raise ValueError("missing or extra model evidence")
        for h, blocks in self.recent_block_mse.items():
            if len(blocks) != self.expected_recent_blocks or not all(math.isfinite(v) and v >= 0 for v in blocks):
                raise ValueError(f"nonfinite/missing recent evidence: {h}")
        if set(self.pair_scores) != expected - {self.current_hash}:
            raise ValueError("pair-score coverage mismatch")
        required = {"recent_mean", "recent_median", "history_mean", "history_robust", "cumulative_mean", "mwu_p"}
        for h, scores in self.pair_scores.items():
            if set(scores) != required or not all(math.isfinite(v) for v in scores.values()) or not 0 <= scores["mwu_p"] <= 1:
                raise ValueError(f"invalid pair scores: {h}")

    def recent(self, h: str) -> float:
        return float(sum(self.recent_block_mse[h]) / self.expected_recent_blocks)


@dataclass(frozen=True)
class Choice:
    action: str
    selected_hash: str
    reason: str
    candidate_eligible: bool
    archived_eligible: tuple[str, ...]
    current_recent_mse: float
    selected_recent_mse: float


def _dual_pass(scores: Mapping[str, float] | None) -> bool:
    return bool(scores is not None and scores["recent_mean"] <= -DUAL_THRESHOLD
                and scores["history_mean"] <= 0.)


def choose(method: str, pool: Pool, evidence: PreDecisionEvidence) -> Choice:
    if method not in METHODS:
        raise ValueError("not one of 11 locked methods")
    pool.validate(evidence.stage)
    evidence.validate(pool)
    cur = pool.current_hash
    cand = evidence.candidate_hash
    archive = pool.eligible_archive(evidence.stage)
    current_recent = evidence.recent(cur)
    cscore = evidence.pair_scores.get(cand)
    candidate_eligible = False
    archived: list[str] = []
    reason = "NO_STRICT_IMPROVEMENT"
    if method == "Frozen":
        return Choice(RETAIN, cur, "FROZEN_ENDPOINT", False, (), current_recent, current_recent)
    if method == "AlwaysFT":
        return Choice(ADAPT if cand != cur else RETAIN, cand if cand != cur else cur,
                      "ALWAYS_UPDATE_ENDPOINT", True, (), current_recent, evidence.recent(cand))
    if method == "CMC_RAVM_ORIGINAL":
        if evidence.cmc_original_commit is None:
            raise ValueError("original CMC gate evidence absent")
        candidate_eligible = bool(evidence.cmc_original_commit)
    elif method == "DUAL_MEAN_FIXED":
        candidate_eligible = _dual_pass(cscore)
    elif method == "RECENT_MEAN_FIXED":
        candidate_eligible = cscore["recent_mean"] <= -RECENT_THRESHOLD
    elif method == "MWU_MEAN_FIXED":
        candidate_eligible = cscore["recent_mean"] <= -MWU_THRESHOLD and cscore["mwu_p"] < .2
    elif method == "POOL_GREEDY_RECENT":
        candidate_eligible = True
        archived = list(archive)
    elif method == "POOL_DUAL_MEAN":
        candidate_eligible = _dual_pass(cscore)
        archived = [h for h in archive if _dual_pass(evidence.pair_scores[h])]
    elif method in ("H1_PROVISIONAL_REVERSIBLE", "H1_NO_RESTORE"):
        candidate_eligible = _dual_pass(cscore)
        archived = list(archive) if method == "H1_PROVISIONAL_REVERSIBLE" else []
        if method == "H1_PROVISIONAL_REVERSIBLE" and pool.predecessor_hash is not None:
            pred = pool.predecessor_hash
            if pred not in archive:
                raise ValueError("provisional predecessor is not a legal archive")
            if evidence.recent(pred) < current_recent:
                return Choice(RESTORE, pred, "PROVISIONAL_PREDECESSOR_BETTER",
                              candidate_eligible, tuple(sorted(archived)),
                              current_recent, evidence.recent(pred))
    elif method == "POOL_SIMPLE_ENSEMBLE":
        candidate_eligible = _dual_pass(cscore)
        if candidate_eligible and cand != cur:
            return Choice(ADAPT, cand, "ENSEMBLE_ADD_DUAL_ACCEPT",
                          True, (), current_recent, evidence.recent(cand))
        return Choice(RETAIN, cur, "ENSEMBLE_POOL_UNCHANGED", False, (), current_recent, current_recent)
    if method not in POOL_METHODS:
        if candidate_eligible and cand != cur:
            return Choice(ADAPT, cand, "FIXED_GATE_ACCEPT", True, (),
                          current_recent, evidence.recent(cand))
        return Choice(RETAIN, cur, "FIXED_GATE_RETAIN", candidate_eligible, (), current_recent, current_recent)
    options: list[tuple[float, int, str, str]] = []
    if candidate_eligible and cand != cur and evidence.recent(cand) < current_recent:
        options.append((evidence.recent(cand), evidence.stage, cand, ADAPT))
    for h in archived:
        if evidence.recent(h) < current_recent:
            e = archive[h]
            age = e.last_archived_stage if e.last_archived_stage is not None else e.first_deployed_stage
            options.append((evidence.recent(h), age, h, RESTORE))
    if not options:
        return Choice(RETAIN, cur, reason, candidate_eligible, tuple(sorted(archived)),
                      current_recent, current_recent)
    score, _, selected, action = min(options, key=lambda x: (x[0], x[1], x[2]))
    return Choice(action, selected, "MIN_ELIGIBLE_RECENT_MSE",
                  candidate_eligible, tuple(sorted(archived)), current_recent, score)
