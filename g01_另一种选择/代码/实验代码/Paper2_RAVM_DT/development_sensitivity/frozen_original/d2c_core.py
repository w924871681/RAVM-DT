"""D2c sandbox interface primitives; synthetic fixtures only, NEVER production authority.
No old model weights, archived Service or new training are read by this module.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
import json, os, math
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey, Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature


def canonical(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=False).encode('utf-8')


def digest(x: bytes) -> str:
    return sha256(x).hexdigest()


def normalize_scope(run: str, center: str, stream: str):
    if not all(isinstance(s, str) and s and 1 <= len(s) <= 128 for s in (run, center, stream)):
        raise ValueError('All scope identifiers must be nonempty bounded strings')
    return run, center, stream


class LedgerCorrupted(RuntimeError):
    pass


class AuditLedger:
    """Hash-chain demo with fsync and restart revalidation. Not a production WORM log."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        if self.path.exists():
            for raw in self.path.read_bytes().splitlines():
                if raw:
                    self.entries.append(json.loads(raw))
        self.verify()

    @property
    def next_seq(self):
        return len(self.entries) + 1

    def verify(self):
        previous = '0'*64
        for seq, event in enumerate(self.entries, 1):
            if event.get('seq') != seq or event.get('prev_hash') != previous:
                raise LedgerCorrupted(f'Seq/parent mismatch at event {seq}')
            actual = digest(canonical({k:v for k,v in event.items() if k!='hash'}))
            if actual != event.get('hash'):
                raise LedgerCorrupted(f'Digest mismatch at event {seq}')
            previous = actual
        return previous

    def append(self, kind: str, scope: tuple[str, str, str], stage: int, data: dict):
        normalize_scope(*scope)
        if stage < 1 or not isinstance(stage, int):
            raise ValueError('Invalid stage')
        event = dict(seq=self.next_seq, kind=kind, run=scope[0], center=scope[1], stream=scope[2],
                     stage=stage, data=data, prev_hash=self.entries[-1]['hash'] if self.entries else '0'*64)
        event['hash'] = digest(canonical(event))
        wire = canonical(event) + b'\n'
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, wire)
            os.fsync(fd)
        finally:
            os.close(fd)
        self.entries.append(event)
        return event

    def event(self, sequence: int):
        if not isinstance(sequence, int) or not (1 <= sequence <= len(self.entries)):
            raise ValueError('Unrecognized event index')
        return self.entries[sequence-1]


class ModelFixtureCAS:
    """Fake model blobs, never neural model serialization; checks file identity on read."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, b: bytes):
        h = digest(b)
        target = self.root/h
        if not target.exists():
            target.write_bytes(b)
        if digest(target.read_bytes()) != h:
            raise LedgerCorrupted('CAS byte identity mismatch')
        return h

    def verify(self, h: str):
        if not isinstance(h,str) or len(h) !=64 or any(c not in '0123456789abcdef' for c in h):
            return False
        target = self.root/h
        return target.is_file() and digest(target.read_bytes()) == h


@dataclass(frozen=True, slots=True)
class LateAudit:
    run: str
    center: str
    stream: str
    stage: int
    score_event_seq: int
    maturity_event_seq: int
    receipt_sha256: str
    parent_sha256: str
    candidate_sha256: str
    delta_late: float
    proxy_old: float
    proxy_new: float
    proxy_recent: float
    scale: float


@dataclass(frozen=True, slots=True)
class VerifiedFeedbackView:
    run: str
    center: str
    stream: str
    current_stage: int
    gate_start_seq: int
    audits: tuple[LateAudit, ...]
    attested: bool
    reason: str
    attestation_signature: str


def view_message(view: VerifiedFeedbackView):
    return canonical({k:v for k,v in asdict(view).items() if k != "attestation_signature"})


class FeedbackVerifier:
    """Checks signed fixture receipts AND hash-linked event references before giving gate scalars.
    In production the maturity issuer and durable log must be external trust roots.
    """
    def __init__(self, ledger: AuditLedger, cas: ModelFixtureCAS, public_key: Ed25519PublicKey,
                 expected_issuer='D2C_ISOLATED_TEST_ONLY'):
        self.ledger, self.cas = ledger, cas
        self.public_key, self.expected_issuer = public_key, expected_issuer
        self.__view_signer = Ed25519PrivateKey.generate()
        self.view_public_key = self.__view_signer.public_key()

    def _attest(self, view):
        unsigned = VerifiedFeedbackView(view.run,view.center,view.stream,view.current_stage,
                                        view.gate_start_seq,view.audits,view.attested,view.reason,"")
        return VerifiedFeedbackView(unsigned.run,unsigned.center,unsigned.stream,unsigned.current_stage,
                   unsigned.gate_start_seq,unsigned.audits,unsigned.attested,unsigned.reason,
                   self.__view_signer.sign(view_message(unsigned)).hex())

    def _verify_one(self, a: LateAudit, scope: tuple[str,str,str], cur_stage: int, frontier: int):
        if (a.run,a.center,a.stream)!=scope or not 1 <= a.stage < cur_stage:
            raise ValueError('Wrong run/center/stream or future/current stage')
        nums=(a.delta_late,a.proxy_old,a.proxy_new,a.proxy_recent,a.scale)
        if any(not math.isfinite(x) for x in nums) or a.scale <=0:
            raise ValueError('Nonfinite feedback or nonpositive scale')
        mature=self.ledger.event(a.maturity_event_seq)
        scored=self.ledger.event(a.score_event_seq)
        if mature['kind']!='LABELS_MATURED' or scored['kind']!='LATE_PAIR_SCORED':
            raise ValueError('Maturity / score event types wrong')
        for e in (mature,scored):
            if (e['run'],e['center'],e['stream'],e['stage'])!=(a.run,a.center,a.stream,a.stage):
                raise ValueError('Event scope mismatch')
        if not (a.maturity_event_seq < a.score_event_seq < frontier):
            raise ValueError('Feedback not ready by current gate-start')
        proof=mature['data']['receipt']
        seal=self.ledger.event(proof['seal_seq'])
        service=self.ledger.event(proof['service_open_seq'])
        snap=self.ledger.event(proof['snapshot_seq'])
        if not (snap['kind']=='SNAPSHOT_SAVED' and seal['kind']=='DECISION_SEALED'
                and service['kind']=='SERVICE_ROLE_OPENED'):
            raise ValueError('Incomplete event chain')
        if not (snap['seq']<seal['seq']<service['seq']<mature['seq']<scored['seq']):
            raise ValueError('Invalid causal sequence')
        for ev in (snap,seal,service):
            if (ev['run'],ev['center'],ev['stream'],ev['stage'])!=(a.run,a.center,a.stream,a.stage):
                raise ValueError('Event chain scope mismatch')
        if (seal['data']['snapshot_seq']!=snap['seq'] or
                service['data']['seal_seq']!=seal['seq'] or
                proof['service_open_seq']!=service['seq']):
            raise ValueError('Event linkage mismatch')
        if (snap['data']['parent_sha256']!=a.parent_sha256 or
            snap['data']['candidate_sha256']!=a.candidate_sha256 or
            scored['data']['parent_sha256']!=a.parent_sha256 or
            scored['data']['candidate_sha256']!=a.candidate_sha256 or
            scored['data']['mature_seq']!=mature['seq']):
            raise ValueError('Parent/candidate snapshot chain mismatch')
        if not self.cas.verify(a.parent_sha256) or not self.cas.verify(a.candidate_sha256):
            raise ValueError('Missing or changed parent/candidate blobs')
        proof=mature['data']['receipt']
        sig=mature['data']['signature']
        if proof['issuer'] != self.expected_issuer or proof['mature_seq'] != mature['seq']:
            raise ValueError('Wrong issuer or receipt sequence')
        if (proof['run'],proof['center'],proof['stream'],proof['stage']) != (a.run,a.center,a.stream,a.stage):
            raise ValueError('Receipt identity invalid')
        if (proof['seal_seq']!=seal['seq'] or proof['service_open_seq']!=service['seq'] or
                proof['snapshot_seq']!=snap['seq'] or proof['service_sha256']!=service['data']['service_sha256'] or
                proof['label_watermark']<proof['max_target_ts']):
            raise ValueError('Receipt not bound to service identity and watermark')
        try:
            self.public_key.verify(bytes.fromhex(sig), canonical(proof))
        except (InvalidSignature,ValueError):
            raise ValueError('Invalid maturity authority signature') from None
        if a.receipt_sha256 != digest(canonical(proof)) or scored['data']['receipt_sha256'] != a.receipt_sha256:
            raise ValueError('Receipt digest mismatch')
        if scored['data']['data_role']!='HISTORICAL_MATURED_SERVICE_AUDIT':
            raise ValueError('Wrong data role')
        if (scored['data']['delta_late']!=a.delta_late or scored['data']['proxy_old']!=a.proxy_old or
                scored['data']['proxy_new']!=a.proxy_new or
                scored['data']['proxy_recent']!=a.proxy_recent or scored['data']['scale']!=a.scale):
            raise ValueError('Audit values do not match append-only record')
        if (snap['data']['old_score']!=a.proxy_old or snap['data']['new_score']!=a.proxy_new or
                snap['data']['recent_score']!=a.proxy_recent or snap['data']['scale']!=a.scale):
            raise ValueError('Late audit proxy differs from pre-seal original evidence')
        if (snap['data'].get('history_block_count') != 12 or
                snap['data'].get('recent_block_count') != 6):
            raise ValueError('Past candidate missing complete historical evidence')

    def snapshot(self, scope: tuple[str,str,str], cur_stage: int, *, audits: tuple[LateAudit, ...],
                 current_history_complete: bool, minimum=4, maximum=6):
        """Open gate-start first; tolerate zero feedback only by declining it entirely."""
        if any(e["kind"]=="GATE_START" and (e["run"],e["center"],e["stream"])==scope
               and e["stage"]==cur_stage for e in self.ledger.entries):
            return self._attest(VerifiedFeedbackView(*scope,cur_stage,0,(),False,
                                                      "FALLBACK_DUPLICATE_GATE_START", ""))
        gate=self.ledger.append('GATE_START', scope, cur_stage, dict(maturity_issuer=self.expected_issuer,
                                  feedback_count=len(audits), history_complete=current_history_complete))
        def decline(reason):
            return self._attest(VerifiedFeedbackView(*scope,cur_stage,gate['seq'],(),False,reason,""))
        try:
            self.ledger.verify()
            current_snaps=[e for e in self.ledger.entries[:gate['seq']-1] if
                    e['kind']=='SNAPSHOT_SAVED' and (e['run'],e['center'],e['stream'])==scope
                    and e['stage']==cur_stage]
            if (len(current_snaps)!=1 or not current_snaps[0]['data'].get('history_complete',False) or
                    current_snaps[0]['data'].get('history_block_count') != 12 or
                    current_snaps[0]['data'].get('recent_block_count') != 6):
                raise ValueError('CURRENT_SNAPSHOT_OR_HISTORY_NOT_ATTESTED')
            if not current_history_complete:
                raise ValueError('HISTORY_INCOMPLETE')
            if not minimum <= len(audits) <= maximum:
                raise ValueError('FEEDBACK_COUNT')
            if len({a.stage for a in audits})!=len(audits):
                raise ValueError('DUPLICATE_STAGE')
            if tuple(sorted(audits,key=lambda x:x.stage))!=audits:
                raise ValueError('NONCANONICAL_ORDER')
            # Last K completed matured feedback entries -- no arbitrary cherry picking.
            eligible=[e for e in self.ledger.entries[:gate['seq']-1] if
                      e['kind']=='LATE_PAIR_SCORED' and (e['run'],e['center'],e['stream'])==scope
                      and e['stage']<cur_stage]
            expected=tuple(e['seq'] for e in sorted(eligible,key=lambda x:x['stage'])[-maximum:])
            if tuple(a.score_event_seq for a in audits)!=expected:
                raise ValueError('NONCANONICAL_FEEDBACK_SELECTION')
            for a in audits:
                self._verify_one(a,scope,cur_stage,gate['seq'])
            # Current Service is not permitted to open before the current gate starts.
            if any(e['kind'] in ('SERVICE_ROLE_OPENED','DECISION_SEALED')
                   and (e['run'],e['center'],e['stream'])==scope
                   and e['stage']==cur_stage for e in self.ledger.entries[:gate['seq']]):
                raise ValueError('CURRENT_DECISION_OR_SERVICE_ALREADY_OPENED')
            return self._attest(VerifiedFeedbackView(*scope,cur_stage,gate['seq'],audits,True,
                                                      'VERIFIED_TEST_FIXTURE',''))
        except (ValueError,KeyError,TypeError,LedgerCorrupted,IndexError) as ex:
            return decline(f'FALLBACK_{str(ex)}')
