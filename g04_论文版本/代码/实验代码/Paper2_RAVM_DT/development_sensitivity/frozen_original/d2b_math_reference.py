"""RAVM-DT D2b-v0.2: pure synthetic arithmetic sanity checks only.

NO archived Service use, NO neural model, NO training, NO online deployment.
This is a minimal expression of the draft control logic, not a scientific experiment.
"""
from dataclasses import dataclass
from math import isfinite
from statistics import median

DELTA_IMP = 2.8018237766243602e-5
KH = 0.5


def mad(values):
    m = median(values)
    return median([abs(x - m) for x in values])


def quantile(values, q):
    s = sorted(values)
    if not s or not 0.0 <= q <= 1.0:
        raise ValueError('Nonempty values and valid q required')
    h = (len(s) - 1) * q
    j = int(h)
    return s[j] * (1 - (h - j)) + s[min(j + 1, len(s) - 1)] * (h - j)


def original(recent, old, new, delta=DELTA_IMP):
    ur = median(recent)  # original k_rec=0
    hist = list(old) + list(new)
    h0 = median(hist) + KH * mad(hist)
    return bool(ur <= -delta and h0 <= 0), ur, h0


@dataclass(frozen=True)
class PastAudit:
    stage: int
    seal_ns: int
    matured_ns: int
    hashes_ok: bool
    snapshots_ok: bool
    delta_service: float  # synthetic, retrospectively available ONLY after matured_ns
    proxy_old: float
    proxy_new: float
    proxy_recent: float
    parent_verification_scale: float


def draft_v02(recent, old, new, past, *, stage, decision_start_ns,
              q=0.8, k_min=4, delta=DELTA_IMP, current_parent_verification_scale=1.0):
    """Returns (commit,reason), strict fallback to original for ineligible input."""
    u0, ur, h0 = original(recent, old, new, delta)
    # For a first-stage 6-block history, any unverifiable evidence, absent maturity,
    # or insufficient complete delayed audits, return the frozen original decision.
    if len(recent) != 6 or len(old) != 6 or len(new) != 6 or not past or len(past) < k_min:
        return u0, 'EXACT_ORIGINAL_FALLBACK'
    pool = tuple(sorted(past, key=lambda x: x.stage))[-6:]
    if any(not (r.stage < stage and r.seal_ns < r.matured_ns <= decision_start_ns
                and r.hashes_ok and r.snapshots_ok and
                all(isfinite(v) for v in (r.delta_service, r.proxy_old, r.proxy_new,
                                            r.proxy_recent, r.parent_verification_scale)) and
                r.parent_verification_scale > 0) for r in pool):
        return u0, 'EXACT_ORIGINAL_FALLBACK'
    if not all(isfinite(v) for v in (*recent, *old, *new)):
        return u0, 'EXACT_ORIGINAL_FALLBACK'

    e_old = [abs(r.delta_service-r.proxy_old)/r.parent_verification_scale for r in pool]
    e_new = [abs(r.delta_service-r.proxy_new)/r.parent_verification_scale for r in pool]
    prefer_new = median(e_new) < median(e_old) and 2*sum(a < b for a, b in zip(e_new,e_old)) > len(pool)
    signed_recent = [(r.delta_service-r.proxy_recent)/r.parent_verification_scale for r in pool]
    s_t = current_parent_verification_scale  # synthetic example 1.0; real rule uses parent Verification MSE
    if not isfinite(s_t) or s_t <= 0:
        return u0, 'EXACT_ORIGINAL_FALLBACK'
    b = s_t * max(0.0, quantile(signed_recent,q))
    if ur+delta+b > 0:
        return False, 'DELAYED_OPTIMISM_VETO'
    m_old, m_new = median(old),median(new)
    if prefer_new and m_old <= 0 < m_new:
        return False, 'NEW_HISTORY_ADVERSE_VETO'
    if u0:
        return bool(h0+b <= 0), 'ORIGINAL_WITH_DELAYED_GUARD'
    if ur + delta <= 0 and h0 > 0 and m_old > 0 >= m_new and prefer_new:
        fresh = list(new) + list(recent)
        fresh_score = median(fresh) + KH * mad(fresh)
        return bool(fresh_score+b <= 0), 'RELATIONALLY_VALIDATED_RELEASE_CHECK'
    return False, 'KEEP_ORIGINAL_VETO'


def audits(n=4, delta_s=-1.1, old_pred=2.0, new_pred=-1.0, recent_pred=-1.2,
           scale=1.0, valid=True):
    return [PastAudit(stage=i+1, seal_ns=10+i*10, matured_ns=15+i*10,
                      hashes_ok=valid, snapshots_ok=valid,
                      delta_service=delta_s, proxy_old=old_pred,
                      proxy_new=new_pred, proxy_recent=recent_pred,
                      parent_verification_scale=scale) for i in range(n)]


def sanity():
    old=[2.0]*6; new=[-2.0]*6; recent=[-3.0]*6
    u0,*_=original(recent,old,new)
    assert not u0
    assert draft_v02(recent,old,new,[],stage=8,decision_start_ns=100)[0]==u0
    assert draft_v02(recent,old,new,audits(valid=False),stage=8,decision_start_ns=100)[0]==u0
    assert draft_v02(recent,old,new,audits(n=3),stage=8,decision_start_ns=100)[0]==u0
    assert draft_v02(recent,old,new,audits(n=4),stage=8,decision_start_ns=15)[0]==u0
    accepted, why = draft_v02(recent,old,new,audits(),stage=8,decision_start_ns=100)
    assert accepted and why=='RELATIONALLY_VALIDATED_RELEASE_CHECK', (accepted,why)
    rev,why_rev=draft_v02(recent,new,old,audits(),stage=8,decision_start_ns=100)
    assert original(recent,old,new)[2] == original(recent,new,old)[2]
    assert not rev
    # Even when original accepts, a sufficiently positive past optimism residual vetoes.
    r2=[-0.3]*6;o2=[-0.25]*6;n2=[-0.25]*6
    assert original(r2,o2,n2)[0]
    a2,why2=draft_v02(r2,o2,n2,audits(delta_s=1.0,old_pred=1.0,new_pred=1.0,
                                              recent_pred=-1.0),stage=8,decision_start_ns=100)
    assert not a2 and why2=='DELAYED_OPTIMISM_VETO'
    print('PASS: exact fallback; invalid maturity; missing feedback; history order effect; release; delay veto.')
    print('WARNING: arithmetic examples only, not measured lifecycle results or numerical claims about risk.')

if __name__=='__main__':
    sanity()
