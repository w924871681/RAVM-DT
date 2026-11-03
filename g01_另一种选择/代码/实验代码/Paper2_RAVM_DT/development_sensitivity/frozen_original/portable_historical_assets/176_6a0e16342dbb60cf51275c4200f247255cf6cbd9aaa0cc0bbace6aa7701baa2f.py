"""Gate receives only signed scalar past feedback and current verification evidence.
No provider, Service API, raw truth, center regime or generator import.
"""
from dataclasses import dataclass
from statistics import median
from math import isfinite
from d2c_gate import d2c_gate
from d2b_math_reference import PastAudit,draft_v02,quantile

@dataclass(frozen=True,slots=True)
class Evidence:
 recent:tuple
 history:tuple
 scale:float

def baseline_decision(method,scores,threshold,fallback=False):
 if fallback:return False
 commit=scores['recent_mean']<=-threshold
 if method=='DUAL_MEAN':commit=commit and scores['history_mean']<=0
 elif method=='MWU_MEAN':commit=commit and scores['mwu_p']<.2
 elif method!='RECENT_MEAN':raise ValueError('Unknown baseline')
 return bool(commit)

def research_decision(e,audits,stage,frontier,guard_only=False):
 past=tuple(PastAudit(a.stage,1,a.score_event_seq,True,True,a.delta_late,a.proxy_old,a.proxy_new,a.proxy_recent,a.scale) for a in audits)
 if guard_only:
  if len(e.history)!=12 or len(past)<4 or not isfinite(e.scale) or e.scale<=0:raise ValueError('Guard inputs incomplete')
  b=e.scale*max(0.,quantile([(a.delta_service-a.proxy_recent)/a.parent_verification_scale for a in past],.8))
  med=median(e.history);h=med+.5*median(abs(x-med) for x in e.history)
  return bool(median(e.recent)+2.8018237766243602e-5+b<=0 and h+b<=0)
 return draft_v02(e.recent,e.history[:6],e.history[6:],past,stage=stage,decision_start_ns=frontier,current_parent_verification_scale=e.scale)[0]

def decide(e,view,original_gate,public_key,scope,stage,guard_only=False):
 return d2c_gate(view=view,original_gate=original_gate,trusted_verifier_public_key=public_key,expected_scope=scope,expected_stage=stage,research_candidate=lambda audits:research_decision(e,audits,stage,view.gate_start_seq,guard_only))
