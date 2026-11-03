"""Matched candidates sealed before reference decision/Service; oracle is postseal only."""
from d3_engine import *
from d3_adapt import make_candidate,METHODS
from core.protocol import Segment
SELECTED=(2,3,4,5,9,10,19,20)
def before_seal(td,ledger,cas,parent,reference,u,c,seed,stream,t,scope):
 if t not in SELECTED:return None
 path=td/f'matched_seal_{t:03}.json';assert not path.exists(),'No unsealed candidate recomputation'
 entries={};parent_entry=cas.put(parent)
 for m in METHODS:
  start=time.perf_counter();model,meta=make_candidate(parent,u,m,seed+c+t*100);sync();seconds=time.perf_counter()-start
  ent=cas.put(model)
  if m=='U_MSE':assert ent['state_hash']==state_hash(reference),'Matched MSE differs from frozen reference candidate'
  entries[m]=dict(candidate_entry=ent,adapt_meta=meta,training_seconds=seconds)
 seal=dict(center=c,seed=seed,stream=stream,stage=t,parent_entry=parent_entry,candidates=entries,update_hash=u.hash,execution_lock_sha256=execution_lock_sha(),scope='MATCHED_PARENT_SCREEN_NO_SERVICE_READ')
 h=atomic(path,seal,exclusive=True);ev=ledger.append('MATCHED_CANDIDATE_SET_SEALED',scope,t,dict(seal_sha256=h,methods=list(METHODS)))
 return dict(path=path.name,sha256=h,event_seq=ev['seq'])
def after_service(td,ledger,cas,parent,service,c,seed,stream,t,scope):
 if t not in SELECTED:return
 result=td/f'matched_result_{t:03}.json'
 if result.exists():return
 sealp=td/f'matched_seal_{t:03}.json';seal=read(sealp);xy=tensors(windows(service));pl=float(errors(parent,xy)[0].mean());rows=[]
 opened=next(e for e in ledger.entries if e['stage']==t and e['kind']=='SERVICE_ROLE_OPENED');event=next(e for e in ledger.entries if e['stage']==t and e['kind']=='MATCHED_CANDIDATE_SET_SEALED')
 assert event['seq']<opened['seq'] and event['data']['seal_sha256']==sha(sealp)
 for m,v in seal['candidates'].items():
  model=cas.load(v['candidate_entry']);cl=float(errors(model,xy)[0].mean());delta=cl-pl
  rows.append(dict(center=c,seed=seed,stream=stream,stage=t,regime_audit_only=REGIMES[IDS['RootScreen'].index(c)%4],method=m,parent_service_mse=pl,candidate_service_mse=cl,delta=delta,positive_delta=max(delta,0.),harmful_candidate=delta>0,training_seconds=v['training_seconds'],target_downweighted_fraction=v['adapt_meta']['target_downweighted_fraction'],parent_state_hash=seal['parent_entry']['state_hash'],candidate_state_hash=v['candidate_entry']['state_hash'],matched_seal_sha256=sha(sealp),service_hash=service.hash,scope='AUDIT_ONLY_MATCHED_PARENT_NOT_LIFECYCLE'))
 atomic(result,dict(rows=rows,seal_sha256=sha(sealp),service_open_seq=opened['seq']),exclusive=True)
 # Oracle information is created only after all legal candidates and the actual
 # reference action were sealed and Service opened. It is never fed to a gate.
 if stream=='coherent_sticky' and t in (3,4):
  with np.load(ROOT/f'data/center_{c}.npz') as z:
   a,b=spans(t)[0];clean=Segment(z['x'][a:b].copy(),z['y'][a:b].copy(),z['ts'][a:b].copy(),'update')
  obs=observed(clean,c,t,stream);oracles=[]
  for label,x,y in [('X_CLEAN_Y_OBS',clean.x,obs.y),('X_OBS_Y_CLEAN',obs.x,clean.y),('X_CLEAN_Y_CLEAN',clean.x,clean.y)]:
   ou=Segment(x.copy(),y.copy(),clean.ts.copy(),'update');start=time.perf_counter();model=adapt(parent,tensors(windows(ou)),steps=50,seed=seed+c+t*100);sync();ent=cas.put(model)
   cl=float(errors(model,xy)[0].mean());oracles.append(dict(center=c,seed=seed,stream=stream,stage=t,oracle=label,candidate_service_mse=cl,parent_service_mse=pl,delta=cl-pl,candidate_entry=ent,training_seconds=time.perf_counter()-start,scope='POSTSEAL_ORACLE_AUDIT_ONLY_EXCLUDED_FROM_SELECTION'))
  atomic(td/f'oracle_result_{t:03}.json',dict(rows=oracles,parent_entry=seal['parent_entry'],created_after_service_open_seq=opened['seq']),exclusive=True)
