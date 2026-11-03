"""Execute the byte-identical original CMC engine.run via causal I/O adapters.

No replacement baseline optimizer, Fisher, replay algorithm or commit rule.
Only data plumbing, common M0 reuse, signed persistence and reporting are adapted.
"""
from d3_engine import *
from f2_plan import LEGACY_NAMES
import core.engine as native,inspect,threading
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
TLS=threading.local()
NATIVE_RUN=native.run
NATIVE_FISHER=native.fisher_diag
NATIVE_PROTOCOL_ATTACK=attack

def finite(x):
 if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [finite(v) for v in x]
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 if isinstance(x,np.integer):return int(x)
 return x

def fisher(model,xy,seed=0,n=32):
 start=time.perf_counter();result=NATIVE_FISHER(model,xy,seed,n);sync()
 ctx=TLS.ctx;ctx['fisher'].append(dict(seed=seed,n=min(n,len(xy[0])),role=xy.role,seconds=time.perf_counter()-start,
     input_digest=digest(xy[0].detach().cpu().numpy(),xy[1].detach().cpu().numpy()),optimizer_steps=0))
 return result

def native_initial(center,seed,backbone,H,smoke=False):
 assert backbone=='GRU' and H==1 and (ROOT/f'initial/{center}_{seed}.pt').is_file()
 return initial(center,seed)

def native_attack(seg,center,t,kind,alpha,location,seed=9417):
 assert kind==TLS.ctx['stream'] and alpha==(0. if kind=='clean' else .1) and location=='both' and seed==9417
 if kind=='coherent_sticky':
  # The generator is the exact frozen D3f2R coherent-sticky implementation.
  from d3_engine import _env
  out,mask=_env['coherent_sticky_attack'](seg,center,t)
  return out,mask,np.full(len(mask),-1,dtype=np.int64),'coherent_sticky'
 return NATIVE_PROTOCOL_ATTACK(seg,center,t,kind,alpha,location,seed)

def persist(ctx,row):
 t=row['stage'];td=ctx['td'];rp=td/f'result_{t:03}.json'
 assert not rp.exists(),'No completed legacy trajectory replay'
 seal=read(td/f'seal_{t:03}.json');pl=row['current_service_mse'];cl=row['candidate_service_mse'];delta=cl-pl
 assert bool(row['harmful_commit'])==bool(row['commit'] and delta>0)
 assert abs(row['downside']-(max(delta,0.) if row['commit'] else 0.))<=1e-9
 extra_init=sum(x['seconds'] for x in ctx['fisher'] if x['seed']==ctx['seed']) if t==1 and ctx['arm']['id']=='EWC' else 0.
 common=dict(phase=ctx['phase'],center=ctx['center'],seed=ctx['seed'],stream=ctx['stream'],arm=ctx['arm']['id'],method=ctx['arm']['method'],stage=t,
  regime_audit_only=REGIMES[IDS[ctx['phase']].index(ctx['center'])%4],commit=bool(row['commit']),
  parent_entry=seal['parent_entry'],candidate_entry=seal['candidate_entry'],deployed_entry=seal['candidate_entry' if row['commit'] else 'parent_entry'],
  decision_sha256=sha(td/f'seal_{t:03}.json'),service_hash=row['service_data_hash'],parent_service_mse=pl,candidate_service_mse=cl,
  deployed_service_mse=row['deployed_service_mse'],delta_deployed=delta if row['commit'] else 0.,harmful=bool(row['commit'] and delta>0),
  downside=max(delta,0.) if row['commit'] else 0.,beneficial_commit=bool(row['commit'] and delta<0),candidate_beneficial_audit_only=delta<0,
  maintenance_wall_seconds=row['maintenance_time_s']+extra_init,training_seconds=row['candidate_update_time_s'],late_score_seconds=0.,
  post_seal_scientific_evaluation_seconds=max(0.,time.perf_counter()-ctx['service_eval_started'][t]),gpu_peak_allocated_bytes=row['peak_gpu_memory_bytes'],
  native_reported_downside=row['downside'],native_reported_delta=row['delta_srv'],ewc_initial_fisher_seconds=extra_init,
  fisher_backward_examples=row['number_of_fisher_backward_passes'],fisher_optimizer_steps=0,reason='NATIVE_ORIGINAL_CMC_'+ctx['arm']['legacy_native_method'],
  fallback=ctx['arm']['id']=='VALIDATION_COMMIT' and ctx['cfg']['validation_retain_all'],calibration_fallback=ctx['arm']['id']=='VALIDATION_COMMIT' and ctx['cfg']['validation_retain_all'],
  feedback_active=False,feedback_used_count=0,original_gate_on_own_state=None,scope='NEW_PROSPECTIVE_'+ctx['phase'].upper(),native_original_row=finite(row))
 assert common['deployed_service_mse']==(cl if row['commit'] else pl)
 identity=atomic(rp,common,exclusive=True);ctx['ledger'].append('STAGE_COMPLETED',ctx['scope'],t,dict(result_sha256=identity))
 atomic(td/'checkpoint.json',dict(stage=t,deployed_entry=common['deployed_entry'],result_sha256=identity,journal_root_hash=ctx['ledger'].verify()))
 ctx['persisted']+=1;ctx['progress'](stage=t,complete_trajectory=t==24)

class NativeTimeline(Timeline):
 def __init__(self,center):
  super().__init__(center)
  self.meta={**self.meta,'center_type':self.meta.get('base_generator',{}).get('center_type','SOURCE_NATIVE'),
             'drift_regime':self.meta['regime_audit_only']}

 def maintenance(self,t):
  frame=inspect.currentframe().f_back;ctx=TLS.ctx
  if t>1 and frame.f_code is NATIVE_RUN.__code__:
   previous=frame.f_locals['rows'][-1]
   persist(ctx,previous)
  return super().maintenance(t)

 def record_decision(self,t,commit):
  frame=inspect.currentframe().f_back;assert frame.f_code is NATIVE_RUN.__code__
  d=frame.f_locals;ctx=TLS.ctx;td=ctx['td'];ledger=ctx['ledger'];parent=d['parent'];candidate=d['candidate']
  pe=ctx['cas'].put(parent);ce=ctx['cas'].put(candidate)
  if t>1:assert pe['state_hash']==read(td/f'result_{t-1:03}.json')['deployed_entry']['state_hash']
  snap=ledger.append('SNAPSHOT_SAVED',ctx['scope'],t,dict(parent_sha256=pe['file_sha256'],candidate_sha256=ce['file_sha256']))
  gate=ledger.append('NATIVE_GATE_COMPLETED',ctx['scope'],t,dict(method=d['method'],commit=bool(commit)))
  replay=list(d['replay']);train_xy=d.get('train_xy')
  sealed=dict(schema='NATIVE_CMC_PRE_SERVICE_DECISION',phase=ctx['phase'],center=ctx['center'],seed=ctx['seed'],stream=ctx['stream'],stage=t,
    arm=ctx['arm']['id'],method=ctx['arm']['method'],native_method=d['method'],arm_spec=ctx['arm'],configuration=ctx['cfg'],
    commit=bool(commit),parent_entry=pe,candidate_entry=ce,snapshot_seq=snap['seq'],gate_start_seq=gate['seq'],
    update_hash=d['u'].hash,verification_hash=d['v'].hash,training_steps=d['updates'],training_seed=ctx['seed']+ctx['center']+t*100,
    training_role=None if d['updates']==0 else d.get('train_xy',d['xy']).role,
    training_window_count=0 if d['updates']==0 else len(d.get('train_xy',d['xy'])[0]),
    validation_mean_delta=finite(d['ur']),past_replay_hashes=[s.hash for s in replay],
    fisher_calls_before_seal=list(ctx['fisher']),frozen_engine_sha256=ctx['engine_sha'],source_sha256=SOURCE[str(ctx['seed'])]['sha256'],
    raw_sha256=sha(ROOT/f'data/center_{ctx["center"]}.npz'),execution_lock_sha256=execution_lock_sha(),
    causal_roles='Native adapt/fisher enforce Update/warmup/replay; Service opens only after this seal')
  h=atomic(td/f'seal_{t:03}.json',sealed,exclusive=True)
  ledger.append('DECISION_SEALED',ctx['scope'],t,dict(decision_sha256=h,decision=bool(commit)))
  super().record_decision(t,commit)

 def service(self,t):
  ctx=TLS.ctx;ledger=ctx['ledger'];sp=ctx['td']/f'seal_{t:03}.json';assert sp.exists()
  se=next(e for e in ledger.entries if e['stage']==t and e['kind']=='DECISION_SEALED');assert se['data']['decision_sha256']==sha(sp)
  result=super().service(t)
  op=ledger.append('SERVICE_ROLE_OPENED',ctx['scope'],t,dict(seal_seq=se['seq'],service_sha256=result.hash))
  proof=dict(scope=ctx['scope'],stage=t,seal_seq=se['seq'],service_open_seq=op['seq'],service_sha256=result.hash,source_claim='SIMULATED_SOURCE_ATTESTED_NOT_FIELD')
  ledger.append('LABELS_MATURED',ctx['scope'],t,dict(receipt=proof,signature=ctx['key'].sign(canonical(proof)).hex(),public_key=ctx['public_key']))
  ctx['service_eval_started'][t]=time.perf_counter()
  return result

native.fisher_diag=fisher;native.initial_model=native_initial;native.attack=native_attack;native.Timeline=NativeTimeline

def run_legacy(phase,center,seed,stream,arm,run_id,progress=lambda **k:None):
 td=ROOT/f'science/{phase}/{center}/{seed}/{stream}/{arm["id"]}'
 assert not td.exists(),'Legacy native execution cannot replay a partially consumed scientific lifecycle'
 td.mkdir(parents=True)
 cfg={**native.DEFAULT,**read(ROOT/'reference/LEGACY_BASELINE_IMPLEMENTATION_IDENTITY.json')['frozen_config']}
 assert cfg['ewc_lambda']==10. and cfg['validation_retain_all'] is True
 key=Ed25519PrivateKey.generate();(td/'PRIVATE_SIMULATION_LABEL_KEY.bin').write_bytes(key.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption()))
 ctx=dict(phase=phase,center=center,seed=seed,stream=stream,arm=arm,td=td,cfg=cfg,cas=CAS(td/'models'),ledger=AuditLedger(td/'journal.jsonl'),
  key=key,public_key=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex(),
  scope=(run_id+':'+phase+':'+str(seed)+':'+arm['id'],str(center),stream),fisher=[],service_eval_started={},persisted=0,progress=progress,
  engine_sha=sha(ROOT/'runtime/core/engine.py'))
 TLS.ctx=ctx
 rr=NATIVE_RUN(center,seed,arm['legacy_native_method'],kind=stream,alpha=(0. if stream=='clean' else .1),location='both',cfg=cfg,stages=24,backbone='GRU',H=1,smoke=False,run_id=run_id)
 persist(ctx,rr[-1]);assert ctx['persisted']==24
 if arm['id']=='EWC':assert len(ctx['fisher'])==25 and sum(e['n'] for e in ctx['fisher'])==800
 atomic(td/'NATIVE_EXACTNESS_RECEIPT.json',dict(status='PASS',native_run_function_byte_identical=True,native_engine_sha256=ctx['engine_sha'],cfg=cfg,
  fisher_backward_examples=sum(e['n'] for e in ctx['fisher']),fisher_optimizer_steps=0,steps_per_stage=0 if arm['id']=='FROZEN' else 50,
  common_measurement='Lifecycle MSE means and harm identical; downside computed from float64 scalar mean difference uniformly across 12 methods; native float32 delta retained, discrepancy bounded 1e-9',fisher_calls=ctx['fisher']),exclusive=True)
 atomic(td/'DONE.json',dict(stages=24,result_sha256=[sha(td/f'result_{t:03}.json') for t in range(1,25)],journal_root=ctx['ledger'].verify()),exclusive=True)
 return [read(td/f'result_{t:03}.json') for t in range(1,25)]

def audit_legacy(td):
 td=Path(td);done=read(td/'DONE.json');ledger=AuditLedger(td/'journal.jsonl');assert ledger.verify()==done['journal_root'];cas=CAS(td/'models');prior=None
 receipt=read(td/'NATIVE_EXACTNESS_RECEIPT.json');assert receipt['status']=='PASS' and receipt['native_engine_sha256']==sha(ROOT/'runtime/core/engine.py')
 expectedcfg={**native.DEFAULT,**read(ROOT/'reference/LEGACY_BASELINE_IMPLEMENTATION_IDENTITY.json')['frozen_config']};assert receipt['cfg']==expectedcfg
 for t in range(1,25):
  s=read(td/f'seal_{t:03}.json');r=read(td/f'result_{t:03}.json');assert sha(td/f'result_{t:03}.json')==done['result_sha256'][t-1]
  assert r['decision_sha256']==sha(td/f'seal_{t:03}.json') and s['configuration']==expectedcfg and s['execution_lock_sha256'] in allowed_execution_shas()
  if prior:assert s['parent_entry']['state_hash']==prior['state_hash']
  else:assert s['parent_entry']['state_hash']==read(ROOT/f'initial/{s["center"]}_{s["seed"]}.json')['state_hash']
  assert s['training_steps']==(0 if s['arm']=='FROZEN' else 50)
  assert s['commit']==(False if s['arm'] in ('FROZEN','VALIDATION_COMMIT') else True)
  events={e['kind']:e for e in ledger.entries if e['stage']==t}
  assert events['SNAPSHOT_SAVED']['seq']<events['NATIVE_GATE_COMPLETED']['seq']<events['DECISION_SEALED']['seq']<events['SERVICE_ROLE_OPENED']['seq']<events['LABELS_MATURED']['seq']<events['STAGE_COMPLETED']['seq']
  mat=events['LABELS_MATURED'];Ed25519PublicKey.from_public_bytes(bytes.fromhex(mat['data']['public_key'])).verify(bytes.fromhex(mat['data']['signature']),canonical(mat['data']['receipt']))
  parent=cas.load(s['parent_entry']);candidate=cas.load(s['candidate_entry'])
  timeline=Timeline(s['center']);
  for j in range(1,t):timeline.record_decision(j,False)
  u,v=timeline.maintenance(t);u=observed(u,s['center'],t,s['stream']);v=observed(v,s['center'],t,s['stream'])
  assert u.hash==s['update_hash'] and v.hash==s['verification_hash']
  if s['training_steps']:assert s['training_role'] in ('update','replay_training')
  if s['arm']=='VALIDATION_COMMIT':
   xy=tensors(windows(v));score=float((errors(candidate,xy)[0]-errors(parent,xy)[0]).mean());assert score==s['validation_mean_delta']
  with np.load(ROOT/f'data/center_{s["center"]}.npz') as z:
   a,b=spans(t)[2];service=Segment(z['x'][a:b].copy(),z['y'][a:b].copy(),z['ts'][a:b].copy(),'service')
  assert service.hash==r['service_hash'];xy=tensors(windows(service));pl=float(errors(parent,xy)[0].mean());cl=float(errors(candidate,xy)[0].mean())
  assert (pl,cl)==(r['parent_service_mse'],r['candidate_service_mse']);assert r['downside']==(max(cl-pl,0) if r['commit'] else 0)
  prior=r['deployed_entry'];assert prior==s['candidate_entry' if s['commit'] else 'parent_entry']
 return dict(status='VALIDATED',rows=24,native_original_engine=True,native_engine_sha256=receipt['native_engine_sha256'],optimizer_steps_by_auditor=0)
