"""Separate auditor process; original checks inherited, extended for candidate training."""
from d3_engine import *
from d3_parallel import execute
from d3_adapt import make_candidate,batch_hash,target_weights
from d3_gate import Evidence,research_decision,baseline_decision
from d2c_core import VerifiedFeedbackView,view_message
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
def audit_td(arg):
 phase,td=arg;td=Path(td);ledger=AuditLedger(td/'journal.jsonl');ledger.verify();cas=CAS(td/'models');done=read(td/'DONE.json')
 assert done['stages']==24 and done['result_sha256']==[sha(td/f'result_{t:03}.json') for t in range(1,25)] and done['journal_root']==ledger.verify()
 from d3_runner import arms
 spec=next(a for a in arms(phase) if a['id']==td.name);timeline=Timeline(int(td.parents[2].name));history=evidence_blocks(timeline.initial_history());records=[];local=[];oracles=[];prior=None;weights=set()
 gate=OriginalCMCGate(ROOT/'runtime',ROOT/'protocol/frozen_v2_final_commit_config.yaml')
 for t in range(1,25):
  s=read(td/f'seal_{t:03}.json');r=read(td/f'result_{t:03}.json');scope=tuple(s['scope']);c=s['center'];seed=s['seed'];stream=s['stream']
  assert s['execution_lock_sha256']==execution_lock_sha() and s['method']==spec['method'] and s['generator']==spec['generator'] and s['threshold']==spec['threshold']
  assert s['source_sha256']==SOURCE[str(seed)]['sha256'] and s['raw_sha256']==sha(ROOT/f'data/center_{c}.npz') and s['training_steps']==50 and s['training_seed']==seed+c+t*100
  assert r['decision_sha256']==sha(td/f'seal_{t:03}.json') and r['commit']==s['commit']
  if prior:assert prior==s['parent_entry']
  else:
   side=read(ROOT/f'initial/{c}_{seed}.json');assert side['state_hash']==s['parent_entry']['state_hash'] and side['sha256']==sha(ROOT/f'initial/{c}_{seed}.pt') and side['source_sha256']==SOURCE[str(seed)]['sha256'];assert side['warmup_steps']==50 and side['warmup_seed']==seed+c and side['warmup_hash']==Timeline(c).warmup().hash
  pm=cas.load(s['parent_entry']);cm=cas.load(s['candidate_entry']);weights.update((s['parent_entry']['file_sha256'],s['candidate_entry']['file_sha256']))
  u,v=timeline.maintenance(t);u=observed(u,c,t,stream);v=observed(v,c,t,stream);recent=evidence_blocks(v)
  assert u.hash==s['update_hash'] and v.hash==s['verification_hash'];sc=score(cm,pm,recent,history,v,gate);assert sc==s['scores']
  meta=s['adapt_meta'];w,_=target_weights(u);assert meta['sampled_indices_sha256']==batch_hash(len(w),seed+c+t*100) and meta['update_hash']==u.hash and meta['steps']==50
  if s['generator']=='U_LABEL_WEIGHT':assert meta['target_weights_sha256']==hashlib.sha256(w.tobytes()).hexdigest() and meta['target_offset']==96
  gs=ledger.event(s['gate_start_seq']);snap=ledger.event(s['snapshot_seq']);se=next(e for e in ledger.entries if e['kind']=='DECISION_SEALED' and e['stage']==t);op=next(e for e in ledger.entries if e['kind']=='SERVICE_ROLE_OPENED' and e['stage']==t);mat=ledger.event(r['maturity_seq'])
  assert snap['seq']<gs['seq']<se['seq']<op['seq']<mat['seq'] and se['data']['decision_sha256']==sha(td/f'seal_{t:03}.json')
  pub=Ed25519PublicKey.from_public_bytes(bytes.fromhex(r['label_authority_public_key']));pub.verify(bytes.fromhex(mat['data']['signature']),canonical(mat['data']['receipt']))
  vj=s['feedback_view'];aa=tuple(LateAudit(**a) for a in vj['audits']);view=VerifiedFeedbackView(**{**vj,'audits':aa});Ed25519PublicKey.from_public_bytes(bytes.fromhex(s['feedback_view_public_key'])).verify(bytes.fromhex(view.attestation_signature),view_message(view))
  if view.attested:
   verifier=FeedbackVerifier(ledger,cas,pub,expected_issuer='SIMULATED_CAUSAL_LABEL_PROVIDER_ONLY')
   for a in aa:verifier._verify_one(a,scope,t,gs['seq'])
  gm=spec['gate_method']
  if spec['fallback']:expected=False
  elif gm=='RAVM_ORIGINAL' or (gm in ('RAVM_D2B_V02','RAVM_DELAYED_GUARD_ONLY') and s['fallback']):expected=sc['original_commit']
  elif gm in ('RAVM_D2B_V02','RAVM_DELAYED_GUARD_ONLY'):assert view.attested;expected=research_decision(Evidence(tuple(sc['recent_blocks']),tuple(sc['history_blocks']),sc['scale']),aa,t,gs['seq'],gm=='RAVM_DELAYED_GUARD_ONLY')
  else:expected=baseline_decision(gm,sc['stage3'],spec['threshold'])
  assert s['commit']==expected
  timeline.record_decision(t,s['commit']);service=timeline.service(t);xy=tensors(windows(service));pl=float(errors(pm,xy)[0].mean());cl=float(errors(cm,xy)[0].mean());assert pl==r['parent_service_mse'] and cl==r['candidate_service_mse'] and service.hash==r['service_hash']
  assert r['harmful']==bool(s['commit'] and cl>pl) and r['downside']==(max(cl-pl,0.) if s['commit'] else 0.)
  assert r['deployed_entry']==s['candidate_entry' if s['commit'] else 'parent_entry'];prior=r['deployed_entry']
  if phase=='RootScreen' and t in (2,3,4,5,9,10,19,20):
   ls=read(td/f'matched_seal_{t:03}.json');lr=read(td/f'matched_result_{t:03}.json');ev=next(e for e in ledger.entries if e['kind']=='MATCHED_CANDIDATE_SET_SEALED' and e['stage']==t)
   assert ev['seq']<se['seq'] and ev['data']['seal_sha256']==sha(td/f'matched_seal_{t:03}.json')==lr['seal_sha256'] and ls['parent_entry']['state_hash']==state_hash(pm)
   assert {q['method'] for q in lr['rows']}=={'U_MSE','U_HUBER','U_LABEL_WEIGHT'}
   for q in lr['rows']:
    ent=ls['candidates'][q['method']];mm=cas.load(ent['candidate_entry']);loss=float(errors(mm,xy)[0].mean());assert loss==q['candidate_service_mse'] and pl==q['parent_service_mse'] and loss-pl==q['delta'];assert ent['adapt_meta']['sampled_indices_sha256']==meta['sampled_indices_sha256'];local.append(q)
   oracle=td/f'oracle_result_{t:03}.json'
   if oracle.exists():
    oo=read(oracle);assert oo['created_after_service_open_seq']==op['seq']
    for q in oo['rows']:
     mm=cas.load(q['candidate_entry']);assert float(errors(mm,xy)[0].mean())==q['candidate_service_mse'];oracles.append(q)
  records.append(r);history.extend(recent)
 return dict(records=records,local=local,oracles=oracles,weight_files=len(weights),initial=(c,seed,read(td/'seal_001.json')['parent_entry']['state_hash']))
def main(phase):
 started=time.monotonic();verify_science_lock()
 from d3_pipeline import verify_phase_permission
 data=read(ROOT/f'data/{phase}_IDENTITY.json');assert data['execution_lock_sha256']==execution_lock_sha() and data['permission_sha256']==verify_phase_permission(phase)
 role=AuditLedger(ROOT/'process/DATA_ROLE_EVENTS.jsonl');role.verify();beg=role.event(data['generation_start_seq']);end=role.event(data['generation_end_seq']);assert beg['seq']<end['seq'] and beg['data']['phase']==end['data']['phase']==phase and beg['data']['permission_sha256']==data['permission_sha256']
 assert {v['center'] for v in data['rows']}==set(IDS[phase])
 for v in data['rows']:assert sha(ROOT/v['path'])==v['sha256']
 paths=sorted((ROOT/'science'/phase).glob('*/*/*/*/DONE.json'));jobs=[(phase,p.parent) for p in paths]
 def tick(z):atomic(ROOT/f'progress/AUDIT_{phase}_PROGRESS.json',dict(status='AUDITING',phase=phase,verified_stage_rows=z['completed']*24,expected_stage_rows=len(jobs)*24,percent=z['percent'],elapsed_seconds=z['elapsed_seconds'],eta_seconds=z['eta_seconds'],active_workers=z['active_workers'],updated_unix=time.time()))
 parts=execute(jobs,audit_td,'AUDIT_'+phase,tick);records=[r for p in parts for r in p['records']];local=[r for p in parts for r in p['local']];oracles=[r for p in parts for r in p['oracles']]
 from d3_pipeline import expected_counts,seeds
 from d3_runner import arms
 import itertools
 expected=set(itertools.product(IDS[phase],seeds(phase),STREAMS,[a['id'] for a in arms(phase)],range(1,25)));assert {(r['center'],r['seed'],r['stream'],r['arm'],r['stage']) for r in records}==expected and len(records)==len(expected)
 initial={}
 for p in parts:
  c,s,h=p['initial'];assert (c,s) not in initial or initial[c,s]==h;initial[c,s]=h
 for v in read(ROOT/'PREFLIGHT_ASSETS.json')['source_asset_files']:assert sha(v['path'])==v['sha256']
 from d3_results import csvwrite,phase_summary
 dest=ROOT/'results'/phase;dest.mkdir(exist_ok=True)
 if phase=='RootScreen':
  assert len(local)==576 and len(oracles)==48;csvwrite(dest/'MATCHED_CANDIDATE_RECORDS.csv',local);csvwrite(dest/'ORACLE_AUDIT_ONLY.csv',oracles)
  csvwrite(dest/'REFERENCE_STAGE_RECORDS.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in records])
 else:phase_summary(phase,records)
 atomic(dest/'INDEPENDENT_AUDIT.json',dict(status='VALIDATED',scope='D3d1_'+phase,verified_stage_rows=len(records),verified_candidate_adaptations=len(local),oracle_audit_rows=len(oracles),causality_pass=True,full_matrix_pass=True,source_asset_immutability_pass=True,real_weights_verified=sum(p['weight_files'] for p in parts),scientific_protocol_sha256=sha(ROOT/'EXPERIMENT_LOCK.json'),execution_lock_sha256=execution_lock_sha(),phase_results_sha256={p.name:sha(p) for p in dest.glob('*.csv')},independent_process_pid=os.getpid(),elapsed_seconds=time.monotonic()-started,scope_limit='NEW_SYNTHETIC_DEVELOPMENT_NOT_FIELD_OR_FINAL'))
 tick(dict(completed=len(jobs),percent=100,elapsed_seconds=time.monotonic()-started,eta_seconds=0,active_workers=0));q=read(ROOT/f'progress/AUDIT_{phase}_PROGRESS.json');q['status']='VALIDATED';atomic(ROOT/f'progress/AUDIT_{phase}_PROGRESS.json',q)
 print('VALIDATED',phase,len(records),len(local))
if __name__=='__main__':
 protect_originals();main(sys.argv[1])
