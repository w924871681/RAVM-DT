"""Independent audit: ledger, sealed identities, lineage, real weights and Service.
Runs only on finished phase records, never exposes any Service to a gate.
"""
from d3_engine import *
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from d2c_core import VerifiedFeedbackView,view_message
from d3_gate import Evidence,research_decision
VARIANTS=()
def audit_phase(phase,expected_rows,verify_service=True,only_td=None):
 records=[];checked=set();ncausal=0;decisions=0;inputs={};input_cache={};initial_states={}
 gate=OriginalCMCGate(ROOT/'runtime',ROOT/'protocol/frozen_v2_final_commit_config.yaml')
 for td in ([Path(only_td)] if only_td else sorted((ROOT/'science'/phase).glob('*/*/*/*'))):
  if not td.is_dir():continue
  ledger=AuditLedger(td/'journal.jsonl');ledger.verify();cas=CAS(td/'models');prior=None
  done=read(td/'DONE.json');assert done['stages']==24 and done['result_sha256']==[sha(td/f'result_{j:03}.json') for j in range(1,25)] and done['journal_root']==ledger.verify(),'Complete lifecycle receipt mismatch'
  for t in range(1,25):
   sp=td/f'seal_{t:03}.json';rp=td/f'result_{t:03}.json';s=read(sp);r=read(rp)
   assert s['stage']==r['stage']==t and r['decision_sha256']==sha(sp)
   assert s['commit']==r['commit'] and s['training_steps']==50
   assert s['training_seed']==s['seed']+s['center']+t*100
   assert s['source_sha256']==SOURCE[str(s['seed'])]['sha256']
   from d3_runner import arms
   spec_arm=next(a for a in arms(phase) if a['id']==s['arm']) if phase!='ENGINEERING_TEST' else dict(method=s['method'],threshold=s['threshold'])
   assert s['method']==spec_arm['method'] and s['threshold']==spec_arm['threshold'] and s['arm_spec']==spec_arm
   if s['method']=='JOINT':
    z=s['scores'];d=z['stage3']['recent_mean'];p=z['stage3']['mwu_p'];scale=max(z['parent_recent_verification_mean_loss'],1e-12)
    expected=bool(d<0 and d+spec_arm['weight']*scale*(p-.2)<=-s['threshold'])
    assert s['commit']==expected and s['reason']=='CURRENT_MEAN_MWU_SOFT_JOINT' and not s['feedback_view']['audits']
   if t==1:
    initial_key=s['center'],s['seed'];initial_hash=s['parent_entry']['state_hash']
    if initial_key in initial_states:assert initial_states[initial_key]==initial_hash,'Initial-model fairness failed'
    initial_states[initial_key]=initial_hash
   if phase!='ENGINEERING_TEST':
    ip=ROOT/f'initial/{s["center"]}_{s["seed"]}.pt';side=read(ip.with_suffix('.json'))
    assert side['sha256']==sha(ip) and side['state_hash']==initial_hash
    assert side['source_sha256']==SOURCE[str(s['seed'])]['sha256'] and side['warmup_steps']==50 and side['warmup_seed']==s['seed']+s['center']
    assert side['warmup_hash']==Timeline(s['center']).warmup().hash
    im=Forecast().to(DEVICE);im.load_state_dict(torch.load(ip,map_location=DEVICE,weights_only=True));assert state_hash(im)==initial_hash;del im
   assert s['execution_lock_sha256'] in allowed_execution_shas()
   assert s['raw_sha256']==sha(ROOT/f'data/center_{s["center"]}.npz')
   if prior:assert s['parent_entry']==prior
   assert r['parent_entry']==s['parent_entry'] and r['candidate_entry']==s['candidate_entry']
   assert r['deployed_entry']==s['candidate_entry' if s['commit'] else 'parent_entry']
   prior=r['deployed_entry']
   snap=ledger.event(s['snapshot_seq']);gs=ledger.event(s['gate_start_seq']);se=next(e for e in ledger.entries if e['stage']==t and e['kind']=='DECISION_SEALED');op=next(e for e in ledger.entries if e['stage']==t and e['kind']=='SERVICE_ROLE_OPENED');mat=ledger.event(r['maturity_seq'])
   assert snap['seq']<gs['seq']<se['seq']<op['seq']<mat['seq']
   assert se['data']['decision_sha256']==sha(sp) and op['data']['seal_seq']==se['seq'] and op['data']['service_sha256']==r['service_hash']
   pub=Ed25519PublicKey.from_public_bytes(bytes.fromhex(r['label_authority_public_key']));pub.verify(bytes.fromhex(mat['data']['signature']),canonical(mat['data']['receipt']))
   assert mat['data']['receipt']['source_claim']=='SIMULATED_SOURCE_ATTESTED_NOT_FIELD'
   v=s['feedback_view'];audits=tuple(LateAudit(**a) for a in v['audits']);view=VerifiedFeedbackView(**{**v,'audits':audits})
   Ed25519PublicKey.from_public_bytes(bytes.fromhex(s['feedback_view_public_key'])).verify(bytes.fromhex(v['attestation_signature']),view_message(view))
   verifier=FeedbackVerifier(ledger,cas,pub,expected_issuer='SIMULATED_CAUSAL_LABEL_PROVIDER_ONLY')
   if view.attested:
    for a in audits:verifier._verify_one(a,tuple(s['scope']),t,gs['seq']);ncausal+=1
    assert tuple(a.score_event_seq for a in audits)==tuple(e['seq'] for e in sorted([e for e in ledger.entries[:gs['seq']-1] if e['kind']=='LATE_PAIR_SCORED' and e['stage']<t],key=lambda e:e['stage'])[-6:])
   sc=s['scores'];rec=np.array(sc['recent_blocks']);hist=np.array(sc['history_blocks']);med=float(np.median(hist));mad=float(np.median(abs(hist-med)))
   assert float(np.median(rec))==sc['recent_score'] and med+.5*mad==sc['history_score']
   anchor=bool(sc['recent_score']<=-2.8018237766243602e-5 and sc['history_score']<=0);assert anchor==sc['original_commit']
   if s['method']=='RAVM_ORIGINAL':assert s['commit']==anchor
   if s['method'] in ('RAVM_D2B_V02','RAVM_DELAYED_GUARD_ONLY') and s['fallback']:assert s['commit']==anchor
   if s['method'] in ('RAVM_D2B_V02','RAVM_DELAYED_GUARD_ONLY') and not s['fallback']:
    assert view.attested
    expected=research_decision(Evidence(tuple(sc['recent_blocks']),tuple(sc['history_blocks']),sc['scale']),audits,t,gs['seq'],s['method']=='RAVM_DELAYED_GUARD_ONLY')
    assert expected==s['commit']
   if s['method'] in VARIANTS:
    out=decide_variant(Evidence(tuple(rec),tuple(hist),sc['scale']),view,lambda:anchor,Ed25519PublicKey.from_public_bytes(bytes.fromhex(s['feedback_view_public_key'])),tuple(s['scope']),t,s['method'],s['threshold'],sc['stage3']['mwu_p'])
    assert (out.action,out.fallback,out.reason)==(s['commit'],s['fallback'],s['reason'])
    if not s['fallback']:assert independent_expected(Evidence(tuple(rec),tuple(hist),sc['scale']),audits,s['method'])==s['commit'],'Independent D3c formula mismatch'
   assert all(k not in s['scores'] for k in ('service_mse','pollution_mask','regime_truth'))
   if s['method'] in ('DUAL_MEAN','RECENT_MEAN','MWU_MEAN'):
    spec=read(ROOT/'DEVCAL_SELECTED_LOCK.json')['baseline_selection'][s['method']] if phase in ('Smoke','NewFinal') else {}
    fallback=spec.get('fallback',False)
    expected=sc['stage3']['recent_mean']<=-s['threshold']
    if s['method']=='DUAL_MEAN':expected=expected and sc['stage3']['history_mean']<=0
    if s['method']=='MWU_MEAN':expected=expected and sc['stage3']['mwu_p']<.2
    assert s['commit']==(bool(expected) and not fallback)
   ik=(s['center'],s['stream'],t)
   values=s['update_hash'],s['verification_hash'],s['raw_sha256']
   if ik in inputs:assert inputs[ik]==values,'Exogenous input fairness failed'
   inputs[ik]=values
   for role in ('parent_entry','candidate_entry'):
    ent=s[role];key=str(td),ent['file_sha256']
    if key not in checked:
     model=cas.load(ent);del model;checked.add(key)
   if verify_service:
    with np.load(ROOT/f'data/center_{s["center"]}.npz') as z:
     a,b=spans(t)[2];seg=Segment(z['x'][a:b].copy(),z['y'][a:b].copy(),z['ts'][a:b].copy(),'service')
    assert seg.hash==r['service_hash'];xy=tensors(windows(seg));pm=cas.load(s['parent_entry']);cm=cas.load(s['candidate_entry'])
    pl=float(errors(pm,xy)[0].mean());cl=float(errors(cm,xy)[0].mean())
    assert pl==r['parent_service_mse'] and cl==r['candidate_service_mse'],'Post-seal real weight forward mismatch'
    key=s['center'],s['stream']
    if key not in input_cache:
     timeline=Timeline(s['center']);hist=evidence_blocks(timeline.initial_history());cache={}
     for j in range(1,25):
      uu,vv=timeline.maintenance(j);uu=observed(uu,s['center'],j,s['stream']);vv=observed(vv,s['center'],j,s['stream'])
      rb=evidence_blocks(vv);cache[j]=(uu,vv,rb,list(hist[-12:]));hist.extend(rb);timeline.record_decision(j,False)
     input_cache[key]=cache
    uu,vv,rb,hb=input_cache[key][t]
    assert uu.hash==s['update_hash'] and vv.hash==s['verification_hash']
    from d3_adapt import batch_hash
    a=s['adapt_meta'];assert a['generator']=='U_MSE' and a['steps']==50 and a['lr']==.01 and a['batch']==64 and a['grad_clip']==1.
    assert a['sampled_indices_sha256']==batch_hash(len(windows(uu)[0]),s['training_seed'])
    recovered=score(cm,pm,rb,hb,vv,gate)
    assert recovered==sc,'Full authenticated per-block/Stage3 score reconstruction mismatch'
    del pm,cm
   delta=r['candidate_service_mse']-r['parent_service_mse'];assert r['deployed_service_mse']==(r['candidate_service_mse'] if r['commit'] else r['parent_service_mse'])
   assert r['harmful']==bool(r['commit'] and delta>0) and r['downside']==(max(delta,0.) if r['commit'] else 0.)
   records.append(r);decisions+=1
 assert len(records)==expected_rows,(len(records),expected_rows)
 assert all(r['center'] in IDS[phase] and r['seed'] in read(ROOT/'RUN_MANIFEST.json')['seeds'][phase] and r['stream'] in STREAMS for r in records)
 if phase!='ENGINEERING_TEST' and only_td is None:
  from d3_runner import arms
  import itertools
  expected=set(itertools.product(IDS[phase],read(ROOT/'RUN_MANIFEST.json')['seeds'][phase],STREAMS,[a['id'] for a in arms(phase)],range(1,25)))
  actual={(r['center'],r['seed'],r['stream'],r['arm'],r['stage']) for r in records}
  assert actual==expected and len(actual)==len(records),'Scientific matrix identity/coverage mismatch'
 audit=dict(status='VALIDATED',scope='D3f2_'+phase,audit_pass=True,causality_pass=True,method_and_data_identity_pass=True,baseline_configurations_same_as_devcal_lock=True,science_lock_sha256=sha(ROOT/'EXPERIMENT_LOCK.json'),devcal_lock_sha256=sha(ROOT/'DEVCAL_SELECTED_LOCK.json') if phase in ('Smoke','NewFinal') else None,passed=True,phase=phase,rows=len(records),real_weight_files_verified=len(checked),past_feedback_chains_verified=ncausal,decision_causal_chains=decisions,post_seal_service_forwards_reverified=bool(verify_service),toy=phase=='ENGINEERING_TEST',expected_rows=expected_rows)
 if phase in ('Smoke','NewFinal'):
  sel=read(ROOT/'SELECTION_SEAL_RECEIPT.json');data=read(ROOT/f'data/{phase}_IDENTITY.json')
  assert data['devcal_lock_sha256']==sel['selection_sha256']==sha(ROOT/'DEVCAL_SELECTED_LOCK.json')
  role=AuditLedger(ROOT/'process/DATA_ROLE_EVENTS.jsonl');role.verify()
  frozen=role.event(sel['role_event_seq']);beg=role.event(data['generation_start_seq']);end=role.event(data['generation_end_seq'])
  assert frozen['kind']=='DEVCAL_SELECTION_DURABLY_FROZEN' and frozen['data']['selection_sha256']==sel['selection_sha256']
  assert frozen['seq']<beg['seq']<end['seq'] and beg['kind']=='INPUT_GENERATION_STARTED' and end['kind']=='INPUT_GENERATION_COMPLETED'
  assert beg['data']['phase']==end['data']['phase']==phase and beg['data']['selection_sha256']==sel['selection_sha256']
  assert data['created_unix']>sel['sealed_unix'] and sel['phase_inputs_existed_before_seal'] is False
  audit['confirm_inputs_generated_after_devcal_lock']=True
 # The separate auditor publishes only after global and immutability checks.
 return records,audit
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('phase');ap.add_argument('--expected',type=int,required=True);a=ap.parse_args();print(audit_phase(a.phase,a.expected)[1])
