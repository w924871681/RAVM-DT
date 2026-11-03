from d3_engine import *
from d3_gate import Evidence,decide,baseline_decision
from dataclasses import asdict
from d3_adapt import make_candidate
VARIANTS=()

def late_from_event(e,scope):
 d=e['data'];return LateAudit(*scope,e['stage'],e['seq'],d['mature_seq'],d['receipt_sha256'],d['parent_sha256'],d['candidate_sha256'],d['delta_late'],d['proxy_old'],d['proxy_new'],d['proxy_recent'],d['scale'])

def run_trajectory(phase,center,seed,stream,arm,run_id,progress=lambda **k:None,stop_after=None):
 td=ROOT/'science'/phase/str(center)/str(seed)/stream/arm['id'];td.mkdir(parents=True,exist_ok=True)
 gate_method=arm.get('gate_method',arm['method'])
 scope=(f'{run_id}:{phase}:{seed}:{arm["id"]}',str(center),stream)
 ledger=AuditLedger(td/'journal.jsonl');cas=CAS(td/'models');provider=CausalProvider(center,stream)
 authority=SimulatedLabelAuthority(td,ledger,cas,provider,scope)
 verifier=FeedbackVerifier(ledger,cas,authority.public_key,expected_issuer=authority.ISSUER)
 gate=OriginalCMCGate(ROOT/'runtime',ROOT/'protocol/frozen_v2_final_commit_config.yaml')
 rows=[]
 present=[t for t in range(1,25) if (td/f'result_{t:03}.json').exists()]
 if present!=list(range(1,len(present)+1)):raise RuntimeError('Noncontiguous completed results; no replay past a hole')
 if any(e['stage']>len(present)+1 for e in ledger.entries):raise RuntimeError('Future journal beyond completed prefix; no unsafe recovery')
 for t in range(1,25):
  path=td/f'result_{t:03}.json'
  if not path.exists():break
  r=read(path);sp=td/f'seal_{t:03}.json';assert r['decision_sha256']==sha(sp)
  assert r['stage']==t and r['method']==arm['method'] and r['arm']==arm['id']
  if not any(e['kind']=='STAGE_COMPLETED' and e['stage']==t for e in ledger.entries):
   ledger.append('STAGE_COMPLETED',scope,t,dict(result_sha256=sha(path),recovered_receipt_only=True))
  rows.append(r)
 completed=len(rows);provider.restore_prefix(completed)
 history=evidence_blocks(provider.initial_history())
 for t in range(1,completed+1):history.extend(evidence_blocks(provider.maintenance(t)[1]))
 parent=cas.load(rows[-1]['deployed_entry']) if rows else initial(center,seed)
 if rows:assert rows[-1]['deployed_entry']['state_hash']==state_hash(parent)
 for t in range(completed+1,25):
  if __import__('shutil').disk_usage(ROOT).free<2*1024**3:raise OSError('DISK_RESERVE_EXHAUSTED; preserve state, no scientific reduction')
  start=time.perf_counter();sp=td/f'seal_{t:03}.json';recovered_seal=sp.exists()
  if sp.exists():
   sealed=read(sp);assert sealed['center']==center and sealed['seed']==seed and sealed['stream']==stream and sealed['arm']==arm['id']
   seals=[e for e in ledger.entries if e['kind']=='DECISION_SEALED' and e['stage']==t]
   if not seals:
    assert sealed['execution_lock_sha256'] in allowed_execution_shas()
    assert ledger.event(sealed['snapshot_seq'])['kind']=='SNAPSHOT_SAVED' and ledger.event(sealed['gate_start_seq'])['kind']=='GATE_START'
    assert not any(e['stage']==t and e['kind']=='SERVICE_ROLE_OPENED' for e in ledger.entries)
    se=ledger.append('DECISION_SEALED',scope,t,dict(snapshot_seq=sealed['snapshot_seq'],decision=sealed['commit'],decision_sha256=sha(sp),recovered_durable_seal_receipt_only=True))
   else:se=seals[0]
   assert se['data']['decision_sha256']==sha(sp)
   snap=ledger.event(sealed['snapshot_seq']);parent=cas.load(sealed['parent_entry']);candidate=cas.load(sealed['candidate_entry']);scores=sealed['scores'];commit=sealed['commit']
   u,v=provider.maintenance(t);assert u.hash==sealed['update_hash'] and v.hash==sealed['verification_hash']
   recent=evidence_blocks(v)
  else:
   # No outcome exists without a seal. Partially written pre-seal attempts are not
   # silently replayed, because that would create duplicate gate frontiers.
   if any(e['stage']==t for e in ledger.entries):raise RuntimeError('UNSEALED_PARTIAL_STAGE_REQUIRES_EXPLICIT_TECHNICAL_RECOVERY; no blind restart')
   ledger.append('STAGE_STARTED',scope,t,dict(training_steps=50,seed=seed+center+t*100))
   u,v=provider.maintenance(t);recent=evidence_blocks(v);train_start=time.perf_counter()
   candidate,adapt_meta=make_candidate(parent,u,arm.get('generator','U_MSE'),seed+center+t*100);sync();train_sec=time.perf_counter()-train_start
   scores=score(candidate,parent,recent,history,v,gate);pe=rows[-1]['deployed_entry'] if rows else cas.put(parent);ce=cas.put(candidate)
   assert pe['state_hash']==state_hash(parent) and cas.verify(pe['file_sha256'])
   snap=ledger.append('SNAPSHOT_SAVED',scope,t,dict(parent_sha256=pe['file_sha256'],candidate_sha256=ce['file_sha256'],parent_state_hash=pe['state_hash'],candidate_state_hash=ce['state_hash'],old_score=scores['old_score'],new_score=scores['new_score'],recent_score=scores['recent_score'],scale=scores['scale'],history_complete=len(scores['history_blocks'])==12,history_block_count=len(scores['history_blocks']),recent_block_count=6))
   audits=tuple(late_from_event(e,scope) for e in ledger.entries if e['kind']=='LATE_PAIR_SCORED' and e['stage']<t)[-6:]
   view=verifier.snapshot(scope,t,audits=audits,current_history_complete=len(scores['history_blocks'])==12)
   reason='FIXED_ORIGINAL_OR_BASELINE';fallback=False
   if arm['method']=='JOINT':
    from f2_gate import joint_gate
    commit,joint_score=joint_gate(scores['stage3']['recent_mean'],scores['parent_recent_verification_mean_loss'],scores['stage3']['mwu_p'],arm['weight'],arm['threshold']);reason='CURRENT_MEAN_MWU_SOFT_JOINT'
   elif arm.get('fallback',False):commit=False;reason='DEV_CAL_NO_QUALIFIED_CONFIGURATION_RETAIN_ALL'
   elif arm['method'] in VARIANTS:
    outcome=decide_variant(Evidence(tuple(scores['recent_blocks']),tuple(scores['history_blocks']),scores['scale']),view,lambda:gate.evaluate(candidate,parent,recent,history).commit,verifier.view_public_key,scope,t,arm['method'],arm['threshold'],scores['stage3']['mwu_p'])
    commit=outcome.action;reason=outcome.reason;fallback=outcome.fallback
   elif gate_method in ('RAVM_D2B_V02','RAVM_DELAYED_GUARD_ONLY'):
    outcome=decide(Evidence(tuple(scores['recent_blocks']),tuple(scores['history_blocks']),scores['scale']),view,lambda:gate.evaluate(candidate,parent,recent,history).commit,verifier.view_public_key,scope,t,gate_method=='RAVM_DELAYED_GUARD_ONLY')
    commit=outcome.action;reason=outcome.reason;fallback=outcome.fallback
   elif gate_method=='RAVM_ORIGINAL':commit=scores['original_commit']
   else:
    commit=baseline_decision(arm['method'],scores['stage3'],arm['threshold'])
   sync();pre=time.perf_counter()-start
   matched_set=None
   if phase=='RootScreen':
    from d3_screen import before_seal
    matched_set=before_seal(td,ledger,cas,parent,candidate,u,center,seed,stream,t,scope)
   sealed=dict(arm_spec=arm,generator=arm.get('generator','U_MSE'),gate_method=gate_method,adapt_meta=adapt_meta,matched_set=matched_set,schema='D3_DURABLE_PRE_SERVICE_DECISION',scope=list(scope),phase=phase,center=center,seed=seed,stream=stream,stage=t,arm=arm['id'],method=arm['method'],threshold=arm['threshold'],commit=bool(commit),reason=reason,fallback=fallback,snapshot_seq=snap['seq'],gate_start_seq=view.gate_start_seq,feedback_view=asdict(view),feedback_view_public_key=verifier.view_public_key.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex(),label_authority_public_key=authority.public_hex,parent_entry=pe,candidate_entry=ce,scores=scores,update_hash=u.hash,verification_hash=v.hash,raw_sha256=sha(ROOT/f'data/center_{center}.npz'),source_sha256=SOURCE[str(seed)]['sha256'],execution_lock_sha256=execution_lock_sha(),training_steps=50,training_seed=seed+center+t*100,training_seconds=train_sec,maintenance_pre_service_seconds=pre)
   identity=atomic(sp,sealed,exclusive=True)
   se=ledger.append('DECISION_SEALED',scope,t,dict(snapshot_seq=snap['seq'],decision=bool(commit),decision_sha256=identity))
  # Include actual pre-Service persistence cost; never use outcome timing as a proxy.
  cost_path=td/f'pre_service_cost_{t:03}.json'
  if not cost_path.exists():
   if recovered_seal:raise RuntimeError('SEALED_STAGE_WITHOUT_DURABLE_COST_RECEIPT; preserve evidence; no fabricated timing')
   atomic(cost_path,dict(seconds=max(sealed['maintenance_pre_service_seconds'],time.perf_counter()-start),scope='includes snapshot, gate, seal and causal receipt persistence; excludes postseal evaluation'),exclusive=True)
  pre_service_cost=read(cost_path)['seconds']
  receipt=dict(event_seq=se['seq'],sha256=sha(sp),commit=bool(commit))
  eval_start=time.perf_counter();service,mat,proof=authority.mature(t,snap,receipt);sxy=tensors(windows(service))
  pl=float(errors(parent,sxy)[0].mean());cl=float(errors(candidate,sxy)[0].mean());sync();eval_sec=time.perf_counter()-eval_start
  late_cost=0.;late_record=None
  if gate_method in ('RAVM_D2B_V02','RAVM_DELAYED_GUARD_ONLY',*VARIANTS) and len(scores['history_blocks'])==12:
   existing=[e for e in ledger.entries if e['kind']=='LATE_PAIR_SCORED' and e['stage']==t]
   if existing:late_record=late_from_event(existing[0],scope);late_cost=existing[0]['data']['late_load_and_forward_seconds']
   else:late_record,late_cost=authority.late_score(t,snap,mat,proof,service,sealed['parent_entry'],sealed['candidate_entry'])
   assert late_record.delta_late==cl-pl,'Late scoring must agree with independent same-state post-seal evaluator'
  deployed=sealed['candidate_entry'] if commit else sealed['parent_entry'];delta=cl-pl;harm=bool(commit and delta>0)
  row=dict(phase='D3d1_'+phase if phase!='ENGINEERING_TEST' else phase,center=center,seed=seed,stream=stream,regime_audit_only=REGIMES[IDS[phase].index(center)%4],arm=arm['id'],method=arm['method'],stage=t,commit=bool(commit),original_gate_on_own_state=scores['original_commit'],reason=sealed['reason'],fallback=sealed['fallback'],parent_entry=sealed['parent_entry'],candidate_entry=sealed['candidate_entry'],deployed_entry=deployed,decision_sha256=sha(sp),service_hash=service.hash,parent_service_mse=pl,candidate_service_mse=cl,deployed_service_mse=cl if commit else pl,delta_deployed=delta if commit else 0.,harmful=harm,downside=max(delta,0.) if commit else 0.,beneficial_commit=bool(commit and delta<0),candidate_beneficial_audit_only=delta<0,maintenance_wall_seconds=pre_service_cost+late_cost,pre_service_including_seal_seconds=pre_service_cost,late_score_seconds=late_cost,post_seal_scientific_evaluation_seconds=eval_sec,training_seconds=sealed['training_seconds'],late_audit_seq=late_record.score_event_seq if late_record else None,maturity_seq=mat['seq'],label_authority_public_key=authority.public_hex,feedback_used_count=len(sealed['feedback_view']['audits']),feedback_active=not sealed['fallback'] if gate_method in ('RAVM_D2B_V02','RAVM_DELAYED_GUARD_ONLY',*VARIANTS) else False,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if DEVICE=='cuda' else None,scope='NEW_SYNTHETIC_DEVELOPMENT_ONLY')
  row['phase']=phase
  row['scope']='NEW_PROSPECTIVE_FINAL_SYNTHETIC' if phase=='NewFinal' else 'NEW_SYNTHETIC_SMOKE_DEVELOPMENT_NOT_FINAL'
  row['calibration_fallback']=bool(arm.get('fallback',False))
  if phase=='RootScreen':
   from d3_screen import after_service
   after_service(td,ledger,cas,parent,service,center,seed,stream,t,scope)
  if phase=='ENGINEERING_TEST':row['scope']='ENGINEERING_TEST_ONLY_NOT_SCIENTIFIC_DATA'
  rp=td/f'result_{t:03}.json';rid=atomic(rp,row,exclusive=True);ledger.append('STAGE_COMPLETED',scope,t,dict(result_sha256=rid))
  atomic(td/'checkpoint.json',dict(stage=t,result_sha256=rid,deployed_entry=deployed,journal_root_hash=ledger.verify()))
  rows.append(row);history.extend(recent);parent=candidate if commit else parent
  progress(stage=t,task=str(td.relative_to(ROOT)),complete_trajectory=t==24)
  if stop_after and t>=stop_after:return rows
 atomic(td/'DONE.json',dict(stages=24,result_sha256=[sha(td/f'result_{t:03}.json') for t in range(1,25)],journal_root=ledger.verify()),exclusive=not (td/'DONE.json').exists())
 return rows
