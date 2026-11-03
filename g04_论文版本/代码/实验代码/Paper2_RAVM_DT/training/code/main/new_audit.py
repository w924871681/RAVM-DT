from d3_engine import *
from d3_audit import audit_phase
from legacy_adapter import audit_legacy
from f2_plan import jobs,LEGACY_IDS
from r_parallel import execute
from r_common import guard,state
import threading

def main(phase):
 guard();torch.set_num_threads(1);verify_science_lock();started=time.time()
 def forbidden(*a,**k):raise PermissionError('Auditor cannot run optimizer')
 torch.optim.SGD.step=forbidden
 from correction_integrity import verify_reuse
 verify_reuse()
 matrix=jobs(phase);paths=[ROOT/f'science/{phase}/{c}/{s}/{stream}/{a["id"]}' for c,s,stream,a in matrix]
 actual=set((ROOT/f'science/{phase}').glob('*/*/*/*'));assert set(paths)==actual
 done=0;mutex=threading.RLock()
 def work(td):
  nonlocal done
  reuse=read(ROOT/'REUSE_PROVENANCE.json')['trajectories']
  expected_lock=read(ROOT/'CORRECTION_PROTOCOL.json')['original_execution_sha256'] if td.relative_to(ROOT).as_posix() in reuse else execution_lock_sha()
  assert all(read(td/f'seal_{t:03}.json')['execution_lock_sha256']==expected_lock for t in range(1,25))
  if td.name in LEGACY_IDS:record=audit_legacy(td)
  else:_,record=audit_phase(phase,24,True,only_td=td)
  assert record['status']=='VALIDATED'
  with mutex:done+=24
  return dict(path=td.relative_to(ROOT).as_posix(),receipt=record,done_sha256=sha(td/'DONE.json'),initial_state_hash=read(td/'seal_001.json')['parent_entry']['state_hash'])
 state(status='AUDITING',work='IndependentFullAudit',audit_done=0,audit_total=len(paths)*24)
 result=execute(paths,work,phase+'_INDEPENDENT_AUDIT',lambda q:state(audit_done=done,active_workers=q['active_workers'],eta_seconds=q['eta_seconds']))
 initial={}
 for x in result:
  p=Path(x['path']);key=tuple(p.parts[2:4]);h=x['initial_state_hash']
  if key in initial:assert initial[key]==h
  initial[key]=h
 assert len(initial)==(16 if phase=='Smoke' else 96)
 events=[json.loads(v) for v in (ROOT/f'process/{phase}_SGD_CALL_ACCOUNTING.jsonl').read_text().splitlines()]
 begins={e['call_id'] for e in events if e['event']=='BEGIN'};ends=[e for e in events if e['event']=='DONE']
 assert begins=={e['call_id'] for e in ends} and len(ends)==len(begins)
 assert sum(e['steps'] for e in ends)==(394400 if phase=='Smoke' else 3806400)
 ledger=AuditLedger(ROOT/'process/DATA_ROLE_EVENTS.jsonl');ledger.verify();data=read(ROOT/f'data/{phase}_IDENTITY.json');frozen=read(ROOT/'ROLE_FREEZE_RECEIPT.json')
 assert frozen['event_seq']<data['generation_start_seq']<data['generation_end_seq']
 if phase=='NewFinal':assert read(ROOT/'CORRECTION_FREEZE_RECEIPT.json')['event_seq']<data['generation_start_seq']
 verify_science_lock();atomic(ROOT/f'results/{phase}/INDEPENDENT_AUDIT.json',dict(status='VALIDATED',trajectories=len(paths),stage_rows=done,initial_center_seed_pairs=len(initial),optimizer_steps_by_auditor=0,
  all_service_and_gate_scores_recomputed=True,all_lineages_and_seals_verified=True,shared_m0_fairness=True,legacy_native_compatibility_validated=True,wall_seconds=time.time()-started,trajectories_evidence=result),exclusive=True)
 state(work='ScientificDecision',audit_done=done,eta_seconds=0)

if __name__=='__main__':main(sys.argv[1])
