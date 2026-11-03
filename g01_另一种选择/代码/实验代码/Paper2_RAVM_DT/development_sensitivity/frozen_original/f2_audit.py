"""Separate-process full numerical and lineage audit; zero SGD."""
from d3_common import *
from d3_audit import audit_phase
from d3_parallel import execute
from f2_plan import arms,seeds,protocol
import itertools,threading
def main(phase):
 protect_originals();verify_science_lock();started=time.time();p=protocol()['proposed_phases'][phase]
 for path,r in read(ROOT/'ORIGINAL_INPUT_IDENTITIES.json').items():assert sha(ROOT/path)==r['sha256'],path
 jobs=sorted((ROOT/'science'/phase).glob('*/*/*/*'));assert len(jobs)==p['expected_trajectories']
 mutex=threading.RLock();count=0
 def job(td):
  nonlocal count
  records,receipt=audit_phase(phase,24,True,only_td=td)
  assert receipt['status']=='VALIDATED'
  c=int(td.parts[-4]);s=int(td.parts[-3]);stream=td.parts[-2];arm=td.name
  assert all((r['center'],r['seed'],r['stream'],r['arm'])==(c,s,stream,arm) for r in records)
  with mutex:count+=24
  return dict(trajectory=str(td.relative_to(ROOT)),receipt=receipt,done_sha256=sha(td/'DONE.json'),initial_parent_state_hash=read(td/'seal_001.json')['parent_entry']['state_hash'])
 def tick(z):
  with mutex:
   state=read(ROOT/'progress/progress_state.json');state.update(status='AUDITING',work='Audit',audit_done=count,active_workers=z['active_workers'],eta_seconds=z['eta_seconds'],updated_unix=time.time());atomic(ROOT/'progress/progress_state.json',state)
 result=execute(jobs,job,phase+'_INDEPENDENT_AUDIT',tick)
 expected=set(itertools.product(read(ROOT/'RUN_MANIFEST.json')['centers'][phase],seeds(phase),['clean','mixed','coherent_sticky'],[a['id'] for a in arms(phase)]))
 actual={(int(p.parts[-4]),int(p.parts[-3]),p.parts[-2],p.name) for p in jobs};assert actual==expected
 initial={}
 for r in result:
  p=Path(r['trajectory']);key=p.parts[-4:-2];h=r['initial_parent_state_hash']
  if key in initial:assert initial[key]==h
  initial[key]=h
 accounting=[json.loads(line) for line in (ROOT/'process/SGD_STEP_ACCOUNTING.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
 current=[r for r in accounting if r['phase']==phase];assert all(r['error'] is None for r in current)
 assert sum(r['actual_steps'] for r in current if r['kind']=='warmup')==p_count(phase)*50
 pp=protocol()['proposed_phases'][phase]
 assert sum(r['actual_steps'] for r in current if r['kind']=='candidate')==pp['expected_stage_rows']*50
 assert count==pp['expected_stage_rows']
 for path,r in read(ROOT/'ORIGINAL_INPUT_IDENTITIES.json').items():assert sha(ROOT/path)==r['sha256'],path
 receipt=dict(status='VALIDATED',phase=phase,trajectories=len(jobs),stage_rows=count,initial_center_seed_pairs=len(initial),all_stage_forwards_and_scores_recomputed=True,full24_lineage_verified=True,all_seal_before_service=True,gate_inputs_authenticated=True,source_immutability=True,optimizer_steps_by_auditor=0,wall_seconds=time.time()-started,pid=os.getpid(),scope='Independent local process using authenticated library; synthetic, not field confirmation',execution_lock_sha256=execution_lock_sha(),trajectories_evidence=result)
 atomic(ROOT/f'results/{phase}/INDEPENDENT_AUDIT.json',receipt,exclusive=True)
 print(phase,'VALIDATED',count,flush=True)
def p_count(phase):return len(read(ROOT/'RUN_MANIFEST.json')['centers'][phase])*len(seeds(phase))
if __name__=='__main__':main(sys.argv[1])
