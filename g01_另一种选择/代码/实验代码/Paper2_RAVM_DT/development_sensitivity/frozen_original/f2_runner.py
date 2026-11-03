"""One phase of actual independent 24-stage lifecycles, with hard SGD accounting."""
from d3_engine import *
from d3_trajectory import run_trajectory
from d3_parallel import execute
from f2_plan import arms,seeds,verify_phase_permission,protocol
import threading,contextlib,traceback

class Meter:
 def __init__(self,phase):
  self.phase=phase;self.local=threading.local();self.mutex=threading.RLock();self.done=dict(warmup=0,candidate=0);self.attempt=dict(warmup=0,candidate=0);self.original=torch.optim.SGD.step
  self.prior={k:sum(read(p)['actual_'+k+'_steps'] for p in (ROOT/'process').glob('*_TRAINING_RECEIPT.json')) for k in self.done}
  pp=protocol()['proposed_phases'][phase];self.limits=dict(warmup=pp['new_centers']*len(seeds(phase))*50,candidate=pp['expected_stage_rows']*50)
  owner=self
  def observed(opt,*a,**kw):
   kind=getattr(owner.local,'kind',None);assert kind in owner.done,'Unscoped SGD forbidden'
   with owner.mutex:
    assert owner.attempt[kind]<owner.limits[kind],'PHASE_SGD_BUDGET_EXHAUSTED'
    assert sum(owner.prior.values())+sum(owner.attempt.values())<3286800,'TOTAL_SGD_BUDGET_EXHAUSTED'
    owner.attempt[kind]+=1
   ans=owner.original(opt,*a,**kw)
   with owner.mutex:owner.done[kind]+=1;owner.local.completed+=1
   return ans
  torch.optim.SGD.step=observed
 @contextlib.contextmanager
 def scope(self,kind,identity):
  self.local.kind=kind;self.local.completed=0;failure=None
  try:yield
  except BaseException:failure=traceback.format_exc();raise
  finally:
   with self.mutex:
    with (ROOT/'process/SGD_STEP_ACCOUNTING.jsonl').open('a',encoding='utf8') as f:
     f.write(json.dumps(dict(phase=self.phase,kind=kind,identity=identity,actual_steps=self.local.completed,error=failure,at_unix=time.time()))+'\n');f.flush();os.fsync(f.fileno())
   self.local.kind=None

def main(phase):
 protect_originals();verify_science_lock();verify_phase_permission(phase)
 assert not (ROOT/'science'/phase).exists(),'No automatic scientific replay'
 started=time.time();mutex=threading.RLock();abort=threading.Event();m=read(ROOT/'RUN_MANIFEST.json');p=protocol()['proposed_phases'][phase];matrix=arms(phase)
 assert len(matrix)==p['arms'];state=read(ROOT/'progress/progress_state.json');completed=0;stages=0;m0=0
 state.update(status='GENERATING',phase=phase,work='NewData',phase_started_unix=started,trajectories_done=0,phase_trajectories_total=p['expected_trajectories'],stage_rows_done=0,phase_stage_rows_total=p['expected_stage_rows'],audit_done=0,audit_total=p['expected_stage_rows'],eta_seconds=None,active_unit=[])
 atomic(ROOT/'progress/progress_state.json',state);generate(phase);meter=Meter(phase)
 def publish(**kw):
  with mutex:
   state.update(kw,m0_created=m0,phase_m0_total=len(IDS[phase])*len(seeds(phase)),trajectories_done=completed,stage_rows_done=stages,actual_warmup_steps=meter.prior['warmup']+meter.done['warmup'],actual_candidate_steps=meter.prior['candidate']+meter.done['candidate'],actual_sgd_steps=sum(meter.prior.values())+sum(meter.done.values()),phase_actual_sgd_steps=sum(meter.done.values()),updated_unix=time.time());atomic(ROOT/'progress/progress_state.json',state)
 def init(job):
  nonlocal m0
  if abort.is_set():raise RuntimeError('Peer failed')
  c,s=job
  try:
   with meter.scope('warmup',dict(center=c,seed=s)):model=initial(c,s)
   assert state_hash(model)==read((ROOT/f'initial/{c}_{s}.pt').with_suffix('.json'))['state_hash']
   with mutex:m0+=1;publish()
  except BaseException:abort.set();raise
 def tick(z):publish(active_workers=z['active_workers'],eta_seconds=z['eta_seconds'],work_elapsed_seconds=z['elapsed_seconds'])
 publish(status='RUNNING',work='SharedM0');execute([(c,s) for c in IDS[phase] for s in seeds(phase)],init,phase+'_M0',tick)
 manifest={f'{c}_{s}':dict(checkpoint_sha256=sha(ROOT/f'initial/{c}_{s}.pt'),**read(ROOT/f'initial/{c}_{s}.json')) for c in IDS[phase] for s in seeds(phase)}
 atomic(ROOT/f'initial/{phase}_M0_IDENTITY_LOCK.json',dict(rows=manifest,shared_across_streams_and_arms=True,execution_lock_sha256=execution_lock_sha()),exclusive=True)
 jobs=[(c,s,stream,a) for c in IDS[phase] for s in seeds(phase) for stream in STREAMS for a in matrix]
 def work(job):
  nonlocal stages,completed
  c,s,stream,a=job
  if abort.is_set():raise RuntimeError('Peer failed')
  def progressed(**kw):
   nonlocal stages
   with mutex:stages+=1;publish()
  try:
   with meter.scope('candidate',dict(center=c,seed=s,stream=stream,arm=a['id'])):
    rr=run_trajectory(phase,c,s,stream,a,m['run_id'],progress=progressed)
   assert len(rr)==24
   with mutex:completed+=1;publish()
   return dict(center=c,seed=s,stream=stream,arm=a['id'],stages=24)
  except BaseException:abort.set();raise
 def ticks(z):
  publish(active_workers=z['active_workers'],eta_seconds=z['eta_seconds'],work_elapsed_seconds=z['elapsed_seconds'],active_unit=[dict(center=jobs[i][0],seed=jobs[i][1],stream=jobs[i][2],arm=jobs[i][3]['id']) for i in z['active_indices']])
 publish(work='Lifecycles');execute(jobs,work,phase+'_LIFECYCLES',ticks)
 assert completed==p['expected_trajectories'] and stages==p['expected_stage_rows'] and meter.done['candidate']==p['expected_stage_rows']*50 and meter.done['warmup']==len(manifest)*50
 atomic(ROOT/f'process/{phase}_TRAINING_RECEIPT.json',dict(status='COMPLETE',phase=phase,wall_seconds=time.time()-started,trajectories=completed,stage_rows=stages,actual_warmup_steps=meter.done['warmup'],actual_candidate_steps=meter.done['candidate'],attempted=meter.attempt,pid=os.getpid(),replayed=False),exclusive=True)
 publish(status='AWAITING_INDEPENDENT_AUDIT',work='Audit',active_workers=0,active_unit=[],eta_seconds=None)
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException:
  atomic(ROOT/f'process/{sys.argv[1]}_TRAINING_FAILURE.json',dict(error=traceback.format_exc(),at_unix=time.time(),automatic_retry=False));raise
