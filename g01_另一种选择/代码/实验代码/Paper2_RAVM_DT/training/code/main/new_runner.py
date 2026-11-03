from d3_engine import *
from d3_trajectory import run_trajectory
from legacy_adapter import run_legacy
from f2_plan import jobs,LEGACY_IDS,verify_phase_permission
from r_parallel import execute
from r_common import guard,state
import contextlib,threading,traceback

class Meter:
 def __init__(self,phase):
  self.phase=phase;self.mutex=threading.RLock();self.tls=threading.local();self.done={'warmup':0,'candidate':0};self.reserved=self.done.copy();self.used=set()
  self.path=ROOT/f'process/{phase}_SGD_CALL_ACCOUNTING.jsonl';assert not self.path.exists(),'No unsafe automatic optimizer replay'
  self.limits={'warmup':800 if phase=='Smoke' else 4800,'candidate':403200 if phase=='Smoke' else 4147200}
  self.total_limit=404000 if phase=='Smoke' else 4152000;self.original=torch.optim.SGD.step;owner=self
  def step(optimizer,*a,**kw):
   kind=owner.tls.kind
   with owner.mutex:
    if owner.tls.count%50==0:
     identity={**owner.tls.identity,'phase':phase,'stage':owner.tls.count//50+1} if kind=='candidate' else {**owner.tls.identity,'phase':phase}
     key=json.dumps(dict(kind=kind,identity=identity),sort_keys=True);assert key not in owner.used
     assert owner.reserved[kind]+50<=owner.limits[kind] and sum(owner.reserved.values())+50<=owner.total_limit
     owner.used.add(key);owner.reserved[kind]+=50;owner.tls.key=key;owner.append(dict(event='BEGIN',call_id=key,kind=kind,reserved_steps=50,identity=identity))
   answer=owner.original(optimizer,*a,**kw)
   with owner.mutex:
    owner.tls.count+=1;owner.done[kind]+=1
    if owner.tls.count%50==0:owner.append(dict(event='DONE',call_id=owner.tls.key,kind=kind,steps=50))
   return answer
  torch.optim.SGD.step=step
 def append(self,event):
  with self.path.open('a',encoding='utf8') as f:f.write(json.dumps(dict(**event,at_unix=time.time()))+'\n');f.flush();os.fsync(f.fileno())
 @contextlib.contextmanager
 def scope(self,kind,identity):
  self.tls.kind=kind;self.tls.identity=identity;self.tls.count=0
  try:yield
  finally:self.tls.kind=None

def main(phase):
 guard();protect_originals();torch.set_num_threads(1);verify_phase_permission(phase);started=time.time();matrix=jobs(phase);m=read(ROOT/'RUN_MANIFEST.json')
 assert len(matrix)==(336 if phase=='Smoke' else 3456)
 meter=Meter(phase);mutex=threading.RLock();stage_count=0;completed=0;m0=0;abort=threading.Event()
 state(status='GENERATING',phase=phase,work='NewData',phase_started_unix=started,trajectories_done=0,stage_rows_done=0,audit_done=0,audit_total=len(matrix)*24,
       phase_trajectories_total=len(matrix),phase_stage_rows_total=len(matrix)*24,phase_m0_total=len(m['centers'][phase])*len(m['seeds'][phase]),m0_created=0,actual_sgd_steps=0,phase_sgd_budget=meter.total_limit)
 generate(phase)
 def publish(**kw):
  with mutex:state(trajectories_done=completed,stage_rows_done=stage_count,m0_created=m0,actual_sgd_steps=sum(meter.done.values()),actual_warmup_steps=meter.done['warmup'],actual_candidate_steps=meter.done['candidate'],remaining_sgd_budget=meter.total_limit-sum(meter.done.values()),**kw)
 def init(pair):
  nonlocal m0
  c,s=pair
  with meter.scope('warmup',dict(center=c,seed=s)):initial(c,s)
  with mutex:m0+=1;publish()
 publish(status='RUNNING',work='SharedM0')
 execute([(c,s) for c in IDS[phase] for s in m['seeds'][phase]],init,phase+'_M0',lambda q:publish(eta_seconds=q['eta_seconds'],active_workers=q['active_workers']))
 atomic(ROOT/f'initial/{phase}_M0_IDENTITY_LOCK.json',dict(rows={f'{c}_{s}':read(ROOT/f'initial/{c}_{s}.json') for c in IDS[phase] for s in m['seeds'][phase]},shared_across_all_methods_and_streams=True),exclusive=True)
 def work(job):
  nonlocal stage_count,completed
  if abort.is_set():raise RuntimeError('Peer failure, no new scientific work')
  c,s,stream,arm=job
  def progressed(**kw):
   nonlocal stage_count
   with mutex:stage_count+=1;publish()
  try:
   with meter.scope('candidate',dict(center=c,seed=s,stream=stream,arm=arm['id'])):
    fn=run_legacy if arm['id'] in LEGACY_IDS else run_trajectory
    result=fn(phase,c,s,stream,arm,m['run_id'],progress=progressed)
   assert len(result)==24
   with mutex:completed+=1;publish()
  except BaseException:abort.set();raise
 publish(work='Lifecycles')
 execute(matrix,work,phase+'_LIFECYCLES',lambda q:publish(active_workers=q['active_workers'],eta_seconds=q['eta_seconds'],work_elapsed_seconds=q['elapsed_seconds']))
 expected=394400 if phase=='Smoke' else 3806400
 assert completed==len(matrix) and stage_count==len(matrix)*24 and sum(meter.done.values())==expected
 atomic(ROOT/f'process/{phase}_TRAINING_RECEIPT.json',dict(status='COMPLETE',trajectories=completed,stage_rows=stage_count,actual_warmup_steps=meter.done['warmup'],actual_candidate_steps=meter.done['candidate'],actual_total_steps=expected,nominal_upper_bound=meter.total_limit,
  fisher_backward_example_upper=6400 if phase=='Smoke' else 230400,fisher_optimizer_steps=0,source_pretraining_steps=0,wall_seconds=time.time()-started,at_unix=time.time()),exclusive=True)
 publish(status='AWAITING_AUDIT',work='Audit',active_workers=0,eta_seconds=None)

if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException:atomic(ROOT/f'process/{sys.argv[1]}_TRAINING_FAILURE.json',dict(error=traceback.format_exc(),at_unix=time.time(),automatic_scientific_retry=False));raise
