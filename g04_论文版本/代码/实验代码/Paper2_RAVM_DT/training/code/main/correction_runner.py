"""Execute the 24 input-corrected legacy Smoke trajectories; keep 312 source trajectories."""
from r_common import *
from correction_meter import RecoveryMeter,load_accounting,check_instrumentation
from correction_integrity import verify_reuse
import threading,traceback

def main():
 guard();freeze_check();assert read(ROOT/'INPUT_CONTRACT_PREFLIGHT.json')['status']=='PASS'
 assert not (ROOT/'process/CORRECTION_TRAINING_STARTED.json').exists()
 verify_reuse();check_instrumentation()
 import new_runner as original
 torch=original.torch;torch.set_num_threads(1);original.verify_phase_permission('Smoke')
 calls,counts=load_accounting(ROOT/'process/Smoke_SGD_CALL_ACCOUNTING.jsonl');assert sum(counts.values())==370400
 meter=RecoveryMeter(torch,'Smoke',calls,counts)
 # Tighten the original accounting guard to exactly the authorized additional 24,000 steps.
 meter.total_limit=394400;meter.limits['candidate']=393600
 tasks=read(ROOT/'CORRECTION_TASKS.json');keys={(x['center'],x['seed'],x['stream'],x['arm']) for x in tasks}
 matrix=[j for j in original.jobs('Smoke') if (j[0],j[1],j[2],j[3]['id']) in keys];assert len(matrix)==24
 def td(job):c,s,stream,a=job;return ROOT/f'science/Smoke/{c}/{s}/{stream}/{a["id"]}'
 assert all(not td(j).exists() for j in matrix)
 started=time.time();mutex=threading.RLock();abort=threading.Event();completed=312;stages=7488
 atomic(ROOT/'process/CORRECTION_TRAINING_STARTED.json',dict(at_unix=started,budget=24000,reused_steps=370400),exclusive=True)
 def publish(**kw):
  with mutex:
   assert sum(meter.done.values())-370400<=24000
   state(status='RUNNING',phase='Smoke',work='Lifecycles',trajectories_done=completed,stage_rows_done=stages,
    actual_warmup_steps=meter.done['warmup'],actual_candidate_steps=meter.done['candidate'],actual_sgd_steps=sum(meter.done.values()),
    remaining_sgd_budget=404000-sum(meter.done.values()),correction_actual_sgd_steps=sum(meter.done.values())-370400,error=None,**kw)
 def work(job):
  nonlocal completed,stages
  assert not abort.is_set() and not td(job).exists()
  c,s,stream,arm=job
  def progressed(**kw):
   nonlocal stages
   with mutex:stages+=1;publish()
  try:
   with meter.scope('candidate',dict(center=c,seed=s,stream=stream,arm=arm['id'])):
    rows=original.run_legacy('Smoke',c,s,stream,arm,read(ROOT/'RUN_MANIFEST.json')['run_id'],progress=progressed)
   assert len(rows)==24
   with mutex:completed+=1;publish()
  except BaseException:abort.set();raise
 publish(active_workers=0,eta_seconds=None)
 original.execute(matrix,work,'Smoke_LIFECYCLES',lambda q:publish(active_workers=q['active_workers'],eta_seconds=q['eta_seconds']))
 _,durable=load_accounting(ROOT/'process/Smoke_SGD_CALL_ACCOUNTING.jsonl')
 assert durable==meter.done and completed==336 and stages==8064 and sum(durable.values())==394400
 verify_reuse();freeze_check()
 atomic(ROOT/'process/Smoke_TRAINING_RECEIPT.json',dict(status='COMPLETE',trajectories=336,stage_rows=8064,actual_warmup_steps=800,
  actual_candidate_steps=393600,actual_total_steps=394400,nominal_upper_bound=404000,correction_optimizer_steps=24000,
  reused_trajectories=312,new_trajectories=24,failed_original_steps_retained_in_cost_accounting=394400,
  combined_smoke_execution_steps=418400,fisher_backward_example_upper=6400,fisher_optimizer_steps=0,source_pretraining_steps=0,
  wall_seconds=time.time()-started,scope='Corrected composite Smoke; reused development data, not a new independent sample',at_unix=time.time()),exclusive=True)
 atomic(ROOT/'COST_ACCOUNTING.json',dict(status='CORRECTION_TRAINING_COMPLETE',failed_round=394400,retained=370400,excluded=24000,
  new_correction_budget=24000,new_correction_actual=24000,corrected_composite_steps=394400,combined_smoke_actual=418400))
 state(status='AWAITING_AUDIT',work='Audit',active_workers=0,eta_seconds=None)

if __name__=='__main__':
 try:main()
 except BaseException:atomic(ROOT/'process/CORRECTION_TRAINING_FAILURE.json',dict(error=traceback.format_exc(),at_unix=time.time(),automatic_retry=False));raise
