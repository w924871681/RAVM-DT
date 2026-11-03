"""Shared dynamically admitted executor, used for every computation and audit stage."""
from d3_common import *
import concurrent.futures as cf
from dynamic_worker_resources_v2 import AdaptiveWorkers,snapshot,physical_limit
def execute(jobs,worker,label,on_tick=None):
 jobs=list(jobs);pending=list(enumerate(jobs));active={};out={};controller=AdaptiveWorkers(start=1);start=time.monotonic();last=0;cap=0
 h=snapshot()
 if not h['total_vram_bytes']:raise RuntimeError('GPU telemetry missing; no speculative concurrency')
 with cf.ThreadPoolExecutor(max_workers=physical_limit(h),thread_name_prefix='D3d1-'+label) as pool:
  while pending or active:
   if time.monotonic()-last>=5:
    cap,res=controller.allowed(len(active),len(out));last=time.monotonic()
    atomic(ROOT/'progress/WORKER_STATUS.json',dict(mode=label,active_workers=len(active),admission_capacity=cap,maximum=controller.physical_max,resources=res,updated_unix=time.time()))
   while pending and len(active)<cap:
    i,j=pending.pop(0);active[pool.submit(worker,j)]=i
   if active:
    done,_=cf.wait(active,timeout=1,return_when=cf.FIRST_COMPLETED)
    for f in done:out[active.pop(f)]=f.result()
   else:time.sleep(2)
   elapsed=time.monotonic()-start;eta=elapsed/len(out)*(len(jobs)-len(out)) if len(out)>=3 else None
   z=dict(completed=len(out),total=len(jobs),elapsed_seconds=elapsed,eta_seconds=eta,active_workers=len(active),active_indices=list(active.values()),percent=len(out)/len(jobs)*100 if jobs else 100)
   atomic(ROOT/f'progress/{label}_PROGRESS.json',dict(**z,status='RUNNING' if pending or active else 'DONE',updated_unix=time.time()))
   if on_tick:on_tick(z)
 atomic(ROOT/'progress/WORKER_STATUS.json',dict(mode=label+'_COMPLETE',active_workers=0,admission_capacity=0,maximum=controller.physical_max,updated_unix=time.time()))
 return [out[i] for i in range(len(jobs))]
