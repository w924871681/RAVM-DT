"""No fixed worker-count ceiling: hardware/RAM/VRAM and measured throughput only."""
from r_common import *
import concurrent.futures as cf,psutil,importlib.util

def execute(jobs,worker,label,on_tick=None,cpu=False):
 spec=importlib.util.spec_from_file_location('_policy_'+label,ROOT/'dynamic_worker_resources_v2.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
 policy.RAM_RESERVE=2*1024**3
 if cpu:
  policy.RAM_PER_WORKER=32*1024**2;policy.VRAM_RESERVE=0
  def snap():
   v=psutil.virtual_memory();return dict(available_ram_bytes=v.available,total_ram_bytes=v.total,logical_cpu_count=os.cpu_count() or 1,free_vram_bytes=0,total_vram_bytes=0)
  policy.snapshot=snap
  policy.physical_limit=lambda s:max(1,min(s['logical_cpu_count'],int((s['total_ram_bytes']-policy.RAM_RESERVE)//policy.RAM_PER_WORKER)))
  policy.memory_capacity=lambda s,a:max(0,min(policy.physical_limit(s),int((s['available_ram_bytes']+a*policy.RAM_PER_WORKER-policy.RAM_RESERVE)//policy.RAM_PER_WORKER)+1))
 else:
  original_capacity=policy.memory_capacity
  policy.memory_capacity=lambda s,a:0 if s['available_ram_bytes']<2*1024**3 else min(policy.physical_limit(s),max(1,original_capacity(s,a)))
 s=policy.snapshot();assert cpu or s['total_vram_bytes']>0
 started=time.monotonic();pending=iter(enumerate(jobs));active={};results={};exhausted=False;last=0;cap=0
 controller=policy.AdaptiveWorkers(start=1 if not cpu else min(4,policy.memory_capacity(s,0)))
 with cf.ThreadPoolExecutor(max_workers=policy.physical_limit(s),thread_name_prefix=label) as pool:
  while not exhausted or active:
   now=time.monotonic()
   if now-last>=3:
    cap,res=controller.allowed(len(active),len(results));last=now
    atomic(ROOT/'progress/WORKER_STATUS.json',dict(mode=label,kind='CPU_FILE' if cpu else 'GPU_SCIENCE',active_workers=len(active),admission_capacity=cap,maximum=controller.physical_max,resources=res,ram_admission_min_bytes=2*1024**3,no_fixed_worker_count_cap=True,updated_unix=time.time()))
   while not exhausted and len(active)<cap:
    if psutil.virtual_memory().available<2*1024**3:break
    try:i,job=next(pending)
    except StopIteration:exhausted=True;break
    active[pool.submit(worker,job)]=i
   if active:
    done,_=cf.wait(active,timeout=.5,return_when=cf.FIRST_COMPLETED)
    for future in done:results[active.pop(future)]=future.result()
   else:time.sleep(1)
   elapsed=time.monotonic()-started
   z=dict(completed=len(results),total=len(jobs),active_workers=len(active),elapsed_seconds=elapsed,eta_seconds=elapsed/len(results)*(len(jobs)-len(results)) if len(results)>=3 and (active or psutil.virtual_memory().available>=2*1024**3) else None)
   atomic(ROOT/f'progress/{label}_PROGRESS.json',dict(**z,status='DONE' if exhausted and not active else 'RUNNING',updated_unix=time.time()))
   if on_tick:on_tick(z)
 atomic(ROOT/'progress/WORKER_STATUS.json',dict(mode=label+'_COMPLETE',active_workers=0,admission_capacity=0,no_fixed_worker_count_cap=True,updated_unix=time.time()))
 return [results[i] for i in range(len(jobs))]
