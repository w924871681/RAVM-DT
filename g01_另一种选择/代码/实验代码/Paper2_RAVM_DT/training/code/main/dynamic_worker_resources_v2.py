"""No fixed 2/4-worker cap: physical resources and measured throughput govern admission."""
import os,time,psutil,subprocess,shutil
GIB=2**30
RAM_RESERVE=1536*2**20
RAM_PER_WORKER=384*2**20
VRAM_RESERVE=512*2**20
VRAM_PER_WORKER=512*2**20

def snapshot():
    vm=psutil.virtual_memory()
    s=dict(available_ram_bytes=vm.available,total_ram_bytes=vm.total,
           free_vram_bytes=0,total_vram_bytes=0,logical_cpu_count=os.cpu_count() or 1)
    if shutil.which('nvidia-smi'):
        try:
            wire=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],
                text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            free,total=map(int,wire.splitlines()[0].split(','));s.update(free_vram_bytes=free*2**20,total_vram_bytes=total*2**20)
        except (OSError,ValueError,subprocess.SubprocessError):pass
    return s

def physical_limit(s):
    ram=max(0,(s['total_ram_bytes']-RAM_RESERVE)//RAM_PER_WORKER)
    gpu=max(0,(s['total_vram_bytes']-VRAM_RESERVE)//VRAM_PER_WORKER)
    return max(1,int(min(ram,gpu)))

def memory_capacity(s,active):
    # Physical ceiling prevents the active-worker credit from self-amplifying.
    ram=max(0,(s['available_ram_bytes']+active*RAM_PER_WORKER-RAM_RESERVE)//RAM_PER_WORKER)
    gpu=max(0,(s['free_vram_bytes']+active*VRAM_PER_WORKER-VRAM_RESERVE)//VRAM_PER_WORKER)
    return int(min(physical_limit(s),ram,gpu))

class AdaptiveWorkers:
    def __init__(self,start=2,clock=time.monotonic):
        self.clock=clock;self.target=start;self.started=clock();self.done=0
        self.previous=None;self.performance_limit=None;self.history=[];self.physical_max=1
        self.retry_after=0.;self.retry_failures=0
    def allowed(self,active,completed,s=None):
        s=snapshot() if s is None else s
        self.physical_max=physical_limit(s);funded=memory_capacity(s,active)
        now=self.clock();reason='STABLE'
        if funded<self.target:
            self.target=funded;self.previous=None;self.started=now;self.done=completed;reason='MEMORY_DRAIN'
        elif self.target==0 and funded>0:
            self.target=1;self.started=now;self.done=completed;reason='MEMORY_RECOVERED'
        if self.performance_limit is not None and now>=self.retry_after:
            self.performance_limit=None;self.started=now;self.done=completed
            reason='RETRY_BASELINE_MEASUREMENT'
        elapsed=now-self.started;finished=completed-self.done
        if self.target>0 and elapsed>=30 and finished>=6:
            rate=finished/elapsed
            self.history.append(dict(workers=self.target,completed=finished,seconds=elapsed,trajectories_per_second=rate))
            if self.previous and rate<self.previous[1]*.95:
                self.target=self.previous[0];self.performance_limit=self.target;self.previous=None
                self.retry_failures+=1
                self.retry_after=now+min(1800,300*2**min(self.retry_failures-1,3))
                reason='THROUGHPUT_REGRESSION_DRAIN'
            else:
                if self.previous:
                    self.previous=None;self.performance_limit=None;self.retry_failures=0;self.retry_after=0.
                if funded>self.target and self.performance_limit is None:
                    self.previous=(self.target,rate);self.target+=1;reason='PROBE_NEXT_RESOURCE_FUNDED_WORKER'
                elif self.performance_limit is not None:
                    reason='THROUGHPUT_COOLDOWN'
            self.started=now;self.done=completed
        return min(self.target,funded),dict(**s,resource_capacity=funded,physical_resource_ceiling=self.physical_max,
            policy_target=self.target,throughput_limit=self.performance_limit,policy_reason=reason,
            throughput_history=self.history[-12:],ram_reserve_bytes=RAM_RESERVE,vram_reserve_bytes=VRAM_RESERVE,
            retry_cooldown_remaining_seconds=max(0.,self.retry_after-now),retry_failures=self.retry_failures)
