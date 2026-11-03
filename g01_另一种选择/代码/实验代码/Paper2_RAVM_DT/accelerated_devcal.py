"""Measure DevCal concurrency, reuse verified source/warmup, then run all cases.

Baseline, cached serial and cached two-process probes all use the original
trajectory, causal audit and frozen numerical comparison. Probe work is separate
from the full run. Full aggregation reruns the original audit and statistics.
"""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from accelerated_matrix import read,write
B=Path(__file__).resolve().parent

def run_jobs(commands,out):
    out.mkdir(exist_ok=False);active=[];pending=list(enumerate(commands));started=time.monotonic();last=-1e9;peak=0;low=None
    while pending or active:
        for entry in list(active):
            proc,log=entry;rc=proc.poll()
            if rc is None:continue
            log.close();active.remove(entry)
            if rc:raise RuntimeError('Development command failed; remaining children must be inspected, not duplicated')
        free=psutil.virtual_memory().available/2**30;low=free if low is None else min(low,free);rss=0
        for proc,_ in active:
            try:
                p=psutil.Process(proc.pid);rss+=sum(x.memory_info().rss for x in [p,*p.children(recursive=True)])
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        peak=max(peak,rss)
        if pending and free>=2 and time.monotonic()-last>=45:
            q=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
            if int(q.stdout.strip().splitlines()[0])>=1536:
                i,cmd=pending.pop(0);log=(out/f'worker_{i}.log').open('x',encoding='utf-8')
                proc=subprocess.Popen(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT);active.append((proc,log));last=time.monotonic()
        write(out/'PROGRESS.json',dict(elapsed_seconds=time.monotonic()-started,active=[p.pid for p,_ in active],pending=len(pending),peak_child_rss_bytes=peak,available_gib=free))
        if pending or active:time.sleep(2)
    result=dict(seconds=time.monotonic()-started,peak_child_rss_bytes=peak,minimum_available_gib=low)
    write(out/'MEASUREMENT.json',result);return result

def main(a):
    out=a.output.resolve()
    if out.exists():raise RuntimeError('Fresh output required')
    out.mkdir(parents=True)
    base=[sys.executable,'-X','utf8','-B',str(B/'reproduce_devcal.py'),'--mode','source-to-final','--source-python',str(a.source_python)]
    if a.reuse_sources:base+=['--reuse-sources',str(a.reuse_sources)]
    prep=out/'prepared';run_jobs([base+['--output',str(prep),'--prepare-only']],out/'prepare_monitor')
    if read(prep/'EXECUTION_RECEIPT.json')['status']!='PREPARED_SOURCE_AND_EIGHT_INITIALS':raise RuntimeError('Development preparation failed')
    # The prepared source is made local and exact in every probe/full worker.
    # Use its original source-training directory via a standard named pool.
    from accelerated_matrix import copy_tree
    pool=out/'source_pool'/'synthetic__GRU_1_7311';copy_tree(prep/'source_training',pool)
    base=[sys.executable,'-X','utf8','-B',str(B/'reproduce_devcal.py'),'--mode','source-to-final','--source-python',str(a.source_python),'--reuse-sources',str(pool.parent),'--reuse-initials',str(prep)]
    cases=[]
    for name,n,cache in [('baseline_one',1,False),('cached_one',1,True),('cached_two',2,True)]:
        if n>a.workers:continue
        commands=[]
        for i in range(n):
            cmd=base+['--output',str(out/(name+f'_{i}')),'--limit',str(8//n),'--task-shards',str(n),'--task-shard-index',str(i)]
            if cache:cmd+=['--read-cache']
            commands.append(cmd)
        metrics=run_jobs(commands,out/(name+'_monitor'))
        coverage=set()
        for i in range(n):
            receipt=read(out/(name+f'_{i}')/'EXECUTION_RECEIPT.json')
            if receipt['status']!='PASS_PARTIAL_DEVCAL_ENTRY':raise RuntimeError('Development benchmark failed')
            coverage.update((x['key']['center'],x['key']['seed'],x['key']['stream'],x['key']['arm']) for x in receipt['records'])
        if len(coverage)!=8:raise RuntimeError('Probe coverage differs')
        cases.append(dict(case=name,workers=n,cache=cache,**metrics))
    one=min((x for x in cases if x['workers']==1),key=lambda x:x['seconds']);best=min(cases,key=lambda x:x['seconds'])
    if best['workers']==2 and best['seconds']>one['seconds']*.9:best=one
    write(out/'BENCHMARK_RECEIPT.json',dict(status='PASS_MEASURED_DEVCAL_PROBES',cases=cases,selected=best,scientific_acceptance='Every probe passes original numerical and causal audit'))
    n=best['workers'];commands=[];shards=[]
    for i in range(n):
        shard=out/f'full_shard_{i}';shards.append(shard)
        cmd=base+['--output',str(shard),'--task-shards',str(n),'--task-shard-index',str(i)]
        if best['cache']:cmd+=['--read-cache']
        commands.append(cmd)
    run_jobs(commands,out/'full_monitor')
    if n==1:
        # This shard already ran the full original selection/table pipeline.
        receipt=read(shards[0]/'EXECUTION_RECEIPT.json')
    else:
        final=out/'combined';cmd=base+['--output',str(final),'--aggregate-from',*[str(x) for x in shards]]
        if best['cache']:cmd+=['--read-cache']
        run_jobs([cmd],out/'aggregation_monitor');receipt=read(final/'EXECUTION_RECEIPT.json')
    if receipt['status']!='PASS_FULL_DEVCAL_SOURCE_AND_LIFECYCLES':raise RuntimeError('Full DevCal acceptance failed')
    write(out/'EXECUTION_RECEIPT.json',dict(receipt,full_output=str(shards[0] if n==1 else out/'combined'),acceleration_benchmark=str(out/'BENCHMARK_RECEIPT.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-python',type=Path,required=True);p.add_argument('--reuse-sources',type=Path);p.add_argument('--workers',type=int,choices=[1,2],default=2);a=p.parse_args()
    try:main(a)
    except BaseException:
        if a.output.exists():write(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),live_children_must_not_be_duplicated=[x.pid for x in psutil.Process().children(recursive=True)],automatic_retry=False))
        raise
