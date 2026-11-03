"""Disjoint, process-isolated trajectory batches with verified consolidation.

The original per-trajectory implementation stays sequential and unchanged.
Every batch has its own mutable control files, model store and RNG process.
The parent uses only lightweight libraries and samples the complete process tree.
"""
import argparse,csv,hashlib,itertools,json,os,shutil,sqlite3,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from compare_outputs import compare_records,load_actual,reference_records
from execution_reuse import read,sha,copy_verified
B=Path(__file__).resolve().parent
REFERENCE_CACHE_ROOT=None
REFERENCE_INDEX={}
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf-8')
    # Windows readers may briefly hold the destination without delete sharing.
    # Retry only replacement, never an experiment or a numerical operation.
    for attempt in range(101):
        try:os.replace(tmp,p);break
        except PermissionError:
            if attempt==100:raise
            time.sleep(.05)
def key(k):return (k['group'],int(k['center']),int(k['seed']),k['stream'],k['arm'])
def specs():return {p.stem:read(p) for p in sorted((B/'training/groups').glob('*.json'))}
def tasks(ss):return [dict(group=s['group'],center=c,seed=se,stream=st,arm=a['id']) for s in sorted(ss.values(),key=lambda s:(bool(s.get('cost_only')),s['group'])) for c,se,st,a in itertools.product(s['centers'],s['seeds'],s['stream_ids'],s['arms'])]
def td(root,k,ss):return root/'groups'/k['group']/'science'/ss[k['group']]['phase']/str(k['center'])/str(k['seed'])/k['stream']/k['arm']
def materialize(source,dest):
    """Immutable completed evidence: native hardlinks or verified private copy.

    Hardlinks have no dependency on the old pathname. No directory links are
    created, and the donor's files are never modified after handoff.
    """
    if source.is_symlink():raise RuntimeError('Linked source rejected')
    dest.parent.mkdir(parents=True,exist_ok=True)
    before=sha(source)
    if dest.exists():
        if sha(dest)!=before:raise RuntimeError('Conflicting completed evidence: '+str(dest))
    else:
        try:os.link(source,dest)
        except OSError:shutil.copy2(source,dest)
    if sha(dest)!=before:raise RuntimeError('Materialization hash mismatch')
    return dict(file=str(dest),sha256=before,bytes=dest.stat().st_size)
def copy_tree(source,dest):
    files=[]
    for p in sorted(source.rglob('*')):
        if p.is_symlink():raise RuntimeError('Directory links are prohibited')
        if p.is_file():files.append(materialize(p,dest/p.relative_to(source)))
    return files
def comparison(root,keys,ss,receipt_root=None):
    wanted={key(k):k for k in keys};seen=set();records=[]
    for group in sorted({k['group'] for k in keys}):
        if REFERENCE_CACHE_ROOT is not None:
            if group not in REFERENCE_INDEX:
                path=REFERENCE_CACHE_ROOT/(group+'.sqlite');path.parent.mkdir(parents=True,exist_ok=True)
                conn=sqlite3.connect(path);conn.execute('CREATE TABLE records (identity TEXT PRIMARY KEY, payload TEXT, digest TEXT)')
                # reference_records checks the original compressed file's SHA.
                # An index avoids reparsing an entire group for each small batch.
                for ref in reference_records(B,group):
                    payload=json.dumps(ref,separators=(',',':'));conn.execute('INSERT INTO records VALUES (?,?,?)',(json.dumps(key(ref['key'])),payload,hashlib.sha256(payload.encode()).hexdigest()))
                conn.commit();conn.close();REFERENCE_INDEX[group]=path
            conn=sqlite3.connect(REFERENCE_INDEX[group].as_uri()+'?mode=ro',uri=True)
            selected=[]
            for k in keys:
                if k['group']!=group:continue
                row=conn.execute('SELECT payload,digest FROM records WHERE identity=?',(json.dumps(key(k)),)).fetchone()
                if row is None or hashlib.sha256(row[0].encode()).hexdigest()!=row[1]:raise RuntimeError('Frozen reference index changed')
                selected.append(json.loads(row[0]))
            conn.close();references=selected
        else:references=reference_records(B,group)
        for ref in references:
            ident=key(ref['key'])
            if ident not in wanted:continue
            actual=td(root,ref['key'],ss);audit=read(actual/'AUDIT.json')
            if audit['status']!='VALIDATED':raise RuntimeError('Missing original numerical/causal audit')
            result=compare_records(ref['results'],load_actual(actual))
            if result['status']!='PASS_SINGLE_LIFECYCLE':
                write((receipt_root or root)/'SHARD_COMPARISON_FAILURE.json',dict(key=ref['key'],comparison=result));raise RuntimeError('Scientific mismatch')
            seen.add(ident);records.append(dict(key=ref['key'],comparison=result))
    if seen!=set(wanted):raise RuntimeError('Reference coverage mismatch')
    receipt=dict(status='PASS_EXACT_SHARD_COVERAGE_AND_ORIGINAL_TOLERANCE',completed=len(seen),records=records,full_matrix=False)
    write((receipt_root or root)/'SHARD_COMPARISON.json',receipt);return receipt
def import_completed(donor,out,ss,donor_index=0,already=None):
    """Resume only CSV-published, DONE-hash-verified, audited frozen matches."""
    with (donor/'TRAJECTORIES.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    keys=[dict(group=x['group'],center=int(x['center']),seed=int(x['seed']),stream=x['stream'],arm=x['arm']) for x in rows]
    if len({key(k) for k in keys})!=len(keys):raise RuntimeError('Duplicate donor trajectories')
    if {key(k) for k in keys}&(already or set()):raise RuntimeError('Overlapping resume donors')
    archive=out/'resumed_evidence'/f'donor_{donor_index:03d}';archive.mkdir(parents=True)
    comparison(donor,keys,ss,receipt_root=archive)
    # Keep signed journals and their original path text unaltered. Blob bytes are
    # preserved locally; the relocation map documents their new physical roots.
    files=[]
    for name in ('sources','CAS','_control'):
        if (donor/name).exists():files.extend(copy_tree(donor/name,archive/name))
    for p in donor.iterdir():
        if p.is_file():files.append(materialize(p,archive/'original_root'/p.name))
    for k in keys:
        src=td(donor,k,ss);dst=td(out,k,ss);files.extend(copy_tree(src,dst))
    for group in sorted({k['group'] for k in keys}):
        old=donor/'groups'/group;new=out/'groups'/group
        for p in old.rglob('*'):
            if p.is_file() and p.relative_to(old).parts[0]!='science':
                files.append(materialize(p,archive/'groups'/group/p.relative_to(old)))
                if not (new/p.relative_to(old)).exists():files.append(materialize(p,new/p.relative_to(old)))
    write(archive/'RESUMPTION_RECEIPT.json',dict(status='PASS_COPIED_AUDITED_FROZEN_MATCHING_COMPLETE_TRAJECTORIES',completed=len(keys),
          original_path_text_is_historical_evidence=True,relocation={str(donor/'CAS'):str(archive/'CAS')},files=files,
          incomplete_donor_trajectories_not_counted=True))
    return keys
def merge_batch(batch,out,keys,ss):
    receipt=read(batch/'EXECUTION_RECEIPT.json')
    if receipt['completed']!=len(keys) or receipt['full_matrix']:raise RuntimeError('Invalid partial batch receipt')
    comparison(batch,keys,ss)
    for k in keys:copy_tree(td(batch,k,ss),td(out,k,ss))
    # Keep first prepared inputs for aggregation; source-dependent runtime state
    # remains in each private batch and is independently auditable there.
    for group in {k['group'] for k in keys}:
        src=batch/'groups'/group;dest=out/'groups'/group
        for p in src.rglob('*'):
            if p.is_file() and p.relative_to(src).parts[0]!='science' and not (dest/p.relative_to(src)).exists():materialize(p,dest/p.relative_to(src))

def run(a):
    global REFERENCE_CACHE_ROOT
    run_started=time.monotonic()
    out=a.output.resolve();ss=specs()
    if out.exists():raise RuntimeError('Fresh output required')
    out.mkdir(parents=True)
    REFERENCE_CACHE_ROOT=out/'reference_index'
    asset_pins=None;asset_receipt=None
    if a.read_cache or not a.task_list:
        from verified_asset_pins import AssetPins
        asset_receipt=out/'asset_verification.json';asset_pins=AssetPins(asset_receipt)
    all_tasks=tasks(ss);requested=read(a.task_list) if a.task_list else all_tasks
    wanted={key(k) for k in requested}
    if len(wanted)!=len(requested) or not wanted.issubset({key(k) for k in all_tasks}):raise RuntimeError('Invalid task coverage')
    carried=[]
    for i,donor in enumerate(a.resume_from or []):
        carried.extend(import_completed(donor.resolve(),out,ss,i,{key(k) for k in carried}))
    if a.resume_from:
        write(out/'RESUMPTION_RECEIPT.json',dict(status='PASS_DISJOINT_RECHECKED_RESUME_DONORS',completed=len(carried),
              donors=[str(p.resolve()) for p in a.resume_from],receipts=[str(p.relative_to(out)) for p in sorted((out/'resumed_evidence').glob('*/RESUMPTION_RECEIPT.json'))]))
    if not {key(k) for k in carried}.issubset(wanted):raise RuntimeError('Resume scope differs')
    done={key(k) for k in carried};pending=[k for k in requested if key(k) not in done]
    batches=[]
    for group,seq in itertools.groupby(pending,key=lambda k:k['group']):
        seq=list(seq)
        for start in range(0,len(seq),a.batch_size):batches.append(seq[start:start+a.batch_size])
    active={};started=run_started;last_launch=-1e9;peak=0;low_free=None;finished=[];peak_workers=0
    sources=a.reuse_sources.resolve() if a.reuse_sources else None
    initial_cache=(a.reuse_initials.resolve() if a.reuse_initials else None)
    if not a.task_list:
        # One complete preparation is shared by all disjoint batches. A fresh
        # reviewer run trains sources here once; this migration reuses only
        # independently checked completed training and then prepares M0 once.
        prepared=out/'prepared'
        cmd=[sys.executable,'-X','utf8','-B',str(B/'reproduce_matrix.py'),'--mode','source-to-final','--source-python',str(a.source_python),'--output',str(prepared),'--prepare-only']
        if sources:cmd+=['--reuse-sources',str(sources)]
        if initial_cache:cmd+=['--reuse-initials',str(initial_cache)]
        if asset_receipt:cmd+=['--verified-assets',str(asset_receipt)]
        while psutil.virtual_memory().available/2**30<2.0:
            write(out/'PROGRESS.json',dict(status='PREPARATION_WAITING_FOR_MEMORY',completed=len(done),expected=len(requested)));time.sleep(15)
        with (out/'preparation.log').open('x',encoding='utf-8') as log:
            subprocess.run(cmd,cwd=B,stdout=log,stderr=subprocess.STDOUT,check=True)
        if read(prepared/'EXECUTION_RECEIPT.json')['status']!='PREPARED_ONLY':raise RuntimeError('Shared source/initial preparation failed')
        sources=prepared/'sources';initial_cache=prepared
    if a.auto_tune:
        if a.task_list:raise RuntimeError('Auto-tuning belongs to a complete run, not a scientific shard')
        sample=[k for k in all_tasks if k['group']=='E1'][:16]
        taskfile=out/'throughput_tasks.json';write(taskfile,sample);measurements=[]
        for name,n,cache in [('baseline_one',1,False),('cached_one',1,True),('cached_two',2,True)]:
            probe=out/'throughput'/name
            cmd=[sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--output',str(probe),'--source-python',str(a.source_python),'--reuse-sources',str(sources),'--reuse-initials',str(initial_cache),'--task-list',str(taskfile),'--workers',str(n),'--batch-size','8']
            if cache:cmd+=['--read-cache']
            probe.parent.mkdir(exist_ok=True)
            with (probe.parent/(name+'.log')).open('x',encoding='utf-8') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            m=read(probe/'EXECUTION_RECEIPT.json');measurements.append(dict(case=name,workers=n,cache=cache,seconds=m['elapsed_seconds'],peak_child_rss_bytes=m['peak_child_rss_bytes']))
        one=min((x for x in measurements if x['workers']==1),key=lambda x:x['seconds']);best=min(measurements,key=lambda x:x['seconds'])
        if best['workers']==2 and best['seconds']>one['seconds']*.9:best=one
        a.workers=best['workers'];a.read_cache=best['cache']
        write(out/'THROUGHPUT_SELECTION.json',dict(status='PASS_MEASURED_SELECTION',measurements=measurements,selected=best,scope='E1 sample; full coverage still required'))
    while batches or active:
        for proc,(batch,keys,log,t0) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close();del active[proc]
            if rc:raise RuntimeError(f'Batch failed, no blind retry: {batch}, exit {rc}')
            merge_batch(batch,out,keys,ss);done.update(key(k) for k in keys)
            finished.append(dict(path=str(batch),trajectories=len(keys),seconds=time.monotonic()-t0))
        free=psutil.virtual_memory().available/2**30;low_free=free if low_free is None else min(low_free,free)
        rss=0
        for proc in active:
            try:
                p=psutil.Process(proc.pid);rss+=sum(x.memory_info().rss for x in [p,*p.children(recursive=True)])
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        peak=max(peak,rss)
        allowed=a.workers
        if a.external_receipts and not all(Path(p).exists() for p in a.external_receipts):allowed=min(allowed,1)
        cost=bool(batches and ss[batches[0][0]['group']].get('cost_only'))
        if cost and not active and a.coordinated_cost_isolation:
            write(out/'COST_EXCLUSIVE_REQUEST.json',dict(status='REQUESTED'))
            if not (out/'COST_EXCLUSIVE_GRANTED.json').exists():allowed=0
        if cost:allowed=min(allowed,1)
        gpu_free=0
        if batches and len(active)<allowed and free>=2.0:
            q=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
            gpu_free=int(q.stdout.strip().splitlines()[0])
        if batches and len(active)<allowed and free>=2.0 and gpu_free>=1536 and time.monotonic()-last_launch>=a.launch_gap and (not cost or not active):
            keys=batches.pop(0);i=len(finished)+len(active);batch=out/'batches'/f'batch_{i:04d}'
            taskfile=out/'task_lists'/f'batch_{i:04d}.json';write(taskfile,keys)
            cmd=[sys.executable,'-X','utf8','-B',str(B/'reproduce_matrix.py'),'--mode','source-to-final','--source-python',str(a.source_python),'--group',keys[0]['group'],'--task-list',str(taskfile),'--output',str(batch)]
            if sources:cmd+=['--reuse-sources',str(sources)]
            if initial_cache:cmd+=['--reuse-initials',str(initial_cache)]
            if asset_receipt:cmd+=['--verified-assets',str(asset_receipt)]
            if a.read_cache and not ss[keys[0]['group']].get('cost_only'):cmd+=['--read-cache']
            logpath=out/'logs'/f'batch_{i:04d}.log';logpath.parent.mkdir(exist_ok=True);log=logpath.open('x',encoding='utf-8')
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=B)
            active[proc]=(batch,keys,log,time.monotonic());last_launch=time.monotonic()
        peak_workers=max(peak_workers,len(active))
        progress=len(done)
        for proc,(batch,keys,log,t0) in active.items():
            try:progress+=read(batch/'PROGRESS.json')['completed']
            except (OSError,ValueError):pass
        write(out/'PROGRESS.json',dict(completed=progress,verified_completed=len(done),expected=len(requested),carried=len(carried),
              workers_requested=a.workers,workers_active=len(active),available_gib=free,peak_child_rss_bytes=peak,
              elapsed_seconds=time.monotonic()-started,live_children=[p.pid for p in active],full_matrix=False))
        if batches or active:time.sleep(2)
    if done!=wanted:raise RuntimeError('Missing/duplicate aggregate coverage')
    full=not a.task_list and len(done)==17118
    write(out/'EXECUTION_RECEIPT.json',dict(status='EXECUTION_COMPLETE_AWAITING_FROZEN_RESULT_COMPARISON' if full else 'PARTIAL_ENTRY_CHECK',
          completed=len(done),expected=len(requested),full_matrix=full,all_17_groups=full,source_models_retrained=False,
          source_to_final_training_lineage=True,original_completed_source_receipts_preserved=True,carried_complete_trajectories=len(carried),
          peak_child_rss_bytes=peak,peak_active_workers=peak_workers,minimum_available_gib=low_free,elapsed_seconds=time.monotonic()-started,batches=finished))
    if asset_pins:asset_pins.close()
    return out

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--source-python',required=True,type=Path)
    p.add_argument('--reuse-sources',type=Path);p.add_argument('--reuse-initials',type=Path);p.add_argument('--resume-from',type=Path,action='append')
    p.add_argument('--task-list',type=Path);p.add_argument('--read-cache',action='store_true');p.add_argument('--workers',type=int,default=1)
    p.add_argument('--auto-tune',action='store_true',help='Measure baseline/cached serial/two processes before the full matrix')
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--launch-gap',type=float,default=45)
    p.add_argument('--external-receipts',nargs='*');p.add_argument('--coordinated-cost-isolation',action='store_true');a=p.parse_args()
    if a.workers<1 or a.batch_size<1:raise ValueError('workers and batch size must be positive')
    try:run(a)
    except BaseException:
        if a.output.exists():write(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),live_children_must_not_be_duplicated=[x.pid for x in psutil.Process().children(recursive=True)],automatic_retry=False))
        raise
if __name__=='__main__':main()
