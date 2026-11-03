"""Independent frozen DevCal preparation and full source-to-lifecycle execution.

The full matrix is 8 centers x 1 seed x 3 streams x 66 configurations.
--limit is an explicitly partial entry check and cannot produce full acceptance.
"""
import argparse,csv,gzip,importlib.metadata,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
# Match the original d3_engine's eager NumPy eigen initialization before torch.
# Import-only checks missed MKL's lazy OpenMP initialization. Three isolated
# native diagnostics proved this order works; both torch-first orders abort.
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
np.polynomial.hermite.hermgauss(21)
import torch
from regenerate_devcal_inputs import B,D,F,read,sha,verify
from compare_outputs import compare_records,load_actual

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.new');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8');temp.replace(path)

def prepare(out):
    if out.exists():raise RuntimeError('Fresh output required')
    if out.is_relative_to(D) or out.is_relative_to(B/'training'):raise RuntimeError('Output overlaps immutable inputs')
    receipt=verify();out.mkdir(parents=True)
    with (out/'input_generation.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8','-B',str(B/'regenerate_devcal_inputs.py'),'--output',str(out/'generated')],stdout=log,stderr=subprocess.STDOUT,check=True)
    root=out/'runtime';root.mkdir()
    for row in receipt['files']:
        name=row['file']
        if name.startswith(('data/','initial/')) or name=='DEVCAL_SELECTED_LOCK.json':continue
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(F/name,dest)
        if sha(dest)!=row['sha256']:raise RuntimeError('Staged source changed')
    shutil.copytree(out/'generated/data',root/'data')
    for directory in ('initial','process','progress','results/DevCal'):(root/directory).mkdir(parents=True,exist_ok=True)
    # The historical lock remains unchanged; its code and source files are still
    # verifiable. Newly fitted weights are bound separately in runtime receipts.
    write(out/'PREPARATION_RECEIPT.json',dict(status='PASS_STAGED_FROZEN_CODE_AND_REGENERATED_INPUTS',input_comparison=read(out/'generated/INPUT_COMPARISON.json'),training_executed=False))
    return root

def execute(a,out,root):
    expected=read(B/'training/ENVIRONMENT_LOCK.json')['packages']
    actual={k:importlib.metadata.version(k) for k in expected}
    if actual!=expected or sys.version_info[:3]!=(3,13,5):raise RuntimeError('Frozen lifecycle environment mismatch')
    refs=D/'references';manifest=read(refs/'MANIFEST.json')
    for record in manifest['files']:
        if sha(refs/record['file'])!=record['sha256']:raise RuntimeError('Frozen reference changed')
    if a.mode=='source-to-final':
        if not a.source_python:raise RuntimeError('--source-python required for source fitting')
        source_out=out/'source_training'
        if a.reuse_sources:
            from execution_reuse import reuse_source
            spec=read(B/'training/groups/E1.json')
            source,_=reuse_source(a.reuse_sources/'synthetic__GRU_1_7311',source_out,spec,7311)
        else:
            with (out/'source_training.log').open('w',encoding='utf-8') as log:
                subprocess.run([str(a.source_python),'-X','utf8','-B',str(B/'training/source_training.py'),'--domain','synthetic','--architecture','GRU','--horizon','1','--seed','7311','--output',str(source_out)],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not read(source_out/'SOURCE_TRAINING_RECEIPT.json')['full_source_training_complete']:raise RuntimeError('Source fitting incomplete')
        source=source_out/'checkpoints/source_GRU_H1_7311.pt'
    else:source=root/'checkpoints/source_GRU_H1_7311.pt'
    sys.path[:0]=[str(root),str(root/'runtime')];sys.dont_write_bytecode=True
    import d3_common as common
    common.verify_science_lock()
    # The original runtime was linked to its project data directory. Replace
    # that location binding, keeping the frozen protocol functions unchanged.
    import core.protocol as protocol
    protocol.ROOT=root
    import d3_engine as engine
    import d3_trajectory as trajectory
    import d3_audit as audit
    from f2_runner import Meter
    from f2_plan import arms
    for module in (common,engine,trajectory,audit):
        if module.ROOT.resolve()!=root:raise RuntimeError('Mixed execution roots')
    torch=engine.torch
    model=engine.Forecast().to(engine.DEVICE);model.load_state_dict(torch.load(source,map_location=engine.DEVICE,weights_only=True))
    original_source=read(F/'ASSET_IDENTITY.json')['sources']['7311']
    if engine.state_hash(model)!=original_source['state_hash']:raise RuntimeError('New source state differs from original')
    engine.SOURCE['7311']=dict(path=os.path.relpath(source,root),sha256=sha(source),state_hash=engine.state_hash(model))
    write(out/'SOURCE_COMPARISON.json',dict(status='PASS_EXACT_SOURCE_STATE',source_training_executed=a.mode=='source-to-final' and not a.reuse_sources,completed_source_training_reused=bool(a.reuse_sources),state_hash=engine.state_hash(model),source_sha256=sha(source),original_sha256=original_source['sha256']))
    if a.reuse_initials:
        previous=read(a.reuse_initials/'INITIAL_COMPARISON.json')
        if previous['status']!='PASS_ALL_EIGHT_INITIAL_STATES':raise RuntimeError('Initial cache is not validated')
        for center in read(F/'RUN_MANIFEST.json')['centers']['DevCal']:
            old=a.reuse_initials/'runtime/initial'/f'{center}_7311.pt';meta=read(old.with_suffix('.json'))
            if sha(old)!=meta['sha256'] or meta['source_sha256']!=sha(source) or meta['warmup_seed']!=7311+center or meta['warmup_steps']!=50:raise RuntimeError('Warmup cache identity differs')
            if meta['warmup_hash']!=engine.Timeline(center).warmup().hash or meta['state_hash']!=read(F/f'initial/{center}_7311.json')['state_hash']:raise RuntimeError('Warmup data/state differs')
            from execution_reuse import copy_verified
            copy_verified(old,root/'initial'/old.name);copy_verified(old.with_suffix('.json'),root/'initial'/old.with_suffix('.json').name)
    # Only newly produced state and packaged inputs are permitted project reads.
    for name in ('TMP','TEMP','TORCH_HOME','CUDA_CACHE_PATH'):
        cache=out/'cache'/name;cache.mkdir(parents=True,exist_ok=True);os.environ[name]=str(cache)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
            mutation=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if mutation and not path.is_relative_to(out):raise PermissionError('Write outside new output: '+str(path))
            system_root=Path(os.environ.get('WINDIR',sys.base_prefix)).resolve()
            if not any(path.is_relative_to(p) for p in (B.resolve(),out,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),system_root)):raise PermissionError('External dependency: '+str(path))
    sys.addaudithook(guard)
    meter=Meter('DevCal');centers=read(F/'RUN_MANIFEST.json')['centers']['DevCal'];initials=[]
    for center in centers:
        with meter.scope('warmup',dict(center=center,seed=7311)):initial=engine.initial(center,7311)
        old=read(F/f'initial/{center}_7311.json')
        if engine.state_hash(initial)!=old['state_hash']:raise RuntimeError('Regenerated initialization differs')
        initials.append(dict(center=center,state_hash=engine.state_hash(initial)))
    write(out/'INITIAL_COMPARISON.json',dict(status='PASS_ALL_EIGHT_INITIAL_STATES',actual_warmup_steps=meter.done['warmup'],states=initials))
    if a.prepare_only:
        write(out/'EXECUTION_RECEIPT.json',dict(status='PREPARED_SOURCE_AND_EIGHT_INITIALS',full_matrix=False,completed=0,actual_steps=meter.done));return
    cache_counters=None
    if a.read_cache:
        from execution_reuse import install_devcal_read_cache
        cache_counters=install_devcal_read_cache(engine,trajectory,audit)
    arm_lookup={arm['id']:arm for arm in arms('DevCal')};done=[];stage_count=0;started=time.monotonic()
    with (refs/'trajectories.jsonl').open('r',encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            if index%a.task_shards!=a.task_shard_index:continue
            if a.limit is not None and len(done)>=a.limit:break
            ref=json.loads(line);key=ref['key'];td=root/'science/DevCal'/str(key['center'])/str(key['seed'])/key['stream']/key['arm']
            if not a.aggregate_from:
                with meter.scope('candidate',key):
                    trajectory.run_trajectory('DevCal',key['center'],key['seed'],key['stream'],arm_lookup[key['arm']],'INDEPENDENT_DEVCAL_REPRODUCTION')
            rows,audited=audit.audit_phase('DevCal',24,True,only_td=td)
            if audited['status']!='VALIDATED':raise RuntimeError('Frozen lineage/causality audit failed')
            comparison=compare_records(ref['results'],load_actual(td));write(td/'SCIENTIFIC_COMPARISON.json',comparison)
            if comparison['status']!='PASS_SINGLE_LIFECYCLE':raise RuntimeError('DevCal scientific output mismatch')
            stage_count+=len(rows);done.append(dict(key=key,comparison= comparison['status'],audit=audited));write(out/'PROGRESS.json',dict(completed=len(done),expected=1584,actual_steps=meter.done,elapsed_seconds=time.monotonic()-started))
    full=a.limit is None and a.task_shards==1 and len(done)==1584
    if full:
        if a.aggregate_from:
            receipts=[read(p/'EXECUTION_RECEIPT.json') for p in a.aggregate_from]
            if sum(x['actual_steps']['candidate'] for x in receipts)!=1900800 or meter.done!={'warmup':0,'candidate':0}:raise RuntimeError('Shard SGD accounting mismatch')
            prepared=read(a.reuse_initials/'EXECUTION_RECEIPT.json')
            if prepared['status']!='PREPARED_SOURCE_AND_EIGHT_INITIALS' or prepared['actual_steps']['warmup']!=400:raise RuntimeError('Original shared warmup accounting mismatch')
        elif meter.done!={'warmup':0 if a.reuse_initials else 400,'candidate':1900800}:raise RuntimeError('SGD accounting mismatch')
        write(root/'results/DevCal/INDEPENDENT_AUDIT.json',dict(status='VALIDATED',trajectories=len(done),stage_rows=stage_count,records=done,original_numerical_and_causal_audit=True))
        import f2_analysis
        # These are the original aggregation, selection and GO/STOP functions.
        f2_analysis.main('DevCal')
        actual_selection=read(root/'DEVCAL_SELECTED_LOCK.json');frozen_selection=read(F/'DEVCAL_SELECTED_LOCK.json')
        for key in ('candidate','best_active_strong','baseline_selection','all_qualified_joint_ids'):
            if actual_selection[key]!=frozen_selection[key]:raise RuntimeError('Frozen development selection changed: '+key)
        tables=[]
        for name in ('RESULTS_BY_ARM.csv','RESULTS_BY_STREAM.csv'):
            def load(path):
                with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
            actual_rows=load(root/'results/DevCal'/name);expected_rows=load(refs/name)
            if len(actual_rows)!=len(expected_rows):raise RuntimeError('Development summary coverage differs')
            for left,right in zip(expected_rows,actual_rows):
                if left.keys()!=right.keys():raise RuntimeError('Development summary schema differs')
                for key,value in left.items():
                    if key.endswith('_seconds') or key=='model_cache_bytes':continue
                    try:equal=math.isclose(float(value),float(right[key]),abs_tol=1e-7,rel_tol=0)
                    except ValueError:equal=value==right[key]
                    if not equal:raise RuntimeError('Development summary differs: '+name+'/'+key)
            tables.append(dict(file=name,status='PASS',rows=len(actual_rows),absolute_tolerance=1e-7))
            joint=[r for r in actual_rows if r['arm'].startswith('JOINT__')]
            target=out/'paper_tables'/('DEVCAL_JOINT_'+name);target.parent.mkdir(exist_ok=True)
            with target.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(joint[0]));writer.writeheader();writer.writerows(joint)
        write(out/'STATISTICS_COMPARISON.json',dict(status='PASS',tables=tables,development_selection_exact=True))
    write(out/'EXECUTION_RECEIPT.json',dict(status='PASS_FULL_DEVCAL_SOURCE_AND_LIFECYCLES' if full and a.mode=='source-to-final' else 'PASS_PARTIAL_DEVCAL_ENTRY',completed=len(done),expected=1584,full_matrix=full,source_models_retrained=a.mode=='source-to-final' and not a.reuse_sources,actual_steps=meter.done,training_in_verified_shards=bool(a.aggregate_from),shard_execution_receipts=[str(p/'EXECUTION_RECEIPT.json') for p in a.aggregate_from],cache_counters=cache_counters,task_shards=a.task_shards,task_shard_index=a.task_shard_index,full_paper_package_accepted=False,records=done))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['prepare','frozen-source','source-to-final'],required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--source-python',type=Path);p.add_argument('--limit',type=int);p.add_argument('--reuse-sources',type=Path)
    p.add_argument('--reuse-initials',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--read-cache',action='store_true');p.add_argument('--task-shards',type=int,default=1);p.add_argument('--task-shard-index',type=int,default=0);p.add_argument('--aggregate-from',type=Path,nargs='*',default=[]);a=p.parse_args()
    if a.task_shards not in (1,2) or not 0<=a.task_shard_index<a.task_shards:raise ValueError('Invalid process shard')
    if a.limit is not None and a.limit<1:raise ValueError('Positive limit required')
    out=a.output.resolve()
    if out.exists():raise RuntimeError('Fresh output required; existing evidence must not be changed')
    try:
        root=prepare(out)
        if a.aggregate_from:
            if a.task_shards!=1 or a.limit or not a.reuse_initials:raise RuntimeError('Aggregation must cover the full development set')
            from accelerated_matrix import copy_tree
            coverage=set()
            for shard in a.aggregate_from:
                receipt=read(shard/'EXECUTION_RECEIPT.json')
                if receipt['status']!='PASS_PARTIAL_DEVCAL_ENTRY' or receipt['completed']!=1584//receipt['task_shards']:raise RuntimeError('Incomplete development shard')
                for record in receipt['records']:
                    k=record['key'];identity=(k['center'],k['seed'],k['stream'],k['arm'])
                    if identity in coverage:raise RuntimeError('Overlapping development shards')
                    coverage.add(identity)
                    rel=Path('science/DevCal')/str(k['center'])/str(k['seed'])/k['stream']/k['arm']
                    copy_tree(shard/'runtime'/rel,root/rel)
            if len(coverage)!=1584:raise RuntimeError('Incomplete development coverage')
        if a.mode!='prepare':execute(a,out,root)
    except BaseException as error:
        if out.exists():write(out/'FAILURE.json',dict(error=repr(error),automatic_retry=False))
        raise
if __name__=='__main__':main()
