"""Run the complete frozen matrix in a new directory with a shared input/CAS store.

No original workstation directory is used. Each trajectory is audited before
aggregation. Source refitting uses the separate source Python interpreter.
The default inventory command runs no training; --mode source-to-final is explicit.
"""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,os,shutil,subprocess,sys,time

B=Path(__file__).resolve().parent
T=B/'training'
sys.path.insert(0,str(T))
import run as r

def source_key(s,seed):
    return '_'.join(map(str,[s['domain'],s.get('building',''),s['architecture'],s['horizon'],seed]))

def checkpoint_state(path):
    import torch
    value=torch.load(path,map_location='cpu',weights_only=True);digest=hashlib.sha256()
    for key,tensor in value.items():digest.update(key.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--mode',choices=['inventory','frozen-source','source-to-final'],default='inventory')
    p.add_argument('--output',type=Path)
    p.add_argument('--source-python',type=Path)
    p.add_argument('--group',action='append',help='Omit for all 17 frozen groups')
    p.add_argument('--limit',type=int,help='Entry validation only; any limit prevents a full-matrix PASS')
    p.add_argument('--task-list',type=Path,help='Explicit disjoint trajectory keys; always a partial shard receipt')
    p.add_argument('--reuse-sources',type=Path,help='Completed source directory; exact recipe/environment/state checked and copied')
    p.add_argument('--reuse-initials',type=Path,help='Verified warmup cache; identity mismatches do not reuse')
    p.add_argument('--read-cache',action='store_true',help='Bounded pinned immutable model decoding, never scores or audits')
    p.add_argument('--prepare-only',action='store_true',help='Prepare sources and initial states, without lifecycle execution')
    p.add_argument('--verified-assets',type=Path,help='Live parent full verification with duplicated deny-write file handles')
    p.add_argument('--coordinated-cost-isolation',action='store_true',help='Ask the local parallel scheduler to drain other project training before timing experiments')
    a=p.parse_args()
    specs=[r.read(f) for f in sorted((T/'groups').glob('*.json')) if not a.group or f.stem in a.group]
    specs.sort(key=lambda s:(bool(s.get('cost_only')),s['group']))
    if not specs or (a.group and set(a.group)!={s['group'] for s in specs}):raise RuntimeError('Unknown group selection')
    selected=None
    keytuple=lambda k:(k['group'],k['center'],k['seed'],k['stream'],k['arm'])
    if a.task_list:
        entries=r.read(a.task_list);selected={keytuple(k) for k in entries}
        allowed={(s['group'],c,se,st,arm['id']) for s in specs for c,se,st,arm in itertools.product(s['centers'],s['seeds'],s['stream_ids'],s['arms'])}
        if len(selected)!=len(entries) or not selected or not selected.issubset(allowed):raise RuntimeError('Invalid/duplicate shard keys')
    count=sum(len(s['centers'])*len(s['seeds'])*len(s['stream_ids'])*len(s['arms']) for s in specs)
    if a.mode=='inventory':
        print(json.dumps({'groups':[s['group'] for s in specs],'trajectories':count,'training_executed':False},indent=2));return
    if a.output is None:raise RuntimeError('--output is required')
    out=a.output.resolve()
    if out==B or out.is_relative_to(T):raise RuntimeError('Output overlaps immutable training inputs')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Fresh output required; failed attempts are never silently overwritten')
    if a.mode=='source-to-final' and a.source_python is None:raise RuntimeError('--source-python must name the pinned Python 3.11 source environment')
    out.mkdir(parents=True,exist_ok=True)
    pinned_assets=None
    if a.verified_assets:
        from verified_asset_pins import InheritedAssetPins
        pinned_assets=InheritedAssetPins(a.verified_assets)
        r.atomic(out/'ASSET_VERIFICATION.json',dict(status='PASS_INHERITED_FULL_SHA_VERIFICATION_AND_OS_READ_LOCKS',files=len(pinned_assets.manifest['files']),parent_receipt=str(a.verified_assets)))
    else:r.verify()
    env=r.read(T/'ENVIRONMENT_LOCK.json')
    import importlib.metadata
    actual={k:importlib.metadata.version(k) for k in env['packages']}
    if actual!=env['packages'] or sys.version_info[:3]!=(3,13,5):raise RuntimeError({'runtime_mismatch':actual,'expected':env['packages']})
    r.atomic(out/'RUNTIME.json',{'python':sys.version,'packages':actual,'mode':a.mode})
    sources={};started=time.monotonic()
    if a.mode=='source-to-final':
        for spec in specs:
            for seed in spec['seeds']:
                key=source_key(spec,seed)
                if key in sources:continue
                target=out/'sources'/key;target.parent.mkdir(exist_ok=True)
                if selected is not None and not any(k[0]==spec['group'] and k[2]==seed for k in selected):continue
                if a.reuse_sources and (a.reuse_sources/key).is_dir():
                    from execution_reuse import reuse_source
                    sources[key]=reuse_source(a.reuse_sources/key,target,spec,seed)
                    continue
                cmd=[str(a.source_python),'-X','utf8','-B',str(T/'source_training.py'),'--domain',spec['domain'],
                     '--architecture',spec['architecture'],'--horizon',str(spec['horizon']),'--seed',str(seed),'--output',str(target)]
                if spec.get('building'):cmd.extend(['--task',spec['building']])
                with (target.parent/(key+'.log')).open('w',encoding='utf-8') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                receipt=r.read(target/'SOURCE_TRAINING_RECEIPT.json')
                if not receipt['full_source_training_complete']:raise RuntimeError('Incomplete source fitting')
                models=list((target/'checkpoints').glob('*.pt'))
                if len(models)!=1:raise RuntimeError('Ambiguous trained source checkpoint')
                actual_state=checkpoint_state(models[0]);expected_state=spec['sources'][str(seed)]['state_hash']
                r.atomic(target/'SOURCE_STATE_COMPARISON.json',dict(status='PASS' if actual_state==expected_state else 'FAIL',actual=actual_state,expected=expected_state))
                if actual_state!=expected_state:raise RuntimeError('New source weights differ: '+key)
                sources[key]=(models[0],target/'SOURCE_TRAINING_RECEIPT.json')
    b=r.bootstrap(out);b.torch.set_num_threads(1);done=0;failures=[];cost_installed=False
    from external_model_storage import install as install_external_model_storage
    install_external_model_storage(out)
    from legacy_reporting_precision import install as install_reporting_precision
    install_reporting_precision(b)
    from main_attack_metadata import install as install_main_attack_metadata
    install_main_attack_metadata(b)
    cache_counters=None
    if a.read_cache:
        if any(s.get('cost_only') for s in specs):raise RuntimeError('Extra read cache must not change cost measurement implementation')
        from execution_reuse import install_read_cache
        cache_counters=install_read_cache(b)
    for original in specs:
        if original.get('cost_only') and not cost_installed and a.coordinated_cost_isolation:
            request=out/'COST_EXCLUSIVE_REQUEST.json';grant=out/'COST_EXCLUSIVE_GRANTED.json'
            r.atomic(request,dict(status='REQUESTED',reason='Keep cost repeats free of other project training'))
            while not grant.exists():time.sleep(15)
            if r.read(grant).get('status')!='GRANTED_PROJECT_TRAINING_EXCLUSIVITY':raise RuntimeError('Invalid cost-isolation grant')
        spec=r.attach(r.stage(original,out),b)
        if spec.get('cost_only') and not cost_installed:
            sys.modules['common'].Path=Path
            import cost_instrument
            cost_instrument.install(b);cost_installed=True
        with b.context(spec):
            if a.mode=='source-to-final':
                for seed in spec['seeds']:
                    if selected is not None and not any(k[0]==spec['group'] and k[2]==seed for k in selected):continue
                    source,receipt=sources[source_key(spec,seed)]
                    model=b.Forecast().to(b.DEVICE);model.load_state_dict(b.torch.load(source,map_location=b.DEVICE,weights_only=True))
                    if b.state_hash(model)!=spec['sources'][str(seed)]['state_hash']:raise RuntimeError(f'Source state mismatch: {source_key(spec,seed)}')
                    spec['sources'][str(seed)]={'path':str(source),'sha256':r.sha(source),'state_hash':b.state_hash(model)}
                    if spec['domain']=='alibaba':
                        initial_out=out/'historical_initials'/f'{spec["group"]}_{seed}'
                        initial_out.parent.mkdir(exist_ok=True)
                        cmd=[str(a.source_python),'-X','utf8','-B',str(B/'historical_alibaba_initial.py'),'--source',str(source),'--seed',str(seed),'--output',str(initial_out)]
                        from alibaba_initial_reuse import reuse as reuse_alibaba_initial
                        reused=bool(a.reuse_initials and reuse_alibaba_initial(a.reuse_initials,original,seed,source,initial_out))
                        if not reused:
                            with initial_out.with_suffix('.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                        for center in spec['centers']:
                            for suffix in ['.pt','.json']:shutil.copy2(initial_out/f'{center}_{seed}{suffix}',Path(spec['root'])/'initial'/f'{center}_{seed}{suffix}')
                        continue
                    for center in spec['centers']:
                        if selected is not None and not any(k[0]==spec['group'] and k[1]==center and k[2]==seed for k in selected):continue
                        path=Path(spec['root'])/'initial'/f'{center}_{seed}.pt'
                        if a.reuse_initials:
                            from execution_reuse import reuse_initial
                            if reuse_initial(a.reuse_initials,spec,center,seed,path,source,receipt,b):continue
                        # Original adapt copies its input model. Explicit reload prevents
                        # any future implementation from carrying states across centers.
                        model.load_state_dict(b.torch.load(source,map_location=b.DEVICE,weights_only=True))
                        warmup=b.Timeline(center).warmup();warm_seed=b.training_seed(seed,center,0)
                        initial=b.adapt(model,b.tensors(b.windows(warmup)),steps=50,seed=warm_seed)
                        path=Path(spec['root'])/'initial'/f'{center}_{seed}.pt'
                        expected_meta=r.read(path.with_suffix('.json'))
                        if b.state_hash(initial)!=expected_meta['state_hash']:raise RuntimeError(f'Initial state mismatch: {spec["group"]}/{center}/{seed}')
                        b.torch.save(initial.state_dict(),path)
                        meta=r.read(path.with_suffix('.json'));meta.update(sha256=r.sha(path),state_hash=b.state_hash(initial),source_sha256=r.sha(source),warmup_hash=warmup.hash,warmup_steps=50,warmup_seed=warm_seed,path_b_source_receipt=str(receipt))
                        r.atomic(path.with_suffix('.json'),meta)
            for center,seed,stream,arm in itertools.product(spec['centers'],spec['seeds'],spec['stream_ids'],spec['arms']):
                if a.prepare_only:break
                if selected is not None and (spec['group'],center,seed,stream,arm['id']) not in selected:continue
                if a.limit is not None and done>=a.limit:break
                key=dict(group=spec['group'],center=center,seed=seed,stream=stream,arm=arm['id'])
                td=Path(spec['root'])/'science'/spec['phase']/str(center)/str(seed)/stream/arm['id']
                try:
                    fn=b.run_legacy if arm['id'] in b.LEGACY_NAMES else b.run_trajectory
                    def progress(**kw):
                        if spec.get('cost_only') and arm['id'] not in b.LEGACY_NAMES:cost_instrument.finish(b,td,kw['stage'])
                    fn(spec['phase'],center,seed,stream,arm,'PORTABLE_REPRODUCTION_NOT_NEW_PAPER_EVIDENCE',progress=progress)
                    audit=b.audit_trajectory(spec['phase'],td)
                    if audit.get('status')!='VALIDATED':raise RuntimeError(audit)
                    r.atomic(td/'AUDIT.json',dict(audit,execution_lock_sha256=b.execution_lock_sha()))
                    rows=[r.read(td/f'result_{t:03}.json') for t in range(1,spec['stages']+1)]
                    mean=lambda k:sum(x[k] for x in rows)/len(rows)
                    benefit=sum(x['candidate_beneficial_audit_only'] for x in rows)
                    summary=dict(**key,status='VALIDATED',opportunities=len(rows),mse=mean('deployed_service_mse'),harm_frequency=mean('harmful'),mean_downside=mean('downside'),commit_rate=mean('commit'),beneficial_candidates=benefit,beneficial_commits=sum(x['beneficial_commit'] for x in rows),beneficial_recall=sum(x['beneficial_commit'] for x in rows)/benefit if benefit else None)
                    done+=1
                    with (out/'TRAJECTORIES.csv').open('w' if done==1 else 'a',encoding='utf-8-sig' if done==1 else 'utf-8',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(summary))
                        if done==1:writer.writeheader()
                        writer.writerow(summary)
                    r.atomic(out/'PROGRESS.json',dict(completed=done,expected=len(selected) if selected is not None else count,last=key,elapsed_seconds=time.monotonic()-started,full_matrix=done==count and a.limit is None and selected is None))
                    print(json.dumps(dict(completed=done,expected=count,**key)),flush=True)
                except Exception as error:
                    failures.append(dict(**key,error=repr(error)));r.atomic(out/'FAILURES.json',failures);raise
        if a.limit is not None and done>=a.limit:break
    full=done==count and a.limit is None and selected is None and not a.prepare_only
    r.atomic(out/'EXECUTION_RECEIPT.json',dict(status='EXECUTION_COMPLETE_AWAITING_FROZEN_RESULT_COMPARISON' if full else 'PREPARED_ONLY' if a.prepare_only else 'PARTIAL_ENTRY_CHECK',completed=done,expected=len(selected) if selected is not None else count,all_17_groups=len(specs)==17,full_matrix=full,source_models_retrained=a.mode=='source-to-final' and not a.reuse_sources,source_reuse_requested=bool(a.reuse_sources),source_training_scope=('Frozen source models used; no source training in this invocation' if a.mode=='frozen-source' else 'Original full-training receipts preserved; reused models are not retrained in this invocation' if a.reuse_sources else 'New source training'),cache_counters=cache_counters,elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':main()
