"""Explicit registered groups; no data generation or training on import."""
from common import *
import ast,itertools

def all_arms():
    source=(PRIOR/'f2_plan.py').read_text(encoding='utf8');tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arms')
    env=dict(ROOT=PRIOR,read=read,PHASES=('Smoke','NewFinal'),LEGACY_IDS=('ALWAYS_FT','ER','EWC','HUBER_FT','VALIDATION_COMMIT','FROZEN'),LEGACY_NAMES=dict(zip(('ALWAYS_FT','ER','EWC','HUBER_FT','VALIDATION_COMMIT','FROZEN'),('Always-FT','ER','EWC','Huber-FT','ValidationCommit','Frozen'))))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(PRIOR/'f2_plan.py'),'exec'),env)
    return env['arms']('NewFinal')

def source_map(arch,h):
    auth=read(ROOT/'assets/INTERFACE_AUTH.json');binding=read(ROOT/'assets/SOURCE_CHECKPOINT_HISTORY_BINDING.json')
    assert auth['status']=='PASS_MODEL_IDENTITY_AND_SHAPE_ONLY' and binding['status']=='PASS'
    rows={f'{x["architecture"]}_H{x["horizon"]}_{x["seed"]}':x for x in binding['rows']}
    out={}
    for seed in (7311,7312,7313):
        key=f'{arch}_H{h}_{seed}';record=auth['rows'][key];assert record['sha256']==rows[key]['sha256']
        out[str(seed)]=dict(path=record['path'],sha256=record['sha256'],state_hash=record['state_hash'])
    return out

def base(group,stage,centers,streams,arms,arch='GRU',h=1,domain='synthetic',features=24):
    return dict(group=group,stage=stage,phase=group,root=str(STORAGE/'groups'/group),centers=list(centers),seeds=[7311,7312,7313],
        architecture=arch,horizon=h,features=features,domain=domain,stages=24,stream_specs=streams,stream_ids=list(streams),arms=arms,
        sources=source_map(arch,h),physical_cluster='center' if domain=='synthetic' else 'machine',actual_new_scientific_trajectories=True)

def synthetic_specs():
    arms=all_arms();lookup={x['id']:x for x in arms}
    e1_ids=['JOINT__l3__m0','RECENT_MEAN__g12','MWU_MEAN__g04','DUAL_MEAN__g12','RAVM_ORIGINAL','ALWAYS_FT','HUBER_FT','FROZEN']
    streams={}
    for kind in ('mixed','persistent_bias','replay'):
        for alpha in (.05,.1,.2):streams[f'{kind}_a{int(alpha*100):03}']=dict(kind=kind,alpha=alpha,location='both' if kind=='mixed' else 'verification_only')
    specs=[base('E1','E1',range(64000,64024),streams,[lookup[k] for k in e1_ids])]
    for arch in ('MLP','TCN','GRU'):
        for h in (1,4):specs.append(base(f'E2_{arch}_H{h}','E2',range(64100,64124),dict(clean=dict(kind='clean',alpha=0.,location='both'),mixed=dict(kind='mixed',alpha=.1,location='both')),[lookup[k] for k in e1_ids[:6]],arch,h))
    return specs

def external_spec():
    auth=read(ROOT/'assets/E3_ALIBABA_AUTH.json');assert auth['status']=='PASS'
    group=base('E3','E3',range(20060,20100),dict(clean=dict(kind='clean',alpha=0.,location='both'),mixed=dict(kind='mixed',alpha=.1,location='both')),all_arms(),domain='alibaba')
    source={}
    for seed,record in auth['sources'].items():
        meta=record['metadata'];value=meta.get('source_state_hash',meta.get('state_hash'))
        assert value
        source[seed]=dict(path=record['path'],sha256=record['sha256'],state_hash=value)
    group['sources']=source;group['existing_m0']=auth['existing_m0'];group['source_data']=auth['rows'];group['stages']=auth['stages'];return group

def jobs(spec):
    return [dict(center=c,seed=s,stream=stream,arm=arm) for c,s,stream,arm in itertools.product(spec['centers'],spec['seeds'],spec['stream_ids'],spec['arms'])]

def cost_specs():
    manifest=read(PRIOR/'RUN_MANIFEST.json');center=manifest['centers']['NewFinal'][0];seed=7311
    source=read(PRIOR/'ASSET_IDENTITY.json')['sources'][str(seed)]
    source=dict(source,path=str(PRIOR/source['path']))
    out=[]
    for repetition in range(1,4):
        spec=base(f'E5_COST_R{repetition}','E5',[center],dict(clean=dict(kind='clean',alpha=0.,location='both')),all_arms())
        spec.update(seeds=[seed],sources={str(seed):source},cost_only=True,actual_new_scientific_trajectories=False,repetition=repetition,
            cost_timing_serial_ready_graph=True,existing_m0=True,source_data=[dict(center=center,source_path=str(PRIOR/f'data/center_{center}.npz'),sha256=sha(PRIOR/f'data/center_{center}.npz'))],
            reuse_main_initial=str(PRIOR/f'initial/{center}_{seed}.pt'),cost_scope='deterministic Main input timing replicas; excluded from all scientific sample counts and performance CIs')
        out.append(spec)
    return out

def prepare_layout(spec):
    root=Path(spec['root'])
    for d in ('data','initial','science','process','runtime/core','reference'):(root/d).mkdir(parents=True,exist_ok=True)
    source_files={
        'runtime/core/engine.py':PRIOR/'runtime/core/engine.py',
        'reference/LEGACY_BASELINE_IMPLEMENTATION_IDENTITY.json':PRIOR/'reference/LEGACY_BASELINE_IMPLEMENTATION_IDENTITY.json',
        'DEVCAL_SELECTED_LOCK.json':PRIOR/'DEVCAL_SELECTED_LOCK.json',
        'EXPERIMENT_LOCK.json':ROOT/'SCIENCE_EXECUTION.lock.json'}
    for rel,source in source_files.items():
        target=root/rel
        if target.exists():assert sha(target)==sha(source)
        else:target.write_bytes(source.read_bytes())
    target=root/'RUN_MANIFEST.json';value=dict(seeds={spec['phase']:spec['seeds']},centers={spec['phase']:spec['centers']},domain=spec['domain'],physical_cluster=spec['physical_cluster'])
    if not target.exists():atomic(target,value)
    else:assert read(target)==value

def budget(spec):
    trajectories=len(jobs(spec));adapting=sum(a['id']!='FROZEN' for a in spec['arms'])
    warmup=0 if spec.get('existing_m0') else len(spec['centers'])*len(spec['seeds'])*50
    candidate=len(spec['centers'])*len(spec['seeds'])*len(spec['stream_ids'])*adapting*spec['stages']*50
    return dict(trajectories=trajectories,stage_rows=trajectories*spec['stages'],warmup_steps=warmup,candidate_steps=candidate,total_steps=candidate+warmup,source_pretraining_steps=0)
