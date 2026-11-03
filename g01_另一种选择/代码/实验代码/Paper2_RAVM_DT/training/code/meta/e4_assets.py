"""Authenticated historical temperature recipes; no source fitting/calibration."""
from common import *
import importlib.util,sys
_MODULES={}
def original(domain):
    if domain in _MODULES:return _MODULES[domain]
    folder=ROOT.parent/('paper2_revision_execution_20261012' if domain=='uci' else 'paper2_revision_round2_20261013')
    file=folder/'code'/('revision_building.py' if domain=='uci' else 'thermal_study.py')
    lock=read(folder/'audit'/('building_protocol_locked.json' if domain=='uci' else 'thermal_protocol_locked.json'))
    assert sha(file)==lock['code_sha256'],('Original recipe changed',str(file))
    sys.path.insert(0,str(folder/'code'))
    spec=importlib.util.spec_from_file_location('_coverage_recipe_'+domain,file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _MODULES[domain]=module;return module

def authenticate():
    import bridge as b
    import numpy as np
    from stage_specs import base,all_arms
    arms=[a for a in all_arms() if a['id'] in ('JOINT__l3__m0','RECENT_MEAN__g12','MWU_MEAN__g04','DUAL_MEAN__g12','RAVM_ORIGINAL','ALWAYS_FT','FROZEN')]
    groups=[];receipts=[]
    for domain in ('uci','thermbuild'):
        m=original(domain);lock=read(m.BASE/'audit'/('building_protocol_locked.json' if domain=='uci' else 'thermal_protocol_locked.json'))
        if domain=='uci':
            assert sha(m.CSV)==lock['data_hash'];items=m.ROOMS
        else:
            inventory=read(m.AUDIT/'thermal_data_inventory.json');assert sha(m.AUDIT/'thermal_data_inventory.json')==lock['data_inventory_sha256']
            for rel,digest in lock['dependency_sha256'].items():assert sha(m.BASE/'code'/rel)==digest
            items=m.BUILDINGS
        for index,item in enumerate(items):
            center=9000+index*1000 if domain=='uci' else 70001+index
            group=f'E4_{domain}_{item}'
            conditions=m.CONDITIONS
            spec=base(group,'E4',[center],{c:dict(kind=c,alpha=0. if c in ('clean','unmodified') else .1,location='both') for c in conditions},arms,domain=domain,features=40 if domain=='uci' else 26)
            spec.update(physical_cluster='one_home_uci' if domain=='uci' else item,building=item,room_index=index,building_index=index+1)
            if domain=='uci':
                spec.update(target_scale=float(m.STD[index*2]),target_mean=float(m.MEAN[index*2]),raw_stages=list(range(1,25)),
                    role_bounds=dict(warmup=[9000,9300],initial_history=[9300,9500]),
                    role_schedule=[[list(p) for p in ((9500+(t-1)*360,9700+(t-1)*360),(9700+(t-1)*360,9860+(t-1)*360),(9860+(t-1)*360,10060+(t-1)*360))] for t in range(1,25)])
                data_files=[dict(path=str(m.CSV),sha256=sha(m.CSV))]
                refs={x['seed']:x for x in lock['source_checkpoints'] if x['room']==item}
            else:
                data=m.load_building(item);cov=m.coverage(data,'final')
                old=next(x for x in inventory['coverage'] if x['building']==item and x['pool']=='final');assert cov==old
                for row in next(x for x in inventory['data'] if x['building']==item)['files']:assert sha(m.BASE/row['path'])==row['sha256']
                assert cov['pool_eligible'],'Original fixed final coverage unavailable; no alternate start'
                lo=cov['row_bounds'][0]
                spec.update(stages=cov['retained_stages'],target_scale=data['target_scale'],target_mean=data['target_mean'],raw_stages=[r['raw_stage']for r in cov['retained']],role_bounds=dict(warmup=[lo,lo+800],initial_history=[lo+800,lo+1000]),role_schedule=[r['spans'] for r in cov['retained']],coverage=cov)
                data_files=[dict(path=str(m.BASE/r['path']),sha256=r['sha256']) for r in data['files']]
                refs={x['seed']:x for x in lock['source_provenance'] if x['building']==item}
            sources={}
            for seed in spec['seeds']:
                path=m.CHECK/f'source_{item}_{seed}.pt';ref=refs[seed];digest=ref.get('checkpoint_hash',ref.get('checkpoint_sha256'));expected=ref.get('state_hash',ref.get('source_state_hash'))
                assert sha(path)==digest
                model=b.NativeForecast('GRU',H=1,features=spec['features']);model.load_state_dict(b.torch.load(path,map_location='cpu',weights_only=True));assert b.state_hash(model)==expected
                with b.torch.no_grad():assert tuple(model(b.torch.zeros(2,96,spec['features'])).shape)==(2,1)
                del model;sources[str(seed)]=dict(path=str(path),sha256=digest,state_hash=expected)
            spec['sources']=sources;spec['data_files']=data_files;spec['original_protocol_sha256']=sha(m.BASE/'audit'/('building_protocol_locked.json' if domain=='uci' else 'thermal_protocol_locked.json'))
            # Every previous service has matured before a later Update can reuse it.
            for t,parts in enumerate(spec['role_schedule']):
                u,v,s=parts;assert u[1]==v[0] and v[1]==s[0] and s[1]<=len(m.X if domain=='uci' else data['x'])
                if t:assert spec['role_schedule'][t-1][2][1]<=u[1]
            groups.append(spec);receipts.append(dict(group=group,status='PASS',stages=spec['stages'],features=spec['features'],sources=sources,data_files=data_files,physical_cluster=spec['physical_cluster']))
    atomic(ROOT/'assets/E4_TEMPERATURE_AUTH.json',dict(status='PASS',groups=groups,rows=receipts,optimizer_steps=0,old_negative_results_preserved=True,at_unix=time.time()))
    return groups

def prepare_data(spec):
    import numpy as np
    from datasets import require_seal
    require_seal();m=original(spec['domain']);center=spec['centers'][0]
    for row in spec['data_files']:assert sha(row['path'])==row['sha256']
    if spec['domain']=='uci':x=m.X;y=x[:,spec['room_index']*2];ts=m.TS
    else:
        data=m.load_building(spec['building']);x,y,ts=(data[k] for k in ('x','y','ts'));assert m.coverage(data,'final')==spec['coverage']
    target=Path(spec['root'])/'data'/f'center_{center}.npz';meta=dict(pool='historical_external_final',regime_audit_only=spec['building'],domain=spec['domain'],physical_cluster=spec['physical_cluster'],original_files=spec['data_files'])
    if not target.exists():
        with target.open('xb') as f:np.savez_compressed(f,x=x,y=y,ts=ts,meta=json.dumps(meta));f.flush();os.fsync(f.fileno())
    else:
        with np.load(target) as z:assert np.array_equal(z['x'],x,equal_nan=True) and np.array_equal(z['y'],y,equal_nan=True) and np.array_equal(z['ts'],ts)
    atomic(target.parent/'INPUT_IDENTITY.json',dict(rows=[dict(center=center,sha256=sha(target))],execution_lock_sha256=require_seal(),no_fabricated_source_weights=True))

def attach(spec):
    """Thread local role/fault callables; raw recipe data remain read only."""
    if spec['domain'] not in ('uci','thermbuild'):return spec
    import bridge as b
    from immutable_cache import arrays
    import numpy as np
    m=original(spec['domain']);center=spec['centers'][0];x,y,ts,_=arrays(Path(spec['root'])/'data'/f'center_{center}.npz')
    def bounds(t):return spec['role_schedule'][t-1]
    def roles(c,role,t):
        assert c==center
        if role in spec['role_bounds']:a,z=spec['role_bounds'][role]
        else:a,z=bounds(t)[{'update':0,'verification':1,'service':2}[role]]
        return b.Segment(x[a:z].copy(),y[a:z].copy(),ts[a:z].copy(),'verification' if role=='initial_history' else role)
    def fault(seg,c,t,stream):
        picked=np.zeros(len(seg.y),bool)
        if spec['domain']=='uci':
            if stream!='clean':
                rng=np.random.default_rng(np.random.SeedSequence([9487,spec['room_index'],t,0 if seg.role=='update' else 1]))
                for block in np.array_split(np.arange(len(seg.y)),10):
                    n=int(np.floor(len(block)*.10))
                    if n:picked[rng.choice(block,size=n,replace=False)]=True
            out,n=m.attack(seg,spec['building'],stream,t);assert n==int(picked.sum());fault.last_mask=picked;return out
        # Exact original fault function and original raw-stage seed.
        holder=object.__new__(m.ThermalProvider);holder.data=dict(building=spec['building'],target_scale=spec['target_scale']);holder.lo=spec['role_bounds']['warmup'][0];holder.condition=stream
        raw=spec['raw_stages'][t-1]
        if stream!='unmodified':
            rng=np.random.default_rng(np.random.SeedSequence([27013,spec['building_index'],holder.lo,raw,0 if seg.role=='update' else 1]));a=int(rng.integers(0,len(seg.y)-4+1));picked[a:a+4]=True
        out,n=holder._fault(seg,raw);assert n==int(picked.sum());fault.last_mask=picked;return out
    return dict(spec,timeline_roles=roles,thermal_spans=bounds,fault_adapter=fault)

if __name__=='__main__':
    import bridge as b
    b.torch.optim.SGD.step=lambda *a,**k:(_ for _ in ()).throw(PermissionError('Authentication is zero SGD'))
    authenticate()
