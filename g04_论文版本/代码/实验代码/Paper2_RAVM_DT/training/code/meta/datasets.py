"""Registered data creation only after the global science execution seal."""
from common import *
import shutil
import numpy as np
REGIMES=('stationary','gradual','abrupt','recurring')
def require_seal():
    lock=read(ROOT/'SCIENCE_EXECUTION.lock.json');assert lock['status']=='SEALED_FOR_REGISTERED_COVERAGE'
    return sha(ROOT/'SCIENCE_EXECUTION.lock.json')
def synthetic(spec):
    require_seal()
    import sys
    sys.path.insert(0,str(PRIOR/'runtime'))
    from vendor.main_cfg import MainCfg
    from vendor.sim import simulate_centers
    target=Path(spec['root'])/'data';target.mkdir(parents=True,exist_ok=True)
    canonical=STORAGE/'inputs'/('E1' if spec['stage']=='E1' else 'E2');canonical.mkdir(parents=True,exist_ok=True)
    identity=canonical/'INPUT_IDENTITY.json'
    if identity.exists():
        record=read(identity)
        assert record['execution_lock_sha256']==require_seal()
        for row in record['rows']:assert sha(canonical/row['name'])==row['sha256']
    else:
        centers=spec['centers'];cfg=MainCfg();cfg.sim.seed=7311;cfg.sim.n_history=17000
        for kind in 'ABC':getattr(cfg.sim,'type_'+kind).drift_prob=0.
        raw=simulate_centers(cfg,len(centers),seed_offset=centers[0]*17,start_center_id=centers[0]);rows=[]
        def logit(a):a=np.clip(a,1e-5,1-1e-5);return np.log(a/(1-a))
        def sigmoid(a):return 1/(1+np.exp(-a))
        for index,(center,source) in enumerate(zip(centers,raw)):
            regime=REGIMES[index%4];t=np.arange(len(source.y));shift=np.zeros(len(t));onset=1400+8*640
            if regime=='abrupt':shift[t>=onset]=.8
            elif regime=='gradual':shift=.8*np.clip((t-onset)/(8*640),0,1)
            elif regime=='recurring':shift=.8*(((t>=1400+6*640)&(t<1400+12*640))|(t>=1400+18*640))
            y=sigmoid(logit(source.y)+shift).astype('float32');x=np.zeros_like(source.X)
            for k,gain in enumerate(source.metadata['metric_gains']):
                delay=source.metadata['metric_delays'][k]//source.dt
                ds=np.concatenate([np.repeat(shift[0],delay),shift[:len(shift)-delay]]) if delay else shift
                x[:,k]=(sigmoid(logit(source.X[:,k])+gain*ds)*source.M[:,k]).astype('float32')
            x=np.concatenate([x,source.M],1);path=canonical/f'center_{center}.npz'
            meta=dict(center_id=center,regime_audit_only=regime,pool=spec['stage']+'_REGISTERED_NEW_SYNTHETIC',dt=source.dt,data_seed=7311,seed_offset=centers[0]*17,base_generator=source.metadata)
            if path.exists():
                with np.load(path) as z:assert np.array_equal(z['x'],x) and np.array_equal(z['y'],y) and np.array_equal(z['ts'],source.timestamps)
            else:
                with path.open('xb') as f:np.savez_compressed(f,x=x,y=y,ts=source.timestamps,meta=json.dumps(meta));f.flush();os.fsync(f.fileno())
            rows.append(dict(center=center,name=path.name,sha256=sha(path),regime=regime))
        record=dict(status='GENERATED_ONCE_AFTER_SEAL',execution_lock_sha256=require_seal(),rows=rows,at_unix=time.time())
        atomic(identity,record)
    for row in record['rows']:
        src=canonical/row['name'];dst=target/row['name']
        if dst.exists():assert sha(dst)==row['sha256']
        else:
            try:os.link(src,dst)
            except OSError:
                if not dst.exists():shutil.copyfile(src,dst)
                assert sha(dst)==row['sha256']
    atomic(target/'INPUT_IDENTITY.json',record)
    return record
def external(spec):
    require_seal();target=Path(spec['root'])/'data';target.mkdir(parents=True,exist_ok=True);rows=[]
    for row in spec['source_data']:
        src=Path(row['source_path']);assert sha(src)==row['sha256'];dst=target/src.name
        if dst.exists():assert sha(dst)==row['sha256']
        else:shutil.copyfile(src,dst);assert sha(dst)==row['sha256']
        rows.append(dict(center=row['center'],sha256=row['sha256'],source=str(src)))
    identity=dict(status='EXACT_AUTHENTICATED_EXTERNAL_INPUT_COPY',execution_lock_sha256=require_seal(),rows=rows,at_unix=time.time())
    atomic(target/'INPUT_IDENTITY.json',identity);return identity
