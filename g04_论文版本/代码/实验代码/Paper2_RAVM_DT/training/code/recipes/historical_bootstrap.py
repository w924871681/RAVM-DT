import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json, hashlib, platform, subprocess
import numpy as np
import torch
from vendor.main_cfg import MainCfg
from vendor.sim import simulate_centers
from core.protocol import ROOT,POOLS,REGIMES,digest,save_json

def generate(pool):
    cfg=MainCfg();cfg.sim.seed=7311;cfg.sim.n_history=17000
    for kind in 'ABC':getattr(cfg.sim,'type_'+kind).drift_prob=0.
    ids=POOLS[pool]
    centers=simulate_centers(cfg,len(ids),seed_offset=ids[0]*17,start_center_id=ids[0])
    for i,c in enumerate(centers):
        regime='stationary' if pool=='source' else REGIMES[i%4]
        t=np.arange(len(c.y));shift=np.zeros(len(t))
        onset=1400+8*640
        if regime=='abrupt':shift[t>=onset]=.8
        if regime=='gradual':shift=.8*np.clip((t-onset)/(8*640),0,1)
        if regime=='recurring':shift=.8*((t>=1400+6*640)&(t<1400+12*640) | (t>=1400+18*640))
        # Coupled latent-logit and metric shifts preserve the original metric mapping coefficients.
        eps=1e-5
        def logit(a):return np.log(np.clip(a,eps,1-eps)/(1-np.clip(a,eps,1-eps)))
        def sig(a):return 1/(1+np.exp(-a))
        y=sig(logit(c.y)+shift).astype('float32')
        x=np.zeros_like(c.X)
        for k,gain in enumerate(c.metadata['metric_gains']):
            delay=c.metadata['metric_delays'][k]//c.dt
            dshift=np.concatenate([np.repeat(shift[0],delay),shift[:len(shift)-delay]]) if delay else shift
            x[:,k]=(sig(logit(c.X[:,k])+gain*dshift)*c.M[:,k]).astype('float32')
        x=np.concatenate([x,c.M],1)
        meta={'center_id':c.center_id,'center_type':c.center_type,'drift_regime':regime,'pool':pool,'dt':c.dt,'data_seed':7311,'clean_hash':digest(x,y,c.timestamps),'base_generator':c.metadata}
        np.savez_compressed(ROOT/'data'/f'center_{c.center_id}.npz',x=x,y=y,ts=c.timestamps,meta=json.dumps(meta))
    print('generated',pool,len(centers),flush=True)

if __name__=='__main__':
    save_json(ROOT/'audits/environment.json',{'python':sys.executable,'python_version':platform.python_version(),'torch':torch.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'platform':platform.platform()})
    save_json(ROOT/'pool_manifest.json',{'pools':POOLS,'stress_subset':POOLS['heldout'][:20],'diagnostic_subset':POOLS['heldout'][:10],'generality_subset':POOLS['heldout'][:10],'data_seed':7311,'model_seeds':[7311,7312,7313]})
    # Heldout generation is deferred until configurations and audits have been frozen.
    for pool in ['source','development','calibration']:
        if not all((ROOT/'data'/f'center_{c}.npz').exists() for c in POOLS[pool]):generate(pool)
