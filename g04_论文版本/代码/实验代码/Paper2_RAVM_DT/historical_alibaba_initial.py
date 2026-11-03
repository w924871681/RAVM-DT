"""Rebuild Alibaba M0 in its original PyTorch 2.5.1 source environment.

The later 2.13 lifecycle consumes this M0; it did not originally train it.
Only paths/output metadata are adapted. The original windows/adapt functions run.
"""
import argparse,json,sys,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;T=B/'training'
sys.path[:0]=[str(T),str(T/'code/main/runtime')]
import numpy as np,torch
from core.models import Forecast,DEVICE,adapt,tensors,state_hash
from core.protocol import Segment,windows
from run import read,atomic,sha

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--center',type=int,action='append')
    a=p.parse_args();s=read(T/'groups/E3.json')
    if sys.version_info[:2]!=(3,11) or torch.__version__!='2.5.1+cu121':raise RuntimeError('Original Alibaba M0 requires Python 3.11 / torch 2.5.1+cu121')
    if a.output.exists():raise RuntimeError('Fresh initial-output directory required')
    a.output.mkdir(parents=True)
    centers=a.center or s['centers'];records=[]
    for center in centers:
        if center not in s['centers']:raise ValueError('Unknown center')
        with np.load(T/s['input_mapping'][f'data/center_{center}.npz']) as z:
            warm=Segment(z['x'][:800],z['y'][:800],z['ts'][:800],'warmup')
        model=Forecast('GRU',H=1).to(DEVICE)
        model.load_state_dict(torch.load(a.source,map_location=DEVICE,weights_only=True))
        if state_hash(model)!=s['sources'][str(a.seed)]['state_hash']:raise RuntimeError('Source state mismatch')
        model=adapt(model,tensors(windows(warm,H=1)),steps=50,seed=a.seed+center)
        target=a.output/f'{center}_{a.seed}.pt';torch.save(model.state_dict(),target)
        expected=read(T/s['input_mapping'][f'initial/{center}_{a.seed}.json'])
        meta=dict(sha256=sha(target),state_hash=state_hash(model),source_sha256=sha(a.source),warmup_hash=warm.hash,warmup_seed=a.seed+center,warmup_steps=50,python=sys.version,torch=str(torch.__version__))
        atomic(target.with_suffix('.json'),meta)
        matched=meta['state_hash']==expected['state_hash'] and meta['warmup_hash']==expected['warmup_hash']
        records.append(dict(center=center,seed=a.seed,status='PASS' if matched else 'FAIL',state_hash=meta['state_hash'],expected_state_hash=expected['state_hash']))
        atomic(a.output/'INITIAL_COMPARISON.json',dict(records=records,all_match=all(r['status']=='PASS' for r in records)))
        if not matched:raise RuntimeError(records[-1])
    print(json.dumps(dict(status='PASS',initials=len(records),source_environment_used=True)),flush=True)

if __name__=='__main__':main()
