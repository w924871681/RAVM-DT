import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time,argparse
import numpy as np
import torch
from core.protocol import ROOT,POOLS,Timeline,windows,save_json
from core.models import Forecast,DEVICE,tensors,state_hash

def pretrain(seed=7311,backbone='GRU',H=1,smoke=False,domain='synthetic'):
    prefix='alibaba_' if domain=='alibaba' else ''
    path=ROOT/'checkpoints'/f'{prefix}{"smoke_" if smoke else ""}source_{backbone}_H{H}_{seed}.pt'
    if path.exists():return
    torch.manual_seed(seed);np.random.seed(seed)
    model=Forecast(backbone,H).to(DEVICE)
    centers=list(range(20000,20020)) if domain=='alibaba' else POOLS['source'][:2] if smoke else POOLS['source']
    ws=[windows(Timeline(c).source(),H=H) for c in centers]
    x,y=tensors((np.concatenate([w[0] for w in ws]),np.concatenate([w[1] for w in ws])))
    opt=torch.optim.Adam(model.parameters(),lr=.001);epochs=1 if smoke else 50
    rng=np.random.default_rng(seed);start=time.time()
    for epoch in range(epochs):
        ids=rng.permutation(len(x));total=0.
        for i in range(0,len(ids),64):
            idx=ids[i:i+64];opt.zero_grad(set_to_none=True);loss=(model(x[idx])-y[idx]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('source pretraining loss')
            loss.backward();opt.step();total+=float(loss.detach())*len(idx)
        print(f'pretrain {backbone} H={H} seed={seed} epoch={epoch+1}/{epochs} mse={total/len(x):.8g} elapsed={time.time()-start:.1f}s',flush=True)
        save_json(ROOT/'logs/pretrain_progress.json',{'backbone':backbone,'H':H,'seed':seed,'epoch':epoch+1,'epochs':epochs,'elapsed_s':time.time()-start,'smoke':smoke})
    torch.save(model.state_dict(),path)
    save_json(path.with_suffix('.json'),{'seed':seed,'backbone':backbone,'horizon':H,'epochs':epochs,'source_centers':centers,'raw_segment':[0,1000],'source_state_hash':state_hash(model),'elapsed_s':time.time()-start,'smoke':smoke})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=7311);p.add_argument('--backbone',default='GRU');p.add_argument('--horizon',type=int,default=1);a=p.parse_args();pretrain(a.seed,a.backbone,a.horizon,a.smoke)
