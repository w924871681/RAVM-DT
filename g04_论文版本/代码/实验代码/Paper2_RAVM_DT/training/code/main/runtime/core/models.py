import copy, hashlib, time
import numpy as np
import torch
from torch import nn

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

class Forecast(nn.Module):
    def __init__(self,backbone='GRU',H=1,features=24):
        super().__init__();self.backbone=backbone;self.H=H
        if backbone=='GRU':self.body=nn.GRU(features,32,batch_first=True);self.head=nn.Linear(32,H)
        elif backbone=='MLP':self.body=nn.Sequential(nn.Flatten(),nn.Linear(96*features,64),nn.ReLU(),nn.Linear(64,H))
        elif backbone=='TCN':
            self.body=nn.Sequential(nn.Conv1d(features,32,3,padding=2),nn.ReLU(),nn.Conv1d(32,32,3,dilation=2,padding=4),nn.ReLU());self.head=nn.Linear(32,H)
        else:raise ValueError(backbone)
    def forward(self,x):
        if self.backbone=='GRU':
            self.body.flatten_parameters()
            return self.head(self.body(x)[0][:,-1])
        if self.backbone=='MLP':return self.body(x)
        # Trim right padding at each convolution for a causal stack.
        h=self.body[1](self.body[0](x.transpose(1,2))[:,:,:-2])
        h=self.body[3](self.body[2](h)[:,:,:-4]);return self.head(h[:,:,-1])

def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class Batch(tuple):
    def __new__(cls,x,y,role):
        obj=super().__new__(cls,(x,y));obj.role=role;return obj

def tensors(w):return Batch(torch.as_tensor(w[0],device=DEVICE),torch.as_tensor(w[1],device=DEVICE),getattr(w,'role','source'))
def sync():
    if DEVICE=='cuda':torch.cuda.synchronize()

def adapt(parent,xy,steps=50,seed=0,loss='mse',ewc=None,lam=0.):
    assert getattr(xy,'role',None) in ('update','warmup','replay_training'), 'forbidden candidate training role'
    model=copy.deepcopy(parent);model.train();x,y=xy
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    rng=np.random.default_rng(seed)
    for _ in range(steps):
        idx=torch.as_tensor(rng.integers(len(x),size=min(64,len(x))),device=DEVICE)
        opt.zero_grad(set_to_none=True);pred=model(x[idx])
        l=(nn.functional.huber_loss(pred,y[idx],delta=.1) if loss=='huber' else nn.functional.mse_loss(pred,y[idx]))
        if ewc is not None:
            anchor,fisher=ewc
            l=l+.5*lam*sum((fisher[k]*(p-anchor[k]).square()).sum() for k,p in model.named_parameters())
        if not torch.isfinite(l):raise FloatingPointError('nonfinite training loss')
        l.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    return model

def robust_candidate(parent,xy,B,beta,seed,mean=False,H=1):
    from core.protocol import block_indices
    x,y=xy;blocks=block_indices(len(x),B,H);deltas=[]
    initial=state_hash(parent)
    for b,ids in enumerate(blocks):
        assert len(ids)>0
        c=adapt(parent,Batch(x[ids],y[ids],xy.role),50//B,seed+b)
        deltas.append({k:v-parent.state_dict()[k] for k,v in c.state_dict().items()})
        assert state_hash(parent)==initial
    candidate=copy.deepcopy(parent);state={};trim=0 if mean else int(B*beta)
    for k,v in parent.state_dict().items():
        d=torch.stack([a[k] for a in deltas]);d=d.sort(dim=0).values
        state[k]=v+d[trim:B-trim].mean(0)
    candidate.load_state_dict(state);return candidate

@torch.no_grad()
def errors(model,xy):
    model.eval();x,y=xy;pred=torch.cat([model(x[i:i+256]) for i in range(0,len(x),256)])
    return (pred-y).square().mean(1).cpu().numpy(),(pred-y).abs().mean(1).cpu().numpy()

def fisher_diag(model,xy,seed=0,n=32):
    assert xy.role in ('update','warmup','replay_training')
    # Empirical diagonal Fisher for unit-variance Gaussian regression, per-example scores.
    model.train();x,y=xy;out={k:torch.zeros_like(p) for k,p in model.named_parameters()}
    ids=np.random.default_rng(seed).choice(len(x),min(n,len(x)),replace=False)
    for i in ids:
        model.zero_grad(set_to_none=True);l=.5*(model(x[i:i+1])-y[i:i+1]).square().sum();l.backward()
        for k,p in model.named_parameters():out[k]+=p.grad.detach().square()/len(ids)
    model.zero_grad(set_to_none=True);return out
