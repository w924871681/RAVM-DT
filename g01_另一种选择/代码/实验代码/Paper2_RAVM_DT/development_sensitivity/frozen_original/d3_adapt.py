"""Training capability: matured Update only. No Service/mask/provider imports."""
import copy,hashlib,numpy as np,torch
from core.models import adapt,state_hash,tensors
from core.protocol import windows
from helpers.label_weight_reference import label_weights_from_update_raw,target_weights_for_windows
METHODS=('U_MSE','U_HUBER','U_LABEL_WEIGHT')
def batch_hash(n,seed):
 rng=np.random.default_rng(seed);h=hashlib.sha256()
 for _ in range(50):h.update(rng.integers(n,size=min(64,n)).astype('int64').tobytes())
 return h.hexdigest()
def target_weights(update):
 if update.role!='update':raise PermissionError('Only current matured Update accepted')
 w=label_weights_from_update_raw(update.y)
 return target_weights_for_windows(w),w
def weighted_adapter(parent,xy,w,seed,trace=False):
 if xy.role!='update':raise PermissionError('Forbidden training role')
 x,y=xy;weights=torch.as_tensor(w,device=x.device)
 assert y.ndim==2 and y.shape[1]==1 and len(weights)==len(y)
 model=copy.deepcopy(parent);model.train();opt=torch.optim.SGD(model.parameters(),lr=.01);rng=np.random.default_rng(seed);details=[]
 for _ in range(50):
  idx=torch.as_tensor(rng.integers(len(x),size=min(64,len(x))),device=x.device);opt.zero_grad(set_to_none=True);pred=model(x[idx]);wb=weights[idx].view(-1,1)
  loss=(wb*(pred-y[idx]).square()).sum()/wb.sum()
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite weighted training loss')
  loss.backward()
  if trace:details.append(dict(loss=float(loss.detach()),gradient=torch.cat([p.grad.detach().flatten() for p in model.parameters()]).cpu().numpy()))
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 return model,details
def make_candidate(parent,update,method,seed):
 if update.role!='update' or method!='U_MSE':raise PermissionError('Unregistered training source or generator')
 before=state_hash(parent);xy=tensors(windows(update));w,raw=target_weights(update)
 if method=='U_LABEL_WEIGHT':model,_=weighted_adapter(parent,xy,w,seed)
 else:model=adapt(parent,xy,steps=50,seed=seed,loss='huber' if method=='U_HUBER' else 'mse')
 assert state_hash(parent)==before
 return model,dict(generator=method,steps=50,lr=.01,batch=64,grad_clip=1.,training_seed=seed,sampled_indices_sha256=batch_hash(len(xy[0]),seed),update_hash=update.hash,weight_source='matured Update labels only',target_offset=96,horizon=1,target_weights_sha256=hashlib.sha256(w.tobytes()).hexdigest() if method=='U_LABEL_WEIGHT' else None,target_weight_mean=float(w.mean()) if method=='U_LABEL_WEIGHT' else 1.,target_downweighted_fraction=float(np.mean(w<1)) if method=='U_LABEL_WEIGHT' else 0.,raw_weight_min=float(raw.min()) if method=='U_LABEL_WEIGHT' else 1.,raw_weight_quantiles=np.quantile(raw,[0,.1,.25,.5,.75,.9,1]).tolist() if method=='U_LABEL_WEIGHT' else [1.]*7)
