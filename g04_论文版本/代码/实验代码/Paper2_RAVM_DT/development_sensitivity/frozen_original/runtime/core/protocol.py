"""Raw-first causal lifecycle. Evaluation access is gated by a decision token."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # Stage3A isolated execution root
POOLS = {'source': list(range(10000,10020)), 'development':list(range(11000,11020)),
         'calibration':list(range(12000,12020)), 'heldout':list(range(13000,13030))}
REGIMES = ['stationary','gradual','abrupt','recurring']
ATTACKS = ['random','persistent_bias','burst','replay']

def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()

def save_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def spans(stage):
    assert 1<=stage<=24
    v=1400+(stage-1)*640
    return ((1000,1400) if stage==1 else (v-400,v), (v,v+240), (v+240,v+640))

@dataclass
class Segment:
    x: np.ndarray
    y: np.ndarray
    ts: np.ndarray
    role: str
    def copy(self,role=None):
        return Segment(self.x.copy(),self.y.copy(),self.ts.copy(),role or self.role)
    @property
    def hash(self):return digest(self.x,self.y,self.ts)

class WindowSet(tuple):
    def __new__(cls,x,y,ts,role):
        obj=super().__new__(cls,(x,y,ts));obj.role=role;return obj

def windows(seg,L=96,H=1):
    n=len(seg.y)-L-H+1
    if n<=0:raise ValueError('segment too short')
    x=np.lib.stride_tricks.sliding_window_view(seg.x,L,axis=0)[:n].transpose(0,2,1).copy()
    y=np.stack([seg.y[L+i:L+i+n] for i in range(H)],1)
    ts=np.stack([seg.ts[L+i:L+i+n] for i in range(H)],1)
    return WindowSet(x,y,ts,seg.role)

def block_indices(n,B,H=1):
    # For H>1 drop the final H-1 origins of each internal block to keep targets disjoint.
    chunks=np.array_split(np.arange(n),B)
    return [c[:max(0,len(c)-H+1)] if j<B-1 else c for j,c in enumerate(chunks)]

class Timeline:
    def __init__(self,center):
        self.center=center
        z=np.load(ROOT/'data'/f'center_{center}.npz')
        self.__x=z['x'];self.__y=z['y'];self.__ts=z['ts']
        self.meta=json.loads(str(z['meta']))
        for a in (self.__x,self.__y,self.__ts):a.flags.writeable=False
        self.__decisions={}
    def _segment(self,a,b,role):
        return Segment(self.__x[a:b].copy(),self.__y[a:b].copy(),self.__ts[a:b].copy(),role)
    def warmup(self):return self._segment(0,800,'warmup')
    def initial_history(self):return self._segment(800,1000,'verification')
    def source(self):
        assert self.meta['pool']=='source'
        return self._segment(0,1000,'source')
    def maintenance(self,t):
        if t>1:assert t-1 in self.__decisions,'previous service labels not yet matured'
        u,v,s=spans(t)
        return self._segment(*u,'update'),self._segment(*v,'verification')
    def past_replay(self,t,length):
        # Only past warm-up telemetry, never current/future service or hidden regime labels.
        a=max(0,800-length);return self._segment(a,800,'replay_source')
    def record_decision(self,t,commit):
        assert t not in self.__decisions
        self.__decisions[t]=bool(commit)
        assert self.__ts[spans(t)[1][1]-1]<self.__ts[spans(t)[2][0]]
    def service(self,t):
        assert t in self.__decisions,'future service forbidden before decision'
        return self._segment(*spans(t)[2],'service')

def attack(seg,center,stage,kind,alpha,location,seed=9417):
    assert seg.role in ('update','verification')
    out=seg.copy();n=len(seg.y);mask=np.zeros(n,bool);source_ts=np.full(n,-1,dtype=np.int64)
    active= location=='both' or (location=='update_only' and seg.role=='update') or (location=='verification_only' and seg.role=='verification')
    # Independent of model/method/model seed. Balanced 4-type schedule in each four-stage cycle.
    if kind=='mixed':
        order=np.random.default_rng(np.random.SeedSequence([seed,center,(stage-1)//4])).permutation(4)
        kind=ATTACKS[order[(stage-1)%4]]
    rng=np.random.default_rng(np.random.SeedSequence([seed,center,stage,0 if seg.role=='update' else 1]))
    # Per chronological raw-observation stratum caps, common to B=5 and B=10.
    # Burst is contiguous within each bounded corruption block, not a global burst violating block caps.
    if active and alpha>0:
        for b in np.array_split(np.arange(n),10):
            k=int(np.floor(len(b)*alpha+1e-9))
            if k:
                ids=(b[int(rng.integers(0,len(b)-k+1)):][:k] if kind=='burst' else rng.choice(b,k,replace=False))
                mask[ids]=True
        dims=out.x.shape[1]//2
        obs=out.x[:,:dims];obs_mask=out.x[:,dims:]
        sign=np.random.default_rng(np.random.SeedSequence([seed,center,stage])).choice([-1,1])
        if kind=='replay':
            old=Timeline(center).past_replay(stage,n)
            out.x[mask]=old.x[mask];out.y[mask]=old.y[mask];source_ts[mask]=old.ts[mask]
            assert np.all(source_ts[mask]<seg.ts[0])
        else:
            selected=rng.choice(dims,max(1,dims//2),replace=False)
            change=np.zeros_like(obs)
            if kind=='random':
                change[mask]=np.clip(rng.normal(0,.15,(int(mask.sum()),dims)),-.2,.2)
                out.y[mask]=np.clip(out.y[mask]+np.clip(rng.normal(0,.15,int(mask.sum())),-.2,.2),0,1)
            else:
                change[np.ix_(mask,selected)]=sign*.2
                out.y[mask]=np.clip(out.y[mask]+sign*.2,0,1)
            obs[:]=np.clip(obs+change*obs_mask,0,1)*obs_mask
    assert mask.sum()<=np.floor(n*alpha+1e-8) if active else not mask.any()
    audit_dir=ROOT/'data/attack_masks';audit_dir.mkdir(exist_ok=True)
    key=hashlib.sha256(f'{center}:{stage}:{seg.role}:{kind}:{alpha}:{location}:{seed}'.encode()).hexdigest()
    path=audit_dir/(key+'.npz')
    if not path.exists():
        tmp=path.with_name(f'{path.name}.{os.getpid()}.{__import__("threading").get_ident()}.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,mask=mask,source_timestamps=source_ts,target_timestamps=seg.ts,
                                center=center,stage=stage,role=seg.role,kind=kind,alpha=alpha,location=location)
        try:
            if not path.exists():os.replace(tmp,path)
        except PermissionError:
            if not path.exists():raise
        finally:tmp.unlink(missing_ok=True)
    return out,mask,source_ts,kind
