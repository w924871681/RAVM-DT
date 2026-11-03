"""Explicit data/shape port of frozen CMC code; optimizer and gate math reused.

Thread-local contexts isolate dataset, architecture and horizon. Imported source
bytes are never edited. Derivations are recorded before any new scientific run.
"""
from common import ROOT as META,PRIOR,STORAGE,read as read_meta,sha as sha_meta
from pathlib import Path
import ast,collections,contextlib,copy,gc,hashlib,inspect,io,json,math,os,sys,threading,time
from dataclasses import asdict
import numpy as np
import torch
sys.path.insert(0,str(PRIOR/'runtime'));sys.path.append(str(PRIOR))
import core.protocol as protocol
from core.models import Forecast as NativeForecast,DEVICE,adapt,errors,state_hash,tensors,sync,Batch
from cmc_original_gate_adapter import OriginalCMCGate as NativeGate
from stage3_engine import predictions_losses,score_arrays
from d2c_core import AuditLedger,FeedbackVerifier,LateAudit,VerifiedFeedbackView,view_message,digest as digest_bytes
from d3_gate import Evidence,decide,baseline_decision,research_decision
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey,Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
TLS=threading.local();TLS.context={}
def ctx():return getattr(TLS,'context',{})
def current_root():return Path(ctx().get('root',STORAGE/ctx().get('group','UNASSIGNED')))
class ContextRoot:
    def __truediv__(self,p):return current_root()/p
    def __fspath__(self):return str(current_root())
    def __str__(self):return str(current_root())
ROOT=ContextRoot()
protocol.ROOT=ROOT
from core.protocol import Segment,digest,spans as native_spans,windows as native_windows,Timeline as BaseTimeline,attack as native_attack
import core.engine as native
NATIVE_RUN=native.run;NATIVE_FISHER=native.fisher_diag
ORIGINAL_FUNCTIONS={}

def derive(file,names,transform=None):
    source=(PRIOR/file).read_text(encoding='utf8');tree=ast.parse(source);nodes=[]
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:
            ORIGINAL_FUNCTIONS[file+':'+n.name]=hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest()
            nodes.append(n)
    assert {n.name for n in nodes}==set(names),(file,names)
    tree=ast.Module(body=nodes,type_ignores=[])
    if transform:tree=transform.visit(tree)
    exec(compile(ast.fix_missing_locations(tree),str(PRIOR/file),'exec'),globals())

derive('d3_common.py',['canonical','sha','read','_replace_retry','atomic'])
unbuffered_sha=sha
def sha(path):
    if Path(path).suffix.lower()in('.npz','.pt'):
        from immutable_cache import hash_file
        return hash_file(path)
    return unbuffered_sha(path)
import tempfile

@contextlib.contextmanager
def context(spec):
    old=ctx();TLS.context=spec
    try:yield
    finally:TLS.context=old

def horizon():return int(ctx().get('horizon',1))
def stage_limit():return int(ctx().get('stages',24))
def windows(seg,L=96,H=None):
    w=native_windows(seg,L=L,H=horizon() if H is None else H)
    if ctx().get('domain')=='thermbuild':
        from core.protocol import WindowSet
        valid=np.isfinite(w[1]).all(axis=1)
        return WindowSet(w[0][valid],w[1][valid],w[2][valid],seg.role)
    return w
def Forecast(*a,**kw):
    if not a and not kw:return NativeForecast(ctx().get('architecture','GRU'),H=horizon(),features=ctx().get('features',24))
    return NativeForecast(*a,**kw)
from core.engine import evidence_blocks as native_blocks
def evidence_blocks(seg,H=None,B=6):
    # Partition original chronological origins before filtering missing targets.
    h=horizon() if H is None else H
    from core.protocol import block_indices
    chunks=block_indices(len(native_windows(seg,H=h)[0]),B,h);result=[]
    for ids in chunks:
        if len(ids):
            a=int(ids[0]);z=int(ids[-1])+96+h
            result.append(Segment(seg.x[a:z].copy(),seg.y[a:z].copy(),seg.ts[a:z].copy(),'verification'))
    return result
def spans(t):
    provider=ctx().get('thermal_spans')
    return provider(t) if provider else native_spans(t)

def training_seed(seed,center,t):
    domain=ctx().get('domain')
    if domain=='uci':return seed+ctx()['room_index']+t*100 if t else seed+ctx()['room_index']*1000+9000
    if domain=='thermbuild':return seed+ctx()['building_index']*100000+ctx()['role_bounds']['warmup'][0]+(ctx()['raw_stages'][t-1]*100 if t else 0)
    return seed+center+t*100
def regime_for(center):
    return Timeline(center).meta['regime_audit_only']
def source_claim():return 'SIMULATED_SOURCE_ATTESTED_NOT_FIELD' if ctx().get('domain','synthetic')=='synthetic' else 'OFFLINE_EXTERNAL_ROLE_CONTROL_NOT_FIELD_ATTESTATION'
def authority_issuer():return 'SIMULATED_CAUSAL_LABEL_PROVIDER_ONLY' if ctx().get('domain','synthetic')=='synthetic' else 'OFFLINE_EXTERNAL_LABEL_ROLE_CONTROLLER'
class SourceMap:
    def __getitem__(self,seed):return ctx()['sources'][str(seed)]
SOURCE=SourceMap()
class IDMap:
    def __getitem__(self,phase):return ctx()['centers']
    def values(self):return [ctx()['centers']]
IDS=IDMap()
class StreamMap:
    def __contains__(self,value):return value in ctx().get('stream_ids',[])
    def __iter__(self):return iter(ctx().get('stream_ids',[]))
STREAMS=StreamMap();REGIMES=('stationary','gradual','abrupt','recurring')
def execution_lock_sha():return sha(META/'SCIENCE_EXECUTION.lock.json')
def allowed_execution_shas():return {execution_lock_sha()}
class Timeline(BaseTimeline):
    def __init__(self,center):
        from immutable_cache import arrays
        self.center=center
        self._Timeline__x,self._Timeline__y,self._Timeline__ts,self.meta=arrays(ROOT/'data'/f'center_{center}.npz')
        self._Timeline__decisions={}
        self.meta={**self.meta,'regime_audit_only':self.meta.get('regime_audit_only',self.meta.get('drift_regime','external'))}
    def warmup(self):
        custom=ctx().get('timeline_roles')
        return custom(self.center,'warmup',0) if custom else super().warmup()
    def past_replay(self,t,length):
        if not ctx().get('timeline_roles'):return super().past_replay(t,length)
        s=self.warmup();n=min(length,len(s.y));return Segment(s.x[-n:].copy(),s.y[-n:].copy(),s.ts[-n:].copy(),'replay_source')
    def initial_history(self):
        custom=ctx().get('timeline_roles')
        return custom(self.center,'initial_history',0) if custom else super().initial_history()
    def maintenance(self,t):
        TLS.science_stage=t
        custom=ctx().get('timeline_roles')
        if not custom:return super().maintenance(t)
        if t>1:assert t-1 in self._Timeline__decisions
        return custom(self.center,'update',t),custom(self.center,'verification',t)
    def record_decision(self,t,commit):
        if not ctx().get('timeline_roles'):return super().record_decision(t,commit)
        assert t not in self._Timeline__decisions
        self._Timeline__decisions[t]=bool(commit)
    def service(self,t):
        custom=ctx().get('timeline_roles')
        if not custom:return super().service(t)
        assert t in self._Timeline__decisions
        return custom(self.center,'service',t)
def observed(seg,center,t,stream):
    if ctx().get('fault_adapter'):return ctx()['fault_adapter'](seg,center,t,stream)
    spec=ctx()['stream_specs'][stream]
    if spec['kind']=='clean':return seg
    return NATIVE_PROTOCOL_ATTACK(seg,center,t,spec['kind'],spec['alpha'],spec['location'],9417)[0]

def initial(center,seed):
    path=ROOT/'initial'/f'{center}_{seed}.pt';side=path.with_suffix('.json')
    assert path.exists() and side.exists(),'M0 must be prepared, metered and locked before trajectories'
    meta=read(side);assert sha(path)==meta['sha256']
    m=Forecast().to(DEVICE);m.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True))
    assert state_hash(m)==meta['state_hash'];return m

class OriginalCMCGate(NativeGate):
    def __init__(self,*args):
        super().__init__(PRIOR/'runtime',PRIOR/'protocol/frozen_v2_final_commit_config.yaml')
        import core.v2_engine as v2
        v2.windows=windows
    def evaluate(self,candidate,parent,recent_blocks,prior_history_blocks,horizon=None):
        return super().evaluate(candidate,parent,recent_blocks,prior_history_blocks,horizon=int(ctx().get('horizon',1)) if horizon is None else horizon)

class PortTransform(ast.NodeTransformer):
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='input_cache' for t in node.targets) and isinstance(node.value,ast.Dict):
            node.value=ast.Call(ast.Name('InputCacheView',ast.Load()),[],[])
        return self.generic_visit(node)
    def visit_BinOp(self,node):
        text=ast.unparse(node)
        if text in ('seed + center + t * 100',"ctx['seed'] + ctx['center'] + t * 100"):
            a=ast.Name('seed',ast.Load()) if text.startswith('seed') else ast.Subscript(ast.Name('ctx',ast.Load()),ast.Constant('seed'),ast.Load())
            b=ast.Name('center',ast.Load()) if text.startswith('seed') else ast.Subscript(ast.Name('ctx',ast.Load()),ast.Constant('center'),ast.Load())
            return ast.Call(ast.Name('training_seed',ast.Load()),[a,b,ast.Name('t',ast.Load())],[])
        if text=="s['seed'] + s['center'] + t * 100":return ast.parse("training_seed(s['seed'],s['center'],t)",mode='eval').body
        if text=="s['seed'] + s['center']":return ast.parse("training_seed(s['seed'],s['center'],0)",mode='eval').body
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if type(node.value) is int and node.value==24:return ast.Call(func=ast.Name('stage_limit',ast.Load()),args=[],keywords=[])
        if type(node.value) is int and node.value==25:return ast.BinOp(ast.Call(ast.Name('stage_limit',ast.Load()),[],[]),ast.Add(),ast.Constant(1))
        if node.value=='SIMULATED_SOURCE_ATTESTED_NOT_FIELD':return ast.Call(func=ast.Name('source_claim',ast.Load()),args=[],keywords=[])
        if node.value=='SIMULATED_CAUSAL_LABEL_PROVIDER_ONLY':return ast.Call(func=ast.Name('authority_issuer',ast.Load()),args=[],keywords=[])
        if node.value=='NEW_SYNTHETIC_SMOKE_DEVELOPMENT_NOT_FINAL':return ast.Constant('COVERAGE_FROZEN_GATE_SUPPLEMENT; see domain and physical-cluster manifest')
        return node
    def visit_ImportFrom(self,node):
        if node.module=='d3_runner':return None
        if node.module=='f2_gate':
            return None # exact frozen joint_gate is loaded below in this namespace
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='NATIVE_RUN':
            for kw in node.keywords:
                if kw.arg=='backbone':kw.value=ast.Call(ast.Name('current_architecture',ast.Load()),[],[])
                if kw.arg=='H':kw.value=ast.Call(ast.Name('horizon',ast.Load()),[],[])
        return node
    def visit_keyword(self,node):
        if node.arg=='regime_audit_only':
            node.value=ast.Call(ast.Name('regime_for',ast.Load()),[ast.Name('center',ast.Load())],[])
            # persist(ctx,row) has no local center variable.
            if isinstance(getattr(node,'value',None),ast.Call):pass
            return node
        return self.generic_visit(node)

derive('d3_engine.py',['CAS','CausalProvider','SimulatedLabelAuthority','score'],PortTransform())
from model_store import ModelStore
CAS=ModelStore
BaseAuthority=SimulatedLabelAuthority
class SimulatedLabelAuthority(BaseAuthority):
    @property
    def ISSUER(self):return authority_issuer()
    def late_score(self,stage,snap,mat,proof,service,parent_entry,candidate_entry):
        # Byte-identical scalar/proof rule, explicit horizon-aware windows.
        assert self.ledger.event(mat['seq'])['kind']=='LABELS_MATURED'
        start=time.perf_counter();parent=self.cas.load(parent_entry);candidate=self.cas.load(candidate_entry);xy=tensors(windows(service))
        delta=float(errors(candidate,xy)[0].mean())-float(errors(parent,xy)[0].mean());sync();cost=time.perf_counter()-start
        s=snap['data'];fields=dict(data_role='HISTORICAL_MATURED_SERVICE_AUDIT',mature_seq=mat['seq'],receipt_sha256=digest_bytes(canonical(proof)),parent_sha256=s['parent_sha256'],candidate_sha256=s['candidate_sha256'],delta_late=delta,proxy_old=s['old_score'],proxy_new=s['new_score'],proxy_recent=s['recent_score'],scale=s['scale'],late_forward_models=2,late_load_and_forward_seconds=cost)
        ev=self.ledger.append('LATE_PAIR_SCORED',self.scope,stage,fields)
        return LateAudit(*self.scope,stage,ev['seq'],mat['seq'],fields['receipt_sha256'],s['parent_sha256'],s['candidate_sha256'],delta,s['old_score'],s['new_score'],s['recent_score'],s['scale']),cost
from helpers.label_weight_reference import label_weights_from_update_raw,target_weights_for_windows
derive('d3_adapt.py',['batch_hash','target_weights','weighted_adapter','make_candidate'])
original_target_weights=target_weights
def target_weights(update):
    if ctx().get('domain')=='thermbuild':
        # U_MSE ignores these legacy label weights. Never impute missing targets.
        return np.ones(len(windows(update)[0])),np.ones(len(update.y))
    return original_target_weights(update)
original_candidate=make_candidate
def make_candidate(*args,**kwargs):
    model,meta=original_candidate(*args,**kwargs);meta['horizon']=horizon();return model,meta
derive('f2_gate.py',['joint_gate'])
from math import isfinite
derive('d3_trajectory.py',['late_from_event','run_trajectory'],PortTransform())
from audit_input_cache import InputCacheView

def arms(phase):return ctx()['arms']
VARIANTS=()
derive('d3_audit.py',['audit_phase'],PortTransform())

# All six legacy baselines retain the exact original native.run callable.
LEGACY_NAMES=dict(zip(('ALWAYS_FT','ER','EWC','HUBER_FT','VALIDATION_COMMIT','FROZEN'),('Always-FT','ER','EWC','Huber-FT','ValidationCommit','Frozen')))
NATIVE_PROTOCOL_ATTACK=native_attack
class LegacyTransform(PortTransform):
    def visit_keyword(self,node):
        if node.arg=='regime_audit_only':
            node.value=ast.Call(ast.Name('regime_for',ast.Load()),[ast.Subscript(ast.Name('ctx',ast.Load()),ast.Constant('center'),ast.Load())],[])
            return node
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=super().visit_Call(node)
        # Native engine receives the frozen scenario's actual kind/alpha/location.
        if isinstance(node.func,ast.Name) and node.func.id=='NATIVE_RUN':
            for kw in node.keywords:
                if kw.arg in ('kind','alpha','location'):
                    kw.value=ast.Call(ast.Name('scenario_value',ast.Load()),[ast.Constant(kw.arg)],[])
        return node

derive('legacy_adapter.py',['finite','fisher','persist','NativeTimeline','run_legacy','audit_legacy'],LegacyTransform())
def current_architecture():return ctx().get('architecture','GRU')
def scenario_value(name):return ctx()['stream_specs'][TLS.ctx['stream']][name]
def native_initial(center,seed,backbone,H,smoke=False):
    assert backbone==current_architecture() and H==horizon();return initial(center,seed)
def native_attack(seg,center,t,kind,alpha,location,seed=9417):
    spec=ctx()['stream_specs'][TLS.ctx['stream']]
    assert (kind,alpha,location)==(spec['kind'],spec['alpha'],spec['location']) and seed==9417
    if ctx().get('fault_adapter'):
        out=observed(seg,center,t,TLS.ctx['stream'])
        return out,ctx()['fault_adapter'].last_mask.copy(),np.full(len(seg.y),-1,dtype='int64'),kind
    return NATIVE_PROTOCOL_ATTACK(seg,center,t,kind,alpha,location,seed)
native.fisher_diag=fisher;native.initial_model=native_initial;native.attack=native_attack;native.Timeline=NativeTimeline
native.windows=windows;native.evidence_blocks=evidence_blocks
NATIVE_ADAPT=native.adapt
def native_adapt(parent,xy,*args,**kwargs):
    if ctx().get('domain') in ('uci','thermbuild'):
        kwargs['seed']=training_seed(TLS.ctx['seed'],TLS.ctx['center'],TLS.science_stage)
    return NATIVE_ADAPT(parent,xy,*args,**kwargs)
native.adapt=native_adapt
assert NATIVE_RUN.__code__.co_filename==str(PRIOR/'runtime/core/engine.py')

def audit_trajectory(phase,td):
    if Path(td).name in LEGACY_NAMES:return audit_legacy(td)
    return audit_phase(phase,stage_limit(),True,only_td=td)[1]

def export_derivation():
    return dict(source_function_sha256=ORIGINAL_FUNCTIONS,native_engine_sha256=sha_meta(PRIOR/'runtime/core/engine.py'),
        transformations=['explicit registered data root','thread-local architecture/horizon/window interface','stage count from authenticated role schedule',
            'original frozen arms from passed Main','external provenance claim and regime reporting','all legacy native.run byte-identical'],
        algorithm_search=False,source_files_modified=False)
