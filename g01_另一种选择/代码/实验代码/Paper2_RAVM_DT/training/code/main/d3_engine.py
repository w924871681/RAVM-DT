"""Real frozen GRU/50-step adapter, new development data and causal label scheduler.
Trusted simulation/provider module. Gate module has no provider capability.
"""
from d3_common import *
import io,copy,ast,gc,math
import numpy as np
np.polynomial.hermite.hermgauss(21)
import torch
sys.path.insert(0,str(ROOT/'runtime'))
from core.models import Forecast,DEVICE,adapt,errors,state_hash,tensors,sync,Batch
from core.protocol import Timeline,Segment,windows,spans,digest,attack
from core.engine import evidence_blocks
from cmc_original_gate_adapter import OriginalCMCGate
from stage3_engine import predictions_losses,score_arrays
from vendor.main_cfg import MainCfg
from vendor.sim import simulate_centers
from d2c_core import AuditLedger,FeedbackVerifier,LateAudit,digest as digest_bytes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

REGIMES=('stationary','gradual','abrupt','recurring')
STREAMS=('clean','mixed','coherent_sticky')
SEEDS=(7311,7312,7313)
SOURCE=read(ROOT/'ASSET_IDENTITY.json')['sources']
IDS=read(ROOT/'RUN_MANIFEST.json')['centers']
ALL_IDS=tuple(c for v in IDS.values() for c in v)
_tree=ast.parse((ROOT/'runtime/b1_data.py').read_text(encoding='utf8'))
_env=dict(np=np,hashlib=hashlib,CENTERS=ALL_IDS,ATTACK_SEED=9417,ALPHA=.1,STICKY_MAGNITUDE=.2,STICKY_ACTIVE_STAGES=(1,24))
exec(compile(ast.Module(body=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in ('sticky_spec','coherent_sticky_attack')],type_ignores=[]),'authenticated_b1_sticky','exec'),_env)
def observed(seg,center,t,stream):
 if stream not in STREAMS:raise ValueError('Unregistered stream')
 if stream=='clean':return seg
 if stream=='mixed':return attack(seg,center,t,'mixed',.1,'both',9417)[0]
 return _env['coherent_sticky_attack'](seg,center,t)[0]
def _logit(a):
 a=np.clip(a,1e-5,1-1e-5);return np.log(a/(1-a))
def _sigmoid(a):return 1/(1+np.exp(-a))
def generate(phase):
 manifest=ROOT/'data'/f'{phase}_IDENTITY.json'
 if manifest.exists():
  for r in read(manifest)['rows']:assert sha(ROOT/r['path'])==r['sha256']
  return
 from f2_plan import verify_phase_permission
 permission=verify_phase_permission(phase)
 role=AuditLedger(ROOT/'process/DATA_ROLE_EVENTS.jsonl')
 begin=role.append('INPUT_GENERATION_STARTED',('D3D1_ROLE_CONTROL','ALL','ALL'),1,dict(phase=phase,permission_sha256=permission,frozen_execution_sha256=execution_lock_sha(),selection_sha256=sha(ROOT/'DEVCAL_SELECTED_LOCK.json') if phase in ('Smoke','NewFinal') else None))
 centers=IDS[phase];cfg=MainCfg();cfg.sim.seed=7311;cfg.sim.n_history=17000
 for kind in 'ABC':getattr(cfg.sim,'type_'+kind).drift_prob=0.
 raw=simulate_centers(cfg,len(centers),seed_offset=centers[0]*17,start_center_id=centers[0]);rows=[]
 for index,(center,source) in enumerate(zip(centers,raw)):
  path=ROOT/'data'/f'center_{center}.npz';path.parent.mkdir(exist_ok=True)
  regime=REGIMES[index%4];t=np.arange(len(source.y));shift=np.zeros(len(t));onset=1400+8*640
  if regime=='abrupt':shift[t>=onset]=.8
  elif regime=='gradual':shift=.8*np.clip((t-onset)/(8*640),0,1)
  elif regime=='recurring':shift=.8*(((t>=1400+6*640)&(t<1400+12*640))|(t>=1400+18*640))
  y=_sigmoid(_logit(source.y)+shift).astype('float32');x=np.zeros_like(source.X)
  for k,gain in enumerate(source.metadata['metric_gains']):
   delay=source.metadata['metric_delays'][k]//source.dt
   dshift=np.concatenate([np.repeat(shift[0],delay),shift[:len(shift)-delay]]) if delay else shift
   x[:,k]=(_sigmoid(_logit(source.X[:,k])+gain*dshift)*source.M[:,k]).astype('float32')
  x=np.concatenate([x,source.M],1)
  meta=dict(center_id=center,regime_audit_only=regime,pool='D3f2R_'+phase+'_NEW_PROSPECTIVE_POOL',dt=source.dt,data_seed=7311,seed_offset=centers[0]*17,base_generator=source.metadata)
  if path.exists():
   with np.load(path) as z:assert digest(z['x'],z['y'],z['ts'])==digest(x,y,source.timestamps),'Interrupted generation identity mismatch'
  else:
   tmp=path.with_suffix('.tmp')
   with tmp.open('xb') as f:np.savez_compressed(f,x=x,y=y,ts=source.timestamps,meta=json.dumps(meta));f.flush();os.fsync(f.fileno())
   os.replace(tmp,path)
  rows.append(dict(center=center,path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path),regime_audit_only=regime))
 end=role.append('INPUT_GENERATION_COMPLETED',('D3D1_ROLE_CONTROL','ALL','ALL'),1,dict(phase=phase,begin_seq=begin['seq'],selection_sha256=sha(ROOT/'DEVCAL_SELECTED_LOCK.json') if phase in ('Smoke','NewFinal') else None,input_rows_sha256=hashlib.sha256(canonical(rows)).hexdigest()))
 atomic(manifest,dict(phase=phase,rows=rows,generation_start_seq=begin['seq'],generation_end_seq=end['seq'],permission_sha256=permission,devcal_lock_sha256=sha(ROOT/'DEVCAL_SELECTED_LOCK.json') if phase in ('Smoke','NewFinal') else None,created_unix=time.time(),execution_lock_sha256=execution_lock_sha()),exclusive=True)

class CAS:
 def __init__(self,path):self.root=Path(path);self.root.mkdir(parents=True,exist_ok=True)
 def put(self,model):
  buf=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},buf);b=buf.getvalue();h=hashlib.sha256(b).hexdigest();path=self.root/h
  if not path.exists():
   with path.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  assert sha(path)==h
  return dict(file_sha256=h,state_hash=state_hash(model),bytes=len(b))
 def verify(self,h):return isinstance(h,str) and len(h)==64 and (self.root/h).is_file() and sha(self.root/h)==h
 def load(self,entry):
  assert self.verify(entry['file_sha256']) and (self.root/entry['file_sha256']).stat().st_size==entry['bytes']
  model=Forecast().to(DEVICE);model.load_state_dict(torch.load(self.root/entry['file_sha256'],map_location=DEVICE,weights_only=True));assert state_hash(model)==entry['state_hash'];return model

def initial(center,seed):
 path=ROOT/'initial'/f'{center}_{seed}.pt';side=path.with_suffix('.json');path.parent.mkdir(exist_ok=True)
 if path.exists() and side.exists():
  meta=read(side);assert sha(path)==meta['sha256'];m=Forecast().to(DEVICE);m.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True));assert state_hash(m)==meta['state_hash'];return m
 if path.exists() or side.exists():raise RuntimeError('Partial M0 creation; original source not replaced')
 src=ROOT/SOURCE[str(seed)]['path'];assert sha(src)==SOURCE[str(seed)]['sha256']
 m=Forecast().to(DEVICE);m.load_state_dict(torch.load(src,map_location=DEVICE,weights_only=True));assert state_hash(m)==SOURCE[str(seed)]['state_hash']
 timeline=Timeline(center);start=time.perf_counter();m=adapt(m,tensors(windows(timeline.warmup())),steps=50,seed=seed+center);sync();elapsed=time.perf_counter()-start
 tmp=path.with_suffix('.tmp');torch.save(m.state_dict(),tmp);os.replace(tmp,path)
 atomic(side,dict(sha256=sha(path),state_hash=state_hash(m),source_sha256=sha(src),warmup_hash=timeline.warmup().hash,warmup_steps=50,warmup_seed=seed+center,common_initialization_seconds=elapsed,maintenance_cost_excludes_shared_initialization=True),exclusive=True)
 return m

class CausalProvider:
 """Service capability belongs to the trusted scheduler/label authority only."""
 def __init__(self,center,stream):
  if center not in tuple(c for v in IDS.values() for c in v) or stream not in STREAMS:raise ValueError('Unregistered data identity')
  self.center=center;self.stream=stream;self.__timeline=Timeline(center)
 def initial_history(self):return self.__timeline.initial_history()
 def maintenance(self,t):
  u,v=self.__timeline.maintenance(t)
  return observed(u,self.center,t,self.stream),observed(v,self.center,t,self.stream)
 def restore_prefix(self,completed):
  for t in range(1,completed+1):self.__timeline.record_decision(t,False)
 def service(self,t,sealed,ledger):
  if sealed is None:raise PermissionError('No verified seal; no Service')
  ev=ledger.event(sealed['event_seq'])
  if ev['kind']!='DECISION_SEALED' or ev['stage']!=t or ev['data']['decision_sha256']!=sealed['sha256']:raise PermissionError('Seal identity mismatch')
  self.__timeline.record_decision(t,sealed['commit'])
  return self.__timeline.service(t)

class SimulatedLabelAuthority:
 ISSUER='SIMULATED_CAUSAL_LABEL_PROVIDER_ONLY'
 def __init__(self,td,ledger,cas,provider,scope):
  self.ledger,self.cas,self.__provider,self.scope=ledger,cas,provider,scope
  key=Path(td)/'PRIVATE_SIMULATION_LABEL_KEY.bin'
  if key.exists():self.__key=Ed25519PrivateKey.from_private_bytes(key.read_bytes())
  else:
   self.__key=Ed25519PrivateKey.generate()
   with key.open('xb') as f:f.write(self.__key.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption()));f.flush();os.fsync(f.fileno())
  self.public_key=self.__key.public_key()
  self.public_hex=self.public_key.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()
 def mature(self,stage,snap,sealed):
  # This simulator declares complete source-stage labels ready after its Service is generated.
  # Not a real-world/third-party attestation. No old receipt is retroactively minted.
  service=self.__provider.service(stage,sealed,self.ledger)
  opens=[e for e in self.ledger.entries if e['kind']=='SERVICE_ROLE_OPENED' and e['stage']==stage]
  opened=opens[0] if opens else self.ledger.append('SERVICE_ROLE_OPENED',self.scope,stage,dict(seal_seq=sealed['event_seq'],service_sha256=service.hash,max_target_ts=int(service.ts[-1]),data_role='SERVICE'))
  assert opened['data']['service_sha256']==service.hash and opened['data']['seal_seq']==sealed['event_seq']
  mats=[e for e in self.ledger.entries if e['kind']=='LABELS_MATURED' and e['stage']==stage]
  if mats:
   mat=mats[0];proof=mat['data']['receipt'];self.public_key.verify(bytes.fromhex(mat['data']['signature']),canonical(proof));assert proof['service_sha256']==service.hash
   self.ledger.append('RECOVERY_READ_SEALED_MATURED_SERVICE',self.scope,stage,dict(seal_seq=sealed['event_seq'],mature_seq=mat['seq']))
   return service,mat,proof
  proof=dict(issuer=self.ISSUER,run=self.scope[0],center=self.scope[1],stream=self.scope[2],stage=stage,snapshot_seq=snap['seq'],seal_seq=sealed['event_seq'],service_open_seq=opened['seq'],service_sha256=service.hash,max_target_ts=int(service.ts[-1]),label_watermark=int(service.ts[-1]),mature_seq=self.ledger.next_seq,source_claim='SIMULATED_SOURCE_ATTESTED_NOT_FIELD')
  mat=self.ledger.append('LABELS_MATURED',self.scope,stage,dict(receipt=proof,signature=self.__key.sign(canonical(proof)).hex()))
  return service,mat,proof
 def late_score(self,stage,snap,mat,proof,service,parent_entry,candidate_entry):
  assert self.ledger.event(mat['seq'])['kind']=='LABELS_MATURED'
  start=time.perf_counter();parent=self.cas.load(parent_entry);candidate=self.cas.load(candidate_entry);xy=tensors(windows(service))
  delta=float(errors(candidate,xy)[0].mean())-float(errors(parent,xy)[0].mean());sync();cost=time.perf_counter()-start
  s=snap['data'];fields=dict(data_role='HISTORICAL_MATURED_SERVICE_AUDIT',mature_seq=mat['seq'],receipt_sha256=digest_bytes(canonical(proof)),parent_sha256=s['parent_sha256'],candidate_sha256=s['candidate_sha256'],delta_late=delta,proxy_old=s['old_score'],proxy_new=s['new_score'],proxy_recent=s['recent_score'],scale=s['scale'],late_forward_models=2,late_load_and_forward_seconds=cost)
  ev=self.ledger.append('LATE_PAIR_SCORED',self.scope,stage,fields)
  return LateAudit(*self.scope,stage,ev['seq'],mat['seq'],fields['receipt_sha256'],s['parent_sha256'],s['candidate_sha256'],delta,s['old_score'],s['new_score'],s['recent_score'],s['scale']),cost

def score(candidate,parent,recent,history,ver,gate):
 trace=[];original=gate._paired_score
 def capture(*args,**kwargs):
  result=original(*args,**kwargs);trace.append(result);return result
 gate._paired_score=capture
 try:anchor=gate.evaluate(candidate,parent,recent,history)
 finally:gate._paired_score=original
 rec,hist=trace;hv=hist[3];old=hv[:6] if len(hv)==12 else np.array([]);new=hv[6:] if len(hv)==12 else np.array([])
 def robust(v):return float(np.median(v)+.5*np.median(abs(v-np.median(v)))) if len(v) else None
 pairs=[tensors(windows(b)) for b in list(history[-12:])+recent]
 xy=Batch(torch.cat([p[0] for p in pairs]),torch.cat([p[1] for p in pairs]),'verification')
 pl=predictions_losses(parent,xy)[1];cl=predictions_losses(candidate,xy)[1];ps,detail=score_arrays(pl,cl,[len(p[0]) for p in pairs],6)
 scale=max(float(errors(parent,tensors(windows(ver)))[0].mean()),1e-12)
 return dict(parent_recent_verification_mean_loss=float(pl[detail['recent_first_window']:].mean()),original_commit=anchor.commit,recent_score=anchor.recent_score,history_score=anchor.history_score,recent_median=anchor.recent_median,recent_mad=anchor.recent_mad,history_median=anchor.history_median,history_mad=anchor.history_mad,recent_blocks=rec[3].tolist(),history_blocks=hv.tolist(),old_score=robust(old),new_score=robust(new),old_median=float(np.median(old)) if len(old) else None,new_median=float(np.median(new)) if len(new) else None,scale=scale,stage3=ps,history_block_hashes=[b.hash for b in history[-12:]],recent_block_hashes=[b.hash for b in recent])
