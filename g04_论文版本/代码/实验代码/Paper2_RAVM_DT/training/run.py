"""Relocate frozen code/data without changing training or gate mathematics.

Default preflight is read only with every torch optimizer step prohibited.
The explicit `train` subcommand is for a reader's future, separate reproduction.
Bounded fixed-case lifecycle acceptance was executed; full matrix/upstream replication was not.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,json,os,shutil,sys,types,tempfile
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True),'utf-8')
 # Windows readers/antivirus may briefly deny replacement; never retrain for this.
 for attempt in range(100):
  try:os.replace(tmp,p);break
  except PermissionError:
   if attempt==99:raise
   __import__('time').sleep(.02)
def verify():
 manifest=read(B/'ASSET_MANIFEST.json')
 for row in manifest['files']:
  p=B/row['file'];assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['file']
 return manifest
def bootstrap(out):
 out=out.resolve();out.mkdir(parents=True,exist_ok=True)
 meta=out/'_control';meta.mkdir(exist_ok=True);shutil.copy2(B/'identities/SCIENCE_EXECUTION.lock.json',meta/'SCIENCE_EXECUTION.lock.json')
 common=types.ModuleType('common');common.ROOT=meta;common.PRIOR=B/'code/main';common.STORAGE=out;common.read=read;common.sha=sha;common.atomic=atomic
 sys.modules['common']=common
 sys.path[:0]=[str(B/'code/meta'),str(B/'code/main'),str(B/'code/main/runtime')]
 import bridge as b,model_store
 model_store.BLOB_ROOTS=[(out/'CAS',8*2**30)]
 return b
def stage(spec,out):
 spec=copy.deepcopy(spec);root=out/'groups'/spec['group'];root.mkdir(parents=True,exist_ok=True)
 # Resolve stored package-relative source identities only at runtime.
 for row in spec.get('source_data',[]):
  if row.get('source_path') and not Path(row['source_path']).is_absolute():row['source_path']=str((B/row['source_path']).resolve())
 for row in spec.get('existing_m0',[]) if isinstance(spec.get('existing_m0'),list) else []:
  if row.get('path') and not Path(row['path']).is_absolute():row['path']=str((B/row['path']).resolve())
 for row in spec.get('data_files',[]):
  if row.get('path') and not Path(row['path']).is_absolute():row['path']=str((B/row['path']).resolve())
 if isinstance(spec.get('reuse_main_initial'),str) and not Path(spec['reuse_main_initial']).is_absolute():spec['reuse_main_initial']=str((B/spec['reuse_main_initial']).resolve())
 for rel,asset in spec['input_mapping'].items():
  dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert sha(dst)==sha(B/asset)
  else:shutil.copy2(B/asset,dst)
 for rel in ['reference/LEGACY_BASELINE_IMPLEMENTATION_IDENTITY.json','runtime/core/engine.py']:
  dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(B/'code/main'/rel,dst)
 spec['root']=str(root)
 for source in spec['sources'].values():source['path']=str(B/source['path'])
 return spec
def selected_functions(path,names,env):
 source=path.read_text('utf-8');tree=ast.parse(source)
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
 return env
def attach(spec,b):
 import numpy as np
 from immutable_cache import arrays
 if spec['group']=='Main':
  # Same frozen center ordering includes exposed Smoke followed by NewFinal.
  manifest=read(B/'identities/RUN_MANIFEST.json');ids=[c for v in manifest['centers'].values() for c in v]
  env=dict(np=np,hashlib=hashlib,CENTERS=ids,ATTACK_SEED=9417,ALPHA=.1,STICKY_MAGNITUDE=.2,STICKY_ACTIVE_STAGES=(1,24))
  selected_functions(B/'code/main/runtime/b1_data.py',['sticky_spec','coherent_sticky_attack'],env)
  def fault(seg,c,t,stream):
   mask=np.zeros(len(seg.y),bool)
   if stream=='clean':result=seg
   elif stream=='mixed':result,mask,*_=b.NATIVE_PROTOCOL_ATTACK(seg,c,t,'mixed',.1,'both',9417)
   else:result,mask,*_=env['coherent_sticky_attack'](seg,c,t)
   fault.last_mask=mask;return result
  return dict(spec,fault_adapter=fault)
 if spec['domain'] not in ('uci','thermbuild'):return spec
 # Read only exact fault functions, without executing the raw-data loader's globals.
 env=dict(np=np,Segment=b.Segment)
 if spec['domain']=='uci':
  std=np.ones(20);std[spec['room_index']*2]=spec['target_scale']
  env.update(ROOMS=['T1','T2','T3'],STD=std,X=np.empty((0,40)))
  selected_functions(B/'code/recipes/uci.py',['attack'],env)
  recipe=types.SimpleNamespace(**env)
 else:
  env['INPUT_NAMES']=tuple(range(11))
  tree=ast.parse((B/'code/recipes/thermbuild.py').read_text('utf-8'))
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ThermalProvider')
  method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_fault')
  exec(compile(ast.Module(body=[method],type_ignores=[]),str(B/'code/recipes/thermbuild.py'),'exec'),env)
  recipe=types.SimpleNamespace(ThermalProvider=type('ThermalProvider',(),{'_fault':env['_fault']}))
 funcs=dict(Path=Path,original=lambda domain:recipe)
 selected_functions(B/'code/meta/e4_assets.py',['attach'],funcs)
 return funcs['attach'](spec)
def preflight(out):
 import torch
 for cls in [torch.optim.SGD,torch.optim.Adam,torch.optim.AdamW,torch.optim.RMSprop,torch.optim.LBFGS]:
  cls.step=lambda *a,**k:(_ for _ in ()).throw(PermissionError('Preflight prohibits every optimizer step'))
 b=bootstrap(out);groups=[];weights=0
 for path in sorted((B/'groups').glob('*.json')):
  spec=attach(stage(read(path),out),b);shapes=[];role_count=0;fault_count=0
  with b.context(spec):
   # All provided source and initial models are loaded and state hashes checked.
   for source in spec['sources'].values():
    model=b.Forecast().to(b.DEVICE);model.load_state_dict(torch.load(source['path'],map_location=b.DEVICE,weights_only=True));assert b.state_hash(model)==source['state_hash'];weights+=1
   for c in spec['centers']:
    for seed in spec['seeds']:
     model=b.initial(c,seed);weights+=1
    timeline=b.Timeline(c)
    for t in range(1,spec['stages']+1):
     update,ver=timeline.maintenance(t);role_count+=2
     assert update.role=='update' and ver.role=='verification'
     assert len(b.windows(update)[0]) and len(b.windows(ver)[0])
     if c==spec['centers'][0] and t in (1,spec['stages']):
      for stream in spec['stream_ids']:
       u=b.observed(update,c,t,stream);v=b.observed(ver,c,t,stream)
       assert u.role=='update' and v.role=='verification';fault_count+=2
     timeline.record_decision(t,False)
    shapes.append([len(timeline.initial_history().y),int(model.H)])
   with torch.no_grad():assert list(model(torch.zeros(2,96,spec['features'],device=b.DEVICE)).shape)==[2,spec['horizon']]
   # Import original gate and evaluate identical models on existing Verification.
   gate=b.OriginalCMCGate();u,v=b.Timeline(spec['centers'][0]).maintenance(1)
   history=b.evidence_blocks(b.Timeline(spec['centers'][0]).initial_history())
   score=b.score(model,model,b.evidence_blocks(v),history,v,gate)
   assert all(abs(x)<1e-12 for x in score['recent_blocks']),score
  # Enter the actual lifecycle runner, stopping BEFORE any candidate adaptation.
  # Temporary preflight journals are not scientific results or new trajectories.
  class CandidateBoundary(Exception):pass
  def boundary(*a,**k):raise CandidateBoundary('Reached real candidate-training boundary; no optimizer executed')
  checked=[]
  with b.context(spec):
   for legacy in (False,True):
    arms=[a for a in spec['arms']if (a['id'] in b.LEGACY_NAMES)==legacy and a['id']!='FROZEN']
    if not arms:continue
    arm=arms[0];prior=b.make_candidate;native_prior=b.native.adapt
    try:
     b.make_candidate=boundary;b.native.adapt=boundary
     fn=b.run_legacy if legacy else b.run_trajectory
     try:fn(spec['phase'],spec['centers'][0],spec['seeds'][0],spec['stream_ids'][0],arm,'PREFLIGHT_NO_SCIENTIFIC_RUN')
     except CandidateBoundary:checked.append(arm['id'])
     else:raise AssertionError('Runner did not stop at candidate boundary')
    finally:b.make_candidate=prior;b.native.adapt=native_prior
  groups.append(dict(group=spec['group'],source_and_initial_weights='HASH_AND_STATE_CHECKED',role_segments=role_count,fault_calls=fault_count,fault_sampling='First center, first/last stage, every stream and both maintenance roles',forward_shape=[2,spec['horizon']],actual_runner_reached_training_boundary=checked))
  print(spec['group'],'preflight complete',flush=True)
 receipt=dict(status='PASS_INPUTS_IMPORTS_ROLES_FAULTS_AND_FORWARD_ONLY',groups=groups,weight_loads=weights,optimizer_steps=0,
  full_training_executed=False,training_equivalence_claim=False,full_lifecycle_launcher_test='NOT_RUN_NO_NEW_TRAINING_AUTHORIZED',
  boundary='Supplied frozen source/M0 and preprocessed Final inputs. Upstream source fitting and development selection are documented, not independently rerun.',derivation=b.export_derivation())
 atomic(out/'PREFLIGHT_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))
def train(args,out):
 if out.exists() and any(out.iterdir()):raise RuntimeError('Choose a new empty output directory; no implicit resume or overwrite')
 spec=read(B/'groups'/f'{args.group}.json')
 assert args.center in spec['centers'] and args.seed in spec['seeds'] and args.stream in spec['stream_ids']
 arm=next(a for a in spec['arms']if a['id']==args.arm)
 b=bootstrap(out);spec=attach(stage(spec,out),b)
 with b.context(spec):
  fn=b.run_legacy if arm['id'] in b.LEGACY_NAMES else b.run_trajectory
  fn(spec['phase'],args.center,args.seed,args.stream,arm,'EXTERNAL_REPRODUCTION_NOT_NEW_PAPER_EVIDENCE')
  td=Path(spec['root'])/'science'/spec['phase']/str(args.center)/str(args.seed)/args.stream/arm['id']
  verdict=b.audit_trajectory(spec['phase'],td)
  atomic(out/'TRAINING_RUN_RECEIPT.json',dict(group=args.group,center=args.center,seed=args.seed,stream=args.stream,arm=args.arm,audit=verdict,scope='Reader rerun; never merged into frozen paper results'))
if __name__=='__main__':
 parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='mode',required=True)
 sub.add_parser('verify');p=sub.add_parser('preflight');p.add_argument('--output',type=Path,required=True)
 p=sub.add_parser('train');p.add_argument('--output',type=Path,required=True)
 for key in ['group','center','seed','stream','arm']:p.add_argument('--'+key,required=True,type=int if key in ('center','seed')else str)
 args=parser.parse_args();manifest=verify()
 if args.mode=='verify':print(json.dumps(dict(status='PASS_BYTES_ONLY',files=len(manifest['files']),bytes=manifest['total_bytes'])))
 else:
  out=args.output.resolve()
  if out==B or B in out.parents:raise RuntimeError('Use a separate output directory outside immutable bundle')
  preflight(out)if args.mode=='preflight'else train(args,out)
