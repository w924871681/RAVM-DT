"""Authenticate old assets, reserve new identities and prepare an isolated executor. No science imports."""
from pathlib import Path
import json,hashlib,shutil,subprocess,re,time,difflib
BASE=Path(__file__).resolve().parent;ROOT=BASE/'D3f2_ISOLATED_WORK';PKG=BASE/'execution_package'
OLD=BASE.parent/'ravm_d3d1_20261020/D3d1_ISOLATED_WORK'
F1=BASE.parent/'ravm_d3f1_20261020/D3f1_ISOLATED_WORK'
CENTERS=dict(P0=list(range(60000,60004)),DevCal=list(range(60004,60012)),Confirm=list(range(60012,60020)),ExpandedDev=list(range(60020,60032)))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def ids(x):
 if isinstance(x,int):return {x}
 if isinstance(x,(list,tuple)):return set().union(*(ids(v) for v in x)) if x else set()
 if isinstance(x,dict):return set().union(*(ids(v) for v in x.values())) if x else set()
 return set()
def main():
 assert not ROOT.exists(),'Never overwrite an existing scientific attempt'
 for n,r in read(PKG/'FILE_SHA256.json').items():assert sha(PKG/n)==r['sha256'] and (PKG/n).stat().st_size==r['bytes'],n
 assert sha(PKG/'D3f1_INFO_REPORT_ZH.md')==sha(F1/'D3f1_INFO_REPORT_ZH.md')
 for n,r in read(F1/'D3F1_EXECUTION.lock.json')['files_sha256'].items():assert sha(F1/n)==r,n
 a=read(F1/'results/INDEPENDENT_AUDIT.json');assert a['status']=='VALIDATED' and a['verified_opportunities']==1152
 assert read(F1/'results/INFO_DECISION.json')['decision']=='INFO_GO'
 p=read(F1/'delivery/PACKAGE_VERIFICATION.json');assert p['status']=='PASS' and sha(F1/'delivery/D3f1_INFO_RETURN.zip')==p['sha256']
 inventory={};meta=[];path_count=0;fresh=set(sum(CENTERS.values(),[]));used=set(range(47000,47016))|set(range(59000,59032));provenance={}
 for scanroot in [Path(__file__).resolve().parents[2],Path(__file__).resolve().parents[2]]:
  if not scanroot.exists():continue
  proc=subprocess.run(['rg','--files','--hidden','-g','!**/.git/**','-g','!**/node_modules/**','-g','!**/_parquet_deps/**',str(scanroot)],capture_output=True,text=True,encoding='utf8');assert proc.returncode==0
  for wire in proc.stdout.splitlines():
   q=Path(wire)
   if q.is_relative_to(BASE):continue
   path_count+=1
   for c in fresh:
    if str(c) in q.parts or re.search(r'(?:center_|M0_[A-Za-z0-9_]*_|B1_M0_[A-Za-z0-9_]*_)'+str(c)+r'(?:_|\.|$)',q.name):raise RuntimeError('STOP_ASSET_CONFLICT '+wire)
   n=q.name.lower()
   if n in ('historical_id_registry.json','run_manifest.json','data_role_boundaries.json') or n.endswith('id_registry.json'):meta.append(q)
 for q in meta:
  x=read(q);declared=[]
  if 'registry' in q.name.lower():declared=[x[k] for k in ('known_old_ids','new_reserved','centers','pools','reserved','center_ids','used_ids') if k in x]
  else:declared=[v for k,v in x.items() if any(w in k.lower() for w in ('center','pool','reserved','identity')) and k!='identity_manifests']
  values=ids(declared);used|=values;provenance[str(q)]=dict(sha256=sha(q),center_ids=sorted(values));inventory[str(q)]=dict(sha256=sha(q),bytes=q.stat().st_size)
 assert not fresh&used,'STOP_ASSET_CONFLICT registry identity'
 oldlock=read(OLD/'D3D1_EXECUTION.lock.json')
 for n,h in oldlock['code_sha256'].items():assert sha(OLD/n)==h,n;inventory[str(OLD/n)]=dict(sha256=h,bytes=(OLD/n).stat().st_size)
 asset=read(OLD/'ASSET_IDENTITY.json')
 for seed,r in asset['sources'].items():assert sha(OLD/r['path'])==r['sha256'];inventory[str(OLD/r['path'])]=dict(sha256=r['sha256'],bytes=(OLD/r['path']).stat().st_size)
 for n in ('D3f1_INFO_REPORT_ZH.md','results/INDEPENDENT_AUDIT.json','results/INFO_DECISION.json','D3F1_EXECUTION.lock.json','delivery/PACKAGE_VERIFICATION.json'):
  q=F1/n;inventory[str(q)]=dict(sha256=sha(q),bytes=q.stat().st_size)
 ROOT.mkdir()
 for folder in ('runtime','protocol','helpers'):
  for q in (OLD/folder).rglob('*'):
   if q.is_file() and '__pycache__' not in q.parts:
    dst=ROOT/q.relative_to(OLD);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,dst)
 names=('d3_common.py','d3_engine.py','d3_trajectory.py','d3_gate.py','d3_adapt.py','d2c_core.py','d2c_gate.py','d2b_math_reference.py','d3_parallel.py','d3_results.py','d3_audit.py')
 for n in names:shutil.copy2(OLD/n,ROOT/n)
 for q in PKG.iterdir():shutil.copy2(q,ROOT/q.name)
 for s,r in asset['sources'].items():q=ROOT/r['path'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/r['path'],q)
 for folder in ('data','initial','process','progress','engineering','results','science','reference','delivery'):(ROOT/folder).mkdir(exist_ok=True)
 shutil.copy2(PKG/'D3F2_PROTOCOL_PROPOSED.json',ROOT/'EXPERIMENT_LOCK.json')
 shutil.copy2(PKG/'D3F2_PROTOCOL_PROPOSED.json',ROOT/'D3F2_PROTOCOL_LOCK.json')
 shutil.copy2(BASE.parent/'AGENTS.md',ROOT/'reference/APPLICABLE_AGENTS.md')
 shutil.copy2(BASE.parent/'ravm_d3d1_20261020/build/dynamic_worker_resources_v2.py',ROOT/'dynamic_worker_resources_v2.py')
 for n in ('D3D1_EXECUTION.lock.json','EXPERIMENT_LOCK.json','ASSET_IDENTITY.json'):shutil.copy2(OLD/n,ROOT/'reference'/('D3D1_'+n))
 for n in ('D3f1_INFO_REPORT_ZH.md','results/INDEPENDENT_AUDIT.json','results/INFO_DECISION.json','delivery/PACKAGE_VERIFICATION.json'):
  shutil.copy2(F1/n,ROOT/'reference'/('D3F1_'+Path(n).name))
 save(ROOT/'ASSET_IDENTITY.json',asset)
 save(ROOT/'ORIGINAL_INPUT_IDENTITIES.json',inventory)
 save(ROOT/'GLOBAL_CENTER_PREFLIGHT.json',dict(status='PASS',path_count=path_count,metadata_count=len(meta),metadata=provenance,new_centers=CENTERS,collision=[],scope='Identity metadata and filenames only; no old Final numerical payload',created_unix=time.time()))
 save(ROOT/'HISTORICAL_ID_REGISTRY.json',dict(known_old_ids=sorted(used),new_reserved=CENTERS))
 save(ROOT/'RUN_MANIFEST.json',dict(run_id='D3F2_20261020_ATTEMPT01',centers=CENTERS,seeds=dict(P0=[7311],DevCal=[7311],Confirm=[7311,7312,7313],ExpandedDev=[7311,7312,7313]),streams=['clean','mixed','coherent_sticky'],bootstrap_seed=20261020,bootstrap_draws=4000,startup_ram_min_bytes=2*1024**3))
 # Source modifications are prepared before freezing or generating any new scientific data.
 def patch(n,old,new):
  q=ROOT/n;s=q.read_text(encoding='utf8');assert old in s,(n,old);q.write_text(s.replace(old,new),encoding='utf8')
 patch('d3_common.py',"ROOT/'D3D1_EXECUTION.lock.json'","ROOT/'D3F2_EXECUTION.lock.json'")
 patch('d3_engine.py','from d3_pipeline import verify_phase_permission','from f2_plan import verify_phase_permission')
 patch('d3_engine.py',"pool='D3d1_'+phase+'_SYNTHETIC_DEVELOPMENT'","pool='D3f2_'+phase+'_SYNTHETIC_DEVELOPMENT'")
 patch('d3_engine.py',"frozen_execution_sha256=execution_lock_sha()))","frozen_execution_sha256=execution_lock_sha(),selection_sha256=sha(ROOT/'DEVCAL_SELECTED_LOCK.json') if phase in ('Confirm','ExpandedDev') else None))")
 patch('d3_engine.py',"begin_seq=begin['seq'],input_rows_sha256=", "begin_seq=begin['seq'],selection_sha256=sha(ROOT/'DEVCAL_SELECTED_LOCK.json') if phase in ('Confirm','ExpandedDev') else None,input_rows_sha256=")
 patch('d3_engine.py',"permission_sha256=permission,created_unix=", "permission_sha256=permission,devcal_lock_sha256=sha(ROOT/'DEVCAL_SELECTED_LOCK.json') if phase in ('Confirm','ExpandedDev') else None,created_unix=")
 patch('d3_engine.py',"return dict(original_commit=anchor.commit", "return dict(parent_recent_verification_mean_loss=float(pl[detail['recent_first_window']:].mean()),original_commit=anchor.commit")
 patch('d3_adapt.py',"if update.role!='update' or method not in METHODS:","if update.role!='update' or method!='U_MSE':")
 patch('d3_trajectory.py',"if arm.get('fallback',False):commit=False", "if arm['method']=='JOINT':\n    from f2_gate import joint_gate\n    commit,joint_score=joint_gate(scores['stage3']['recent_mean'],scores['parent_recent_verification_mean_loss'],scores['stage3']['mwu_p'],arm['weight'],arm['threshold']);reason='CURRENT_MEAN_MWU_SOFT_JOINT'\n   elif arm.get('fallback',False):commit=False")
 patch('d3_trajectory.py',"scope='D3D1_EXPOSED_DEVELOPMENT_ONLY'", "scope='D3F2_SYNTHETIC_DEVELOPMENT_NOT_FINAL'") if "scope='D3D1_EXPOSED_DEVELOPMENT_ONLY'" in (ROOT/'d3_trajectory.py').read_text() else None
 patch('d3_trajectory.py',"row['calibration_fallback']=", "row['scope']='D3F2_SYNTHETIC_DEVELOPMENT_NOT_FINAL'\n  row['calibration_fallback']=")
 # Observer-only SGD scope; original adapt and all optimizer arithmetic remain byte-authenticated.
 patch('d3_trajectory.py',"sealed=dict(generator=", "sealed=dict(arm_spec=arm,generator=")
 patch('d3_audit.py','from d3c_gate import decide_variant,VARIANTS,independent_expected','VARIANTS=()')
 patch('d3_audit.py','def audit_phase(phase,expected_rows,verify_service=True):','def audit_phase(phase,expected_rows,verify_service=True,only_td=None):')
 patch('d3_audit.py',"for td in sorted((ROOT/'science'/phase).glob('*/*/*/*')):","for td in ([Path(only_td)] if only_td else sorted((ROOT/'science'/phase).glob('*/*/*/*'))):")
 patch('d3_audit.py',"if phase!='ENGINEERING_TEST':\n  from d3_runner import arms", "if phase!='ENGINEERING_TEST' and only_td is None:\n  from d3_runner import arms")
 patch('d3_audit.py',"[7311] if phase=='DevCal' else SEEDS", "read(ROOT/'RUN_MANIFEST.json')['seeds'][phase]")
 patch('d3_audit.py',"assert s['method']==spec_arm['method'] and s['threshold']==spec_arm['threshold']", "assert s['method']==spec_arm['method'] and s['threshold']==spec_arm['threshold'] and s['arm_spec']==spec_arm\n   if s['method']=='JOINT':\n    z=s['scores'];d=z['stage3']['recent_mean'];p=z['stage3']['mwu_p'];scale=max(z['parent_recent_verification_mean_loss'],1e-12)\n    expected=bool(d<0 and d+spec_arm['weight']*scale*(p-.2)<=-s['threshold'])\n    assert s['commit']==expected and s['reason']=='CURRENT_MEAN_MWU_SOFT_JOINT' and not s['feedback_view']['audits']")
 # Prevent an audit task from accepting an unexpected completed trajectory identity.
 patch('d3_audit.py',"assert len(records)==expected_rows,(len(records),expected_rows)", "assert len(records)==expected_rows,(len(records),expected_rows)\n assert all(r['center'] in IDS[phase] and r['seed'] in read(ROOT/'RUN_MANIFEST.json')['seeds'][phase] and r['stream'] in STREAMS for r in records)")
 # Independent paired intervals and final selection are handled by f2_analysis, not old D3c selection logic.
 patch('d3_results.py',"if phase in ('Confirm','ExpandedDev'):","if False:")
 patches=[]
 for n in names:
  patches.extend(difflib.unified_diff((OLD/n).read_text(encoding='utf8').splitlines(True),(ROOT/n).read_text(encoding='utf8').splitlines(True),fromfile='frozen/'+n,tofile='D3f2/'+n))
 (ROOT/'SOURCE_CHANGES.diff').write_text(''.join(patches),encoding='utf8')
 save(ROOT/'PREFLIGHT.json',dict(status='ASSETS_AND_IDENTITIES_PASS_PENDING_ENGINEERING',centers=CENTERS,source_seeds=sorted(asset['sources']),f1_audit='VALIDATED',f1_decision='INFO_GO',protocol_sha256=sha(PKG/'D3F2_PROTOCOL_PROPOSED.json'),scientific_data_generated=False,optimizer_steps=0))
 print(json.dumps(read(ROOT/'PREFLIGHT.json'),ensure_ascii=False))
if __name__=='__main__':main()
