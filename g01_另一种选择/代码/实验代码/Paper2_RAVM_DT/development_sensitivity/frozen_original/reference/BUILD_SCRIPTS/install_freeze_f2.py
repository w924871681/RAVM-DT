"""Complete and freeze the isolated implementation before all scientific data generation."""
from pathlib import Path
import json,hashlib,shutil,ast,difflib,time
BASE=Path(__file__).resolve().parent;ROOT=BASE/'D3f2_ISOLATED_WORK';OLD=BASE.parent/'ravm_d3d1_20261020/D3d1_ISOLATED_WORK'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 assert not (ROOT/'D3F2_EXECUTION.lock.json').exists();assert not list((ROOT/'data').iterdir())
 for p in BASE.iterdir():
  if p.name.startswith(('f2_','test_f2')) or p.name=='d3_runner.py':shutil.copy2(p,ROOT/p.name)
 def patch(n,old,new):
  p=ROOT/n;s=p.read_text(encoding='utf8');assert old in s,(n,old);p.write_text(s.replace(old,new),encoding='utf8')
 patch('d3_audit.py',"scope='D3c_'+phase","scope='D3f2_'+phase")
 patch('d3_audit.py',"assert uu.hash==s['update_hash'] and vv.hash==s['verification_hash']", "assert uu.hash==s['update_hash'] and vv.hash==s['verification_hash']\n    from d3_adapt import batch_hash\n    a=s['adapt_meta'];assert a['generator']=='U_MSE' and a['steps']==50 and a['lr']==.01 and a['batch']==64 and a['grad_clip']==1.\n    assert a['sampled_indices_sha256']==batch_hash(len(windows(uu)[0]),s['training_seed'])")
 patch('d3_trajectory.py',"row['scope']='D3F2_SYNTHETIC_DEVELOPMENT_NOT_FINAL'", "row['phase']=phase\n  row['scope']='D3F2_SYNTHETIC_DEVELOPMENT_NOT_FINAL'")
 patches=[]
 for p in ROOT.glob('*.py'):
  if (OLD/p.name).exists():patches.extend(difflib.unified_diff((OLD/p.name).read_text(encoding='utf8').splitlines(True),p.read_text(encoding='utf8').splitlines(True),fromfile='frozen/'+p.name,tofile='D3f2/'+p.name))
 (ROOT/'SOURCE_CHANGES.diff').write_text(''.join(patches),encoding='utf8')
 save(ROOT/'ANALYSIS_IMPLEMENTATION_LOCK.json',dict(primary='Center-equal clean/mixed lifecycle MSE; equally average all seeds and 24 stages inside each center/stream.',risk='Per stream harmful committed stages / all stage opportunities, downside mean over all opportunities, with all centers equally weighted by complete balanced matrix.',devcal_buffer_all_calibrated_methods=.045,active='At least one clean/mixed deployed commit across full calibrated population.',selection='Among qualified active candidates min balanced MSE, then lower actual matched maintenance cost, smaller lambda, larger margin, final lexicographic ID.',baseline_no_eligible_active='Choose min MSE qualified inactive grid arm if any, otherwise min MSE grid arm marked unqualified and descriptive only. Such arms can never be the best qualified active comparator.',bootstrap=dict(unit='center including all seeds/streams/stages',draws=4000,seed=20261020),no_margin_or_lambda_expansion=True,no_confirm_selection=True,cost='Matched actual per-stage maintenance path including training, verification, model snapshot/cache, seal, and existing v0.2 past feedback where applicable. Postseal Service and independent audit separate.',large_CAS='Over 256MiB summed stage CAS may be external only with path+size+SHA256 index; never delete local weights.',engineering_SGD_budget=0))
 ref=ROOT/'reference/BUILD_SCRIPTS';ref.mkdir()
 for n in ('prepare_f2.py','install_freeze_f2.py'):shutil.copy2(BASE/n,ref/n)
 for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
 originals=json.loads((ROOT/'SOURCE_COPY_IDENTITIES.json').read_text()) if (ROOT/'SOURCE_COPY_IDENTITIES.json').exists() else {}
 code={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml','.html')}
 for n in ('EXPERIMENT_LOCK.json','D3F2_PROTOCOL_LOCK.json','RUN_MANIFEST.json','HISTORICAL_ID_REGISTRY.json','ASSET_IDENTITY.json','ORIGINAL_INPUT_IDENTITIES.json','GLOBAL_CENTER_PREFLIGHT.json','ANALYSIS_IMPLEMENTATION_LOCK.json','SOURCE_CHANGES.diff'):code[n]=sha(ROOT/n)
 save(ROOT/'D3F2_EXECUTION.lock.json',dict(schema='D3F2_PRE_DATA_EXECUTION_LOCK_V1',input_protocol_sha256=sha(ROOT/'EXPERIMENT_LOCK.json'),code_sha256=code,created_unix=time.time(),scope='Preauthorized finite pipeline; scientific configuration frozen before any new data',max_candidate_steps=3283200,max_warmup_steps=3600,max_target_steps=3286800))
 save(ROOT/'RUN_AUTHORIZATION_RECEIPT.json',dict(status='AUTHORIZED_BY_USER_MESSAGE',phases=['Preflight','P0','DevCal','Confirm','ExpandedDev'],ordinary_phase_approval_not_required=True,automatic_stop_on_no_go=True,no_new_final=True,no_paper_edits=True,execution_lock_sha256=sha(ROOT/'D3F2_EXECUTION.lock.json'),at_unix=time.time()))
 save(ROOT/'progress/progress_state.json',dict(run_id='D3F2_20261020_ATTEMPT01',status='PREPARED',phase='Preflight',work='Engineering',run_started_unix=time.time(),actual_sgd_steps=0,actual_candidate_steps=0,actual_warmup_steps=0,trajectories_done=0,stage_rows_done=0,audit_done=0,eta_seconds=None,error=None,stop_decision=None,active_unit=[]))
 print('FROZEN',sha(ROOT/'D3F2_EXECUTION.lock.json'),'code files',len(code))
if __name__=='__main__':main()
