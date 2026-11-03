"""Build an isolated, provenance-bound clean-input correction; no optimizer execution."""
from pathlib import Path
import ast,collections,concurrent.futures as cf,hashlib,json,os,shutil,sys,threading,time
import psutil

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'D3f2R_SMOKE_NEW_FINAL_20261021'
LEGACY=('ALWAYS_FT','ER','EWC','HUBER_FT','VALIDATION_COMMIT','FROZEN')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(262144),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 assert not p.exists(),str(p)
 p.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf8')
def patch(name,old,new):
 p=ROOT/name;s=p.read_text(encoding='utf8');assert s.count(old)==1,(name,old)
 p.write_text(s.replace(old,new),encoding='utf8')

def main():
 assert not (ROOT/'BUILD_STARTED.json').exists(),'One correction build only'
 write(ROOT/'BUILD_STARTED.json',dict(at_unix=time.time(),authorization=['修正，你能修正嘛，还是需要gpt来修正？','那么修正后重新执行']))
 oldlock=read(OLD/'D3F2_EXECUTION.lock.json');oldsha=sha(OLD/'D3F2_EXECUTION.lock.json')
 for name,h in oldlock['code_sha256'].items():assert sha(OLD/name)==h,name
 assert read(OLD/'process/Smoke_TRAINING_RECEIPT.json')['actual_total_steps']==394400
 assert not (OLD/'SMOKE_GATE.json').exists() and not (OLD/'data/NewFinal_IDENTITY.json').exists()
 for directory in ('process','progress','science','initial','data','delivery','engineering','reference'):(ROOT/directory).mkdir(exist_ok=True)
 for name in oldlock['code_sha256']:
  if name in ('RUN_PROTOCOL_LOCK.json','RUN_PROTOCOL_SIGNATURE.json'):continue
  dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OLD/name,dest)
 for name in ('ROLE_FREEZE_RECEIPT.json',):shutil.copyfile(OLD/name,ROOT/name)
 for name in ('D3F2_EXECUTION.lock.json','RUN_PROTOCOL_LOCK.json','RUN_PROTOCOL_SIGNATURE.json'):
  shutil.copyfile(OLD/name,ROOT/'reference'/('FAILED_'+name))
 for name in ('LEGACY_INPUT_MISMATCH_DIAGNOSIS.json','AUDIT_STOP_EXPLANATION_ZH.md','OWNER_RECOVERY_FAILURE.json'):
  shutil.copyfile(OLD/'engineering/recovery_20261021'/name,ROOT/'reference'/name)
 for name in ('Smoke_AUDIT.stderr.log','Smoke_AUDIT_EXIT.json','Smoke_TRAINING_RECEIPT.json'):
  shutil.copyfile(OLD/'process'/name,ROOT/'reference'/('FAILED_'+name))
 shutil.copyfile(OLD/'process/DATA_ROLE_EVENTS.jsonl',ROOT/'process/DATA_ROLE_EVENTS.jsonl')
 manifest=read(OLD/'RUN_MANIFEST.json');manifest['run_id']='D3F2R_CLEAN_INPUT_CORRECTION_20261021'
 (ROOT/'RUN_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 patch('legacy_adapter.py',"alpha==.1 and location=='both'", "alpha==(0. if kind=='clean' else .1) and location=='both'")
 patch('legacy_adapter.py',"kind=stream,alpha=.1,location='both'", "kind=stream,alpha=(0. if stream=='clean' else .1),location='both'")
 patch('legacy_adapter.py',"s['execution_lock_sha256']==execution_lock_sha()", "s['execution_lock_sha256'] in allowed_execution_shas()")
 patch('d3_common.py',"def allowed_execution_shas():return {execution_lock_sha()}",
       "def allowed_execution_shas():return {execution_lock_sha(),read(ROOT/'CORRECTION_PROTOCOL.json')['original_execution_sha256']}")
 patch('d3_common.py'," return lock\n", " prior=read(ROOT/'CORRECTION_PROTOCOL.json');origin=Path(prior['original_root']);assert sha(origin/'D3F2_EXECUTION.lock.json')==prior['original_execution_sha256']\n for rel,h in read(ROOT/'reference/FAILED_D3F2_EXECUTION.lock.json')['code_sha256'].items():assert sha(origin/rel)==h,rel\n return lock\n")
 # Keep every input/score/model assertion; bind old lock acceptance to exact reused directories.
 patch('new_audit.py',"  if td.name in LEGACY_IDS:record=audit_legacy(td)",
       "  reuse=read(ROOT/'REUSE_PROVENANCE.json')['trajectories']\n  expected_lock=read(ROOT/'CORRECTION_PROTOCOL.json')['original_execution_sha256'] if td.relative_to(ROOT).as_posix() in reuse else execution_lock_sha()\n  assert all(read(td/f'seal_{t:03}.json')['execution_lock_sha256']==expected_lock for t in range(1,25))\n  if td.name in LEGACY_IDS:record=audit_legacy(td)")
 patch('new_audit.py'," matrix=jobs(phase);paths=", " from correction_integrity import verify_reuse\n verify_reuse()\n matrix=jobs(phase);paths=")
 patch('new_audit.py'," assert frozen['event_seq']<data['generation_start_seq']<data['generation_end_seq']", " assert frozen['event_seq']<data['generation_start_seq']<data['generation_end_seq']\n if phase=='NewFinal':assert read(ROOT/'CORRECTION_FREEZE_RECEIPT.json')['event_seq']<data['generation_start_seq']")
 # Native engine, optimizer, statistics, scientific thresholds and source assets remain byte-identical.
 patches={n:dict(before=oldlock['code_sha256'][n],after=sha(ROOT/n)) for n in ('legacy_adapter.py','d3_common.py','new_audit.py','RUN_MANIFEST.json')}
 assert sha(ROOT/'runtime/core/engine.py')==oldlock['code_sha256']['runtime/core/engine.py']
 source=ast.parse((OLD/'engineering/recovery_20261021/recovery.py').read_text(encoding='utf8'))
 needed={'load_accounting','RecoveryMeter','check_instrumentation','call_key'}
 copied='\n\n'.join(ast.get_source_segment((OLD/'engineering/recovery_20261021/recovery.py').read_text(encoding='utf8'),n) for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in needed)
 (ROOT/'correction_meter.py').write_text('from r_common import *\nimport ast,inspect,contextlib,threading\n'+copied+'\n',encoding='utf8')
 todo=[];reused={};replacements=[];excluded=[]
 dirs=sorted((OLD/'science/Smoke').glob('*/*/*/*'));assert len(dirs)==336
 for td in dirs:
  rel=td.relative_to(OLD).as_posix();parts=td.relative_to(OLD/'science/Smoke').parts
  assert read(td/'DONE.json')['stages']==24
  if parts[2]=='clean' and parts[3] in LEGACY:
   replacements.append(dict(center=int(parts[0]),seed=int(parts[1]),stream=parts[2],arm=parts[3],old_path=rel,old_done_sha256=sha(td/'DONE.json')))
   excluded.append(rel);continue
  reused[rel]=dict(done_sha256=sha(td/'DONE.json'),original_execution_sha256=oldsha)
  todo.extend((p,p.relative_to(OLD).as_posix()) for p in td.rglob('*') if p.is_file())
 for directory in ('initial',):todo.extend((p,p.relative_to(OLD).as_posix()) for p in (OLD/directory).rglob('*') if p.is_file())
 data=read(OLD/'data/Smoke_IDENTITY.json')
 todo.extend((OLD/row['path'],row['path']) for row in data['rows'])
 todo.append((OLD/'data/Smoke_IDENTITY.json','data/Smoke_IDENTITY.json'))
 assert len(replacements)==24 and len(reused)==312
 assert not any((ROOT/x['old_path']).exists() for x in replacements)
 def transfer(job):
  src,rel=job;dst=ROOT/rel;dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists()
  shutil.copyfile(src,dst);h=sha(src);assert sha(dst)==h
  return rel,h
 files={};pending=iter(todo);active=set();exhausted=False;last=0
 with cf.ThreadPoolExecutor(max_workers=psutil.cpu_count() or 1) as pool:
  while not exhausted or active:
   available=psutil.virtual_memory().available
   cap=max(0,min(psutil.cpu_count() or 1,int((available-2*2**30)//(32*2**20))))
   while not exhausted and len(active)<cap:
    try:job=next(pending)
    except StopIteration:exhausted=True;break
    active.add(pool.submit(transfer,job))
   if active:
    done,_=cf.wait(active,timeout=.5,return_when=cf.FIRST_COMPLETED)
    for f in done:
     active.remove(f);rel,h=f.result();files[rel]=h
   else:time.sleep(1)
   if time.time()-last>10:
    print(json.dumps(dict(copy_verified=len(files),total_files=len(todo),active_workers=len(active),resource_capacity=cap,available_GiB=round(available/2**30,2))),flush=True);last=time.time()
 write(ROOT/'REUSED_FILES_SHA256.json',files)
 write(ROOT/'REUSE_PROVENANCE.json',dict(original_root=str(OLD),original_execution_sha256=oldsha,trajectories=reused,excluded_trajectories=excluded,files=len(files),all_copied_files_sha256_verified=True,no_hardlinks_or_junctions=True))
 write(ROOT/'CORRECTION_TASKS.json',replacements)
 # The derived ledger contains only retained scientific work; the complete failed ledger stays preserved.
 oldlines=(OLD/'process/Smoke_SGD_CALL_ACCOUNTING.jsonl').read_text(encoding='utf8').splitlines();kept=[];drop=[]
 for line in oldlines:
  e=json.loads(line);identity=json.loads(e['call_id'])['identity']
  bad=e['kind']=='candidate' and identity['stream']=='clean' and identity['arm'] in LEGACY
  (drop if bad else kept).append(line)
 (ROOT/'process/Smoke_SGD_CALL_ACCOUNTING.jsonl').write_text('\n'.join(kept)+'\n',encoding='utf8')
 shutil.copyfile(OLD/'process/Smoke_SGD_CALL_ACCOUNTING.jsonl',ROOT/'reference/FAILED_FULL_SGD_CALL_ACCOUNTING.jsonl')
 write(ROOT/'EXCLUDED_OPTIMIZER_CALLS.json',[json.loads(v) for v in drop])
 retained_steps=sum(json.loads(v)['steps'] for v in kept if json.loads(v)['event']=='DONE')
 excluded_steps=sum(json.loads(v)['steps'] for v in drop if json.loads(v)['event']=='DONE')
 assert retained_steps==370400 and excluded_steps==24000
 (ROOT/'reference/RETAINED_LEDGER_PREFIX.jsonl').write_bytes((ROOT/'process/Smoke_SGD_CALL_ACCOUNTING.jsonl').read_bytes())
 protocol=dict(kind='EXPLICIT_ENGINEERING_CORRECTION_NOT_NEW_INDEPENDENT_SMOKE',original_root=str(OLD),original_execution_sha256=oldsha,
  authorization=['修正，你能修正嘛，还是需要gpt来修正？','那么修正后重新执行'],new_training_trajectories=24,reused_trajectories=312,
  correction_optimizer_steps=24000,correction_optimizer_steps_exclusive_cap=24000,additional_m0_steps=0,
  reused_optimizer_steps=370400,excluded_failed_optimizer_steps=24000,failed_round_actual_steps=394400,
  corrected_composite_expected_steps=394400,corrected_composite_nominal_cap=404000,combined_actual_smoke_cost_expected=418400,
  correction_rule='clean native alpha=0, all other streams unchanged; no new candidate/parameter/seed/center selection',patches=patches,
  existing_smoke_data_reused=True,smoke_is_development_only=True,new_final_data_unopened=True,downstream_original_gates_unchanged=True,
  no_paper_edits=True,at_unix=time.time())
 write(ROOT/'CORRECTION_PROTOCOL.json',protocol)
 write(ROOT/'COST_ACCOUNTING.json',dict(status='PLANNED',failed_round=394400,retained=370400,excluded=24000,new_correction_budget=24000,new_correction_actual=None,combined_smoke_actual=None))
 state=dict(status='AWAITING_INPUT_CONTRACT_PREFLIGHT',run_id=manifest['run_id'],phase='Smoke',work='StartupAdmission',current_role='CORRECTION_PREFLIGHT',
  trajectories_done=312,phase_trajectories_total=336,stage_rows_done=7488,phase_stage_rows_total=8064,m0_created=16,phase_m0_total=16,
  actual_warmup_steps=800,actual_candidate_steps=369600,actual_sgd_steps=370400,phase_sgd_budget=404000,remaining_sgd_budget=33600,
  audit_done=0,audit_total=8064,active_workers=0,error=None,eta_seconds=None,run_started_unix=time.time(),phase_started_unix=time.time(),updated_unix=time.time())
 write(ROOT/'progress/progress_state.json',state)
 write(ROOT/'progress/Smoke_M0_PROGRESS.json',dict(status='DONE',completed=16,total=16,active_workers=0,reused=True,updated_unix=time.time()))
 shutil.copytree(OLD/'engineering/dashboard_v2',ROOT/'engineering/dashboard_v2',ignore=shutil.ignore_patterns('*.png','*.log','*.json'))
 hp=ROOT/'engineering/dashboard_v2/index.html';hp.write_text(hp.read_text(encoding='utf8').replace('Smoke · 唯一一次','Smoke · clean 输入修正'),encoding='utf8')
 for name,h in oldlock['code_sha256'].items():assert sha(OLD/name)==h,name
 write(ROOT/'BUILD_RECEIPT.json',dict(status='PASS',reused_trajectories=312,correction_trajectories=24,reused_files=len(files),source_original_frozen_files_unchanged=True,copy_hashes_verified=True,at_unix=time.time()))
 print(json.dumps(dict(status='BUILD_PASS',root=str(ROOT),reused=312,correction=24,extra_sgd=24000)),flush=True)

if __name__=='__main__':main()
