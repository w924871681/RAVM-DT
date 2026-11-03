"""Detached owner of the preauthorized finite GO/STOP pipeline, no model computations here."""
from d3_common import *
from f2_plan import PHASES
import psutil,subprocess,traceback
def state(**kw):
 p=ROOT/'progress/progress_state.json';s=read(p);s.update(kw,updated_unix=time.time());atomic(p,s)
def spawn(script,args=(),role=None):
 role=role or Path(script).stem
 with (ROOT/f'process/{role}.stdout.log').open('ab') as out,(ROOT/f'process/{role}.stderr.log').open('ab') as err:
  p=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(ROOT/script),*args],cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 atomic(ROOT/f'process/{role}_IDENTITY.json',dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),script=script,args=args));return p
def call(script,args=(),role=None):
 state(current_role=role or Path(script).stem)
 p=spawn(script,args,role);last=0
 while p.poll() is None:
  if time.monotonic()-last>=30:
   rec=dict(role=role or script,pid=p.pid,alive=True,available_ram_bytes=psutil.virtual_memory().available,at_unix=time.time(),automatic_retry=False);atomic(ROOT/'progress/SUPERVISION_LATEST.json',rec)
   with (ROOT/'progress/SUPERVISION_HISTORY.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(rec)+'\n')
   last=time.monotonic()
  time.sleep(3)
 atomic(ROOT/f'process/{role or Path(script).stem}_EXIT.json',dict(exit_code=p.returncode,at_unix=time.time(),automatic_retry=False));assert p.returncode==0,f'{script} failed; no blind retry'
def wait_ram():
 while psutil.virtual_memory().available<2*1024**3:state(status='WAITING_FOR_STARTUP_RAM',phase='Preflight',work='Engineering',eta_seconds=None);time.sleep(5)
 atomic(ROOT/'engineering/STARTUP_RAM_ACCEPTANCE.json',dict(status='PASS',required_bytes=2*1024**3,available_ram_bytes=psutil.virtual_memory().available,at_unix=time.time()))
def finalize():
 call('f2_report.py',role='REPORT');s=read(ROOT/'progress/progress_state.json');state(status='STOPPED' if not s.get('error') else 'FAILED',stopped_unix=time.time(),active_workers=0,active_unit=[],eta_seconds=0)
 deadline=time.monotonic()+30
 while not (ROOT/'process/OBSERVER_FINAL.json').exists() and time.monotonic()<deadline:time.sleep(1)
 atomic(ROOT/'process/SUPERVISOR_RECEIPT.json',dict(status='STOPPED',stop_decision=read(ROOT/'progress/progress_state.json').get('stop_decision'),no_new_final=True,no_paper_edits=True,no_retraining_source=True,at_unix=time.time()))
 state(work='Packaging');call('f2_delivery.py',role='DELIVERY')
def main():
 protect_originals();verify_science_lock();me=psutil.Process();atomic(ROOT/'process/SUPERVISOR_IDENTITY.json',dict(pid=me.pid,create_time=me.create_time()),exclusive=True)
 observer=spawn('f2_observer.py',role='OBSERVER');time.sleep(2);assert observer.poll() is None,'Local monitor startup failed'
 wait_ram();state(status='ENGINEERING_ACCEPTANCE',phase='Preflight',work='Engineering')
 for script,role,receipt in [('f2_acceptance.py','ACCEPTANCE','engineering/MATH_AND_EXECUTION_TESTS.json'),('f2_fixture.py','FIXTURE','engineering/FIXTURE_RESULT.json')]:
  if (ROOT/receipt).exists():
   checked=read(ROOT/receipt);assert checked['status']=='PASS' and checked['optimizer_steps']==0;verify_science_lock()
   atomic(ROOT/f'process/{role}_PRELAUNCH_REUSE.json',dict(status='VERIFIED_REUSE',receipt_sha256=sha(ROOT/receipt),execution_lock_sha256=execution_lock_sha(),engineering_only=True))
  else:call(script,role=role)
 assert read(ROOT/'engineering/FIXTURE_RESULT.json')['status']=='PASS';wait_ram();atomic(ROOT/'engineering/PREFLIGHT_ACCEPTED.json',dict(status='PASS',fixture_sha256=sha(ROOT/'engineering/FIXTURE_RESULT.json'),execution_lock_sha256=execution_lock_sha(),at_unix=time.time()),exclusive=True)
 for phase in PHASES:
  call('f2_runner.py',(phase,),phase+'_TRAINING');call('f2_audit.py',(phase,),phase+'_AUDIT');call('f2_analysis.py',(phase,),phase+'_ANALYSIS');decision=read(ROOT/f'results/{phase}/GO_STOP.json')
  if not decision['go']:state(stop_decision=decision['decision']);break
 finalize()
if __name__=='__main__':
 try:main()
 except BaseException:
  failure=traceback.format_exc();atomic(ROOT/'process/SUPERVISOR_FAILURE.json',dict(error=failure,at_unix=time.time(),automatic_retry=False));role=read(ROOT/'progress/progress_state.json').get('current_role','PREFLIGHT');reason='STOP_AUDIT' if role.endswith('_AUDIT') else 'STOP_PRECHECK' if role in ('PREFLIGHT','ACCEPTANCE','FIXTURE') else 'STOP_ENGINEERING';state(status='FAILED',error=failure,stop_decision=reason,active_unit=[])
  if not (ROOT/'D3f2_EXECUTION_REPORT_ZH.md').exists():
   try:finalize()
   except BaseException:atomic(ROOT/'process/FINALIZATION_FAILURE.json',dict(error=traceback.format_exc(),at_unix=time.time()))
  raise
