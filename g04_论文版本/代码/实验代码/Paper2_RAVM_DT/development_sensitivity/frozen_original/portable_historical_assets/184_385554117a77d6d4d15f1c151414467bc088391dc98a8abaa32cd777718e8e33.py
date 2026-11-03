"""Detached local process owner. No automatic model replay or scientific retries."""
from d3_common import *
import subprocess,psutil
def spawn(script,args=()):
 with (ROOT/f'process/{Path(script).stem}.stdout.log').open('ab') as out,(ROOT/f'process/{Path(script).stem}.stderr.log').open('ab') as err:
  return subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(ROOT/script),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,creationflags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0)
def main():
 protect_originals();verify_science_lock();me=psutil.Process();atomic(ROOT/'process/SUPERVISOR_LEASE.json',dict(pid=me.pid,create_time=me.create_time(),execution_lock_sha256=execution_lock_sha()),exclusive=True)
 observer=spawn('d3_observer_dashboard.py');runner=spawn('d3_pipeline.py');atomic(ROOT/'process/LAUNCH_RECEIPT.json',dict(supervisor_pid=me.pid,supervisor_create_time=me.create_time(),observer_pid=observer.pid,observer_create_time=psutil.Process(observer.pid).create_time(),runner_pid=runner.pid,runner_create_time=psutil.Process(runner.pid).create_time(),url='http://127.0.0.1:8767/',progress_seconds=900,supervision_seconds=1800))
 last=0
 while True:
  rc=runner.poll()
  if time.monotonic()-last>1800 or rc is not None:
   verify_science_lock();s=read(ROOT/'progress/progress_state.json') if (ROOT/'progress/progress_state.json').exists() else {};item=dict(at_unix=time.time(),runner_alive=rc is None,observer_alive=observer.poll() is None,status=s.get('status'),phase=s.get('phase'),available_ram_bytes=psutil.virtual_memory().available,errors=[],automatic_retry=False)
   atomic(ROOT/'progress/SUPERVISION_LATEST.json',item)
   with (ROOT/'progress/SUPERVISION_HISTORY.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(item)+'\n')
   last=time.monotonic()
  if rc is not None:
   if not (ROOT/'process/RUNNER_RECEIPT.json').exists():s.update(status='FAILED',error='Native exit without receipt; preserved without replay');atomic(ROOT/'progress/progress_state.json',s)
   time.sleep(5);atomic(ROOT/'process/SUPERVISOR_RECEIPT.json',dict(runner_exit_code=rc,finished_unix=time.time()));pack=spawn('d3_delivery.py');prc=pack.wait();atomic(ROOT/'process/PACKAGER_RECEIPT.json',dict(exit_code=prc,finished_unix=time.time()));return
  time.sleep(5)
if __name__=='__main__':main()
