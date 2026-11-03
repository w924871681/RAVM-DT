"""Local read-only dashboard and 900-second persisted progress, no scientific imports."""
from d3_common import *
import http.server,threading,psutil,subprocess
from dynamic_worker_resources_v2 import snapshot
cache={};mutex=threading.RLock()
def optional(p,default=None):
 try:return read(p)
 except (FileNotFoundError,json.JSONDecodeError):return default
def current():
 state=optional(ROOT/'progress/progress_state.json',{});workers=optional(ROOT/'progress/WORKER_STATUS.json',{});package=optional(ROOT/'progress/PACKAGING_PROGRESS.json',{});manifest=read(ROOT/'RUN_MANIFEST.json');decisions={p:optional(ROOT/f'results/{p}/GO_STOP.json') for p in manifest['centers']};res=snapshot()
 gpu={}
 try:
  wire=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));name,used,total,util,temp=wire.strip().split(',');gpu=dict(name=name.strip(),used_mb=int(used),total_mb=int(total),utilization=int(util),temperature=int(temp))
 except (OSError,ValueError,subprocess.SubprocessError):pass
 live=[]
 for p in (ROOT/'process').glob('*IDENTITY.json'):
  identity=optional(p,{})
  try:
   process=psutil.Process(identity['pid'])
   if process.create_time()==identity['create_time'] and process.is_running():live.append(dict(role=p.stem,pid=process.pid,rss_bytes=process.memory_info().rss))
  except (psutil.Error,KeyError):pass
 return dict(state=state,workers=workers,packaging=package,package=optional(ROOT/'delivery/PACKAGE_VERIFICATION.json'),resources=dict(**res,gpu=gpu,processes=live),decisions=decisions,run_id=manifest['run_id'],updated_unix=time.time(),report_ready=(ROOT/'D3f2_EXECUTION_REPORT_ZH.md').exists())
def refresh():
 last=0;final=False
 while True:
  try:
   now=current()
   with mutex:cache.update(now)
   done=now['state'].get('status') in ('STOPPED','FAILED')
   if not final and (time.time()-last>=900 or done):
    atomic(ROOT/'progress/LATEST_PROGRESS.json',now)
    with (ROOT/'progress/PROGRESS_HISTORY.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(now,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    s=now['state'];text=f"# D3f2 实际进度\n\n阶段 {s.get('phase')} / {s.get('work')}；状态 {s.get('status')}。\n\n轨迹 {s.get('trajectories_done',0)}/{s.get('phase_trajectories_total','未知')}；阶段记录 {s.get('stage_rows_done',0)}/{s.get('phase_stage_rows_total','未知')}；审计 {s.get('audit_done',0)}/{s.get('audit_total','未知')}。\n\n实际SGD {s.get('actual_sgd_steps',0)}；本阶段M0 {s.get('m0_created',0)}。已耗时 {time.time()-s.get('run_started_unix',time.time()):.1f} 秒；当前工作实测ETA {s.get('eta_seconds') if s.get('eta_seconds') is not None else '未知/样本不足'} 秒。\n\n资源与worker：\n```json\n{json.dumps(now['resources'],ensure_ascii=False,indent=2)}\n{json.dumps(now['workers'],ensure_ascii=False,indent=2)}\n```\n\n异常 {s.get('error')}；停止/GO {s.get('stop_decision')}。\n"
    atomic_text(ROOT/'progress/LATEST_PROGRESS_ZH.md',text);last=time.time()
   if done and not final:
    atomic(ROOT/'process/OBSERVER_FINAL.json',dict(at_unix=time.time(),status=now['state']['status']));final=True
  except Exception as ex:
   with mutex:cache['monitor_error']=repr(ex)
  time.sleep(10)
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  path=self.path.split('?',1)[0]
  if path in ('/','/index.html'):wire=(ROOT/'f2_dashboard.html').read_bytes();kind='text/html; charset=utf-8'
  elif path=='/api/status':
   with mutex:wire=json.dumps(cache,ensure_ascii=False).encode('utf8')
   kind='application/json; charset=utf-8'
  elif path in ('/artifacts/report','/artifacts/return','/history/d3f1/report','/history/d3f1/return'):
   f1=ROOT.parent.parent/'ravm_d3f1_20261020/D3f1_ISOLATED_WORK'
   mapping={'/artifacts/report':ROOT/'D3f2_EXECUTION_REPORT_ZH.md','/artifacts/return':ROOT/'delivery/D3f2_EXECUTION_RETURN.zip','/history/d3f1/report':f1/'D3f1_INFO_REPORT_ZH.md','/history/d3f1/return':f1/'delivery/D3f1_INFO_RETURN.zip'};p=mapping[path]
   if path=='/artifacts/return' and not (ROOT/'delivery/PACKAGE_VERIFICATION.json').exists():self.send_error(404,'Package not verified yet');return
   if not p.is_file():self.send_error(404);return
   self.send_response(200);self.send_header('Content-Type','application/zip' if p.suffix=='.zip' else 'text/plain; charset=utf-8');self.send_header('Content-Length',str(p.stat().st_size));self.send_header('Content-Disposition',f'attachment; filename="{p.name}"');self.end_headers()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1024**2),b''):self.wfile.write(b)
   return
  else:self.send_error(404);return
  self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
 def log_message(self,*args):pass
def main():
 protect_originals();me=psutil.Process();atomic(ROOT/'process/OBSERVER_IDENTITY.json',dict(pid=me.pid,create_time=me.create_time()));cache.update(current());threading.Thread(target=refresh,daemon=True).start();http.server.ThreadingHTTPServer(('127.0.0.1',8768),Handler).serve_forever()
if __name__=='__main__':main()
