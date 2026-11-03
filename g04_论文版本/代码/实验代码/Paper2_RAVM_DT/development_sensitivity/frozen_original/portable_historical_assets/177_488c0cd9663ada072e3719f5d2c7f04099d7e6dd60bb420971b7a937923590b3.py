"""Read-only local browser dashboard and 15 minute progress history. No Service."""
from d3_common import *
import http.server,threading,subprocess,psutil,datetime
from scripts.progress_monitor_15min import create_snapshot,format_text
PORT=8767
def optional(p):
 try:return read(p)
 except (OSError,ValueError):return None
def gpu():
 try:
  s=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip()
  return dict(raw=s,recorded_unix=time.time())
 except Exception as ex:return dict(error=str(ex),recorded_unix=time.time())
def snapshot():
 s=optional(ROOT/'progress/progress_state.json') or dict(status='PREPARING',phase='Preflight')
 z=create_snapshot(s,time.time());z['gpu_telemetry']=gpu();z.update(candidate_adaptations_done=s.get('candidate_adaptations_done'),candidate_adaptations_total=s.get('candidate_adaptations_total'),oracle_adaptations_done=s.get('oracle_adaptations_done'),oracle_adaptations_total=s.get('oracle_adaptations_total'))
 if s.get('trajectories_done',0)>=3 and s.get('parallel_compute_eta_seconds') is not None:z['eta_phase_seconds']=s['parallel_compute_eta_seconds']
 with (ROOT/'progress/PROGRESS_HISTORY.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(z,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
 atomic_text(ROOT/'progress/LATEST_PROGRESS_ZH.md',format_text(z)+'\n\n同父候选适配：'+str(z.get('candidate_adaptations_done'))+' / '+str(z.get('candidate_adaptations_total'))+'；离线Oracle：'+str(z.get('oracle_adaptations_done'))+' / '+str(z.get('oracle_adaptations_total'))+'\n\nGPU实测：'+json.dumps(z['gpu_telemetry'],ensure_ascii=False)+'\n')
 atomic(ROOT/'progress/LATEST_SNAPSHOT.json',z)
 return s
HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RAVM-DT · D3d1 实验监督</title><style>
body{margin:0;background:#0b1120;color:#dde7f4;font:15px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1140px;margin:auto;padding:38px 24px}header{display:flex;justify-content:space-between;align-items:center}h1{font-size:29px;margin:8px 0}small,.muted{color:#9aacc3}.tag{color:#68e1bc;border:1px solid #285b51;border-radius:30px;padding:7px 14px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:28px 0}.card,section{background:#131e31;border:1px solid #25334b;border-radius:12px;padding:22px}.value{font-size:29px;font-weight:650;margin:12px 0}progress{width:100%;height:13px;accent-color:#51cdaa}h2{font-size:17px;margin:0 0 15px}section{margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.7 ui-monospace,monospace;margin:0}button{color:#cde1ee;background:#263a55;border:0;padding:9px 16px;border-radius:7px;cursor:pointer}.warn{color:#f4ba78}footer{margin-top:28px;color:#92a3b9;line-height:1.8}</style><main><header><div><small>RAVM-DT / PROSPECTIVE DEVELOPMENT</small><h1>D3d1 实验监督</h1></div><span class="tag" id="status">连接中</span></header><p class="muted">两项冻结修正 · DevCal → Confirm → 条件 ExpandedDev</p><div class="grid"><div class="card"><small>当前阶段</small><div class="value" id="phase">—</div><div id="auditstate"></div></div><div class="card"><small>完整 24 阶段轨迹</small><div class="value" id="tracks">—</div><progress id="bar" max="100" value="0"></progress><p id="pct"></p></div><div class="card"><small>已封存阶段记录</small><div class="value" id="rows">—</div><small>总预算上限 70,560 条</small></div><div class="card"><small>本阶段预计剩余</small><div class="value" id="eta">未知</div><small id="elapsed">待测量</small></div></div><section><h2>动态 worker 与 GPU</h2><pre id="workers">等待状态</pre></section><section><h2>当前任务</h2><pre id="active">等待 Runner</pre></section><section><h2>独立审计进度</h2><pre id="audit">尚未开始</pre></section><section><h2>30 分钟本地监督</h2><pre id="supervision">等待首次检查</pre></section><section><h2>自动裁决与异常</h2><pre class="warn" id="errors">尚未到裁决阶段</pre></section><button id="refresh">立即刷新</button> <small id="time"></small><footer>页面每 10 秒读取进度。进度历史每 15 分钟落盘，监督每 30 分钟记录。<br>监督界面只读取工程进度与资源；不读取 Service，不参与算法决策。ETA 仅按本阶段真实完工速度计算，审计单独估计。<br>开发证据不构成现场安全保证或新 Final。关闭页面后，本地 Runner 继续运行。</footer></main><script>
const el=id=>document.getElementById(id), show=(id,x)=>el(id).textContent=x, json=x=>JSON.stringify(x,null,2),dur=s=>s==null?'未知':Math.round(s/60)+' 分钟';
async function update(){try{let z=await(await fetch('/api/status',{cache:'no-store'})).json(),s=z.state||{},w=z.workers||{},p=z.snapshot||{},a=z.audit||{};show('status',s.status||'PREPARING');show('phase',s.phase||'Preflight');show('tracks',(s.trajectories_done||0)+' / '+(s.trajectories_total||'—'));show('rows',(s.stage_rows_done||0)+' / '+(s.stage_rows_total||'—'));let pc=s.trajectories_total?100*s.trajectories_done/s.trajectories_total:0;el('bar').value=pc;show('pct',pc.toFixed(2)+'%');let eta=s.status==='AUDITING'?a.eta_seconds:s.parallel_compute_eta_seconds;show('eta',dur(eta));show('elapsed','累计耗时 '+dur(Date.now()/1000-s.run_started_unix));show('workers',json({active:w.active_workers,admission:w.admission_capacity,maximum:w.maximum,mode:w.mode,resources:w.resources,gpu:p.gpu_telemetry}));show('active',json(s.active_unit));show('audit',json(z.audit)||'尚未开始');show('supervision',json(z.supervision));show('errors',json({decision:s.auto_gate_decision,error:s.error,package:z.package}));show('time','页面更新：'+new Date().toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'}));}catch(e){show('time','读取失败：'+e.message)}}el('refresh').onclick=update;update();setInterval(update,10000);
</script></html>'''
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path in ('/','/index.html'):wire=HTML.encode('utf8');kind='text/html; charset=utf-8'
  elif self.path=='/api/status':
   s=optional(ROOT/'progress/progress_state.json') or {};phase=s.get('phase','')
   wire=canonical(dict(state=s,workers=optional(ROOT/'progress/WORKER_STATUS.json'),snapshot=optional(ROOT/'progress/LATEST_SNAPSHOT.json'),audit=optional(ROOT/f'progress/AUDIT_{phase}_PROGRESS.json'),supervision=optional(ROOT/'progress/SUPERVISION_LATEST.json'),package=optional(ROOT/'delivery/DELIVERY_RECEIPT.json')));kind='application/json; charset=utf-8'
  else:self.send_error(404);return
  self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
 def log_message(self,*args):pass
def main():
 protect_originals();server=http.server.ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
 threading.Thread(target=server.serve_forever,daemon=True).start();me=psutil.Process()
 atomic(ROOT/'process/LIVE_PROGRESS_SERVER.json',dict(pid=me.pid,create_time=me.create_time(),url=f'http://127.0.0.1:{PORT}',bind='loopback only',no_service_access=True))
 last=0;terminal=False
 while True:
  s=optional(ROOT/'progress/progress_state.json') or {};is_end=s.get('status') in ('DONE','STOPPED','FAILED','BLOCKED')
  if time.monotonic()-last>=900 or (is_end and not terminal):
   snapshot();last=time.monotonic()
   if is_end:terminal=True;atomic(ROOT/'process/OBSERVER_FINAL.json',dict(at_unix=time.time(),status=s['status']))
  time.sleep(3)
if __name__=='__main__':main()
