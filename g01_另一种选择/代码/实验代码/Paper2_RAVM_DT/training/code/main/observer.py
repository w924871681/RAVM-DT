"""Loopback-only live dashboard. No 15-minute logger and no scheduled LLM calls."""
from r_common import *
import http.server,threading,psutil,subprocess

HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D3f2R Smoke → 新Final</title><style>body{font:16px Segoe UI,Microsoft YaHei,sans-serif;background:#edf3f7;color:#17334c;margin:30px auto;max-width:1200px;padding:20px}h1{font-size:28px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:white;padding:22px;border-radius:15px;border:1px solid #dce5ee}.big{font-size:28px;font-weight:600}p{line-height:1.7}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;max-height:300px;overflow:auto;background:#f5f8fb;padding:12px}a{display:inline-block;margin:8px;color:#16779d}progress{width:100%}.muted{color:#658096}@media(max-width:700px){.grid{display:block}.card{margin:15px 0}}</style><h1>RAVM-DT / CMC · Smoke → 独立新Final</h1><p class="muted">V2：Final从全部11种基线中选择风险合格、活跃且MSE最低者。预设5%风险线、参数与中心均冻结。仅本地Runner；无15分钟定时汇报。</p><div class="grid"><section class="card"><div id="status" class="big">连接中</div><p id="role"></p><p id="counts"></p><progress id="bar" max="100" value="0"></progress><p id="eta"></p></section><section class="card"><h2>动态worker与资源</h2><div id="worker" class="big"></div><p id="ram"></p><p id="policy"></p><p>新任务RAM准入≥2 GiB。并发无固定数量上限，按物理资源和实测吞吐调整。</p></section><section class="card"><h2>SGD与独立审计</h2><p id="sgd"></p><p id="audit"></p><p>Smoke名义上限404,000；Final名义上限4,152,000。Frozen不训练；Fisher梯度计算单列，额外SGD为0。</p></section><section class="card"><h2>冻结裁决</h2><p id="gates"></p><pre id="checks"></pre></section><section class="card"><h2>本地压缩与SHA核验</h2><p id="pack"></p><a href="/artifacts/smoke-report">Smoke中文报告</a><a href="/artifacts/final-report">新Final中文报告</a><a href="/artifacts/return">已核验回传包</a><a href="/artifacts/verification">打包回执</a></section><section class="card"><h2>异常与边界</h2><pre id="error"></pre><p>不修改论文/回复信；原STOP_CONFIRM不变。扩展或外部验证只有实际完成后才能报告完成。</p></section></div><p class="muted" id="updated"></p><script>const e=x=>document.getElementById(x),n=x=>Number(x||0).toLocaleString();async function tick(){try{let x=await(await fetch('/api/status')).json(),s=x.state||{},w=x.workers||{},r=w.resources||{},p=x.packaging||{};e('status').textContent=s.status||'预检';e('role').textContent=(s.phase||'')+' / '+(s.work||'')+' · '+(s.current_role||'');e('counts').textContent='轨迹 '+n(s.trajectories_done)+' / '+n(s.phase_trajectories_total)+'；阶段 '+n(s.stage_rows_done)+' / '+n(s.phase_stage_rows_total)+'；M0 '+n(s.m0_created)+' / '+n(s.phase_m0_total);e('bar').value=s.phase_stage_rows_total?100*s.stage_rows_done/s.phase_stage_rows_total:0;e('eta').textContent='实测ETA '+(s.eta_seconds==null?'资源等待或样本不足':Math.ceil(s.eta_seconds/60)+'分钟');e('worker').textContent=n(w.active_workers)+' / '+n(w.admission_capacity);e('ram').textContent='系统可用RAM '+(x.available_ram_bytes/2**30).toFixed(2)+' GiB；GPU可用显存 '+((r.free_vram_bytes||0)/2**30).toFixed(2)+' GiB';e('policy').textContent=(w.mode||'')+' · '+(r.policy_reason||'等待任务');e('sgd').textContent='当前阶段实际SGD '+n(s.actual_sgd_steps)+' / '+n(s.phase_sgd_budget)+'；warmup '+n(s.actual_warmup_steps)+'；candidate '+n(s.actual_candidate_steps);e('audit').textContent='审计 '+n(s.audit_done)+' / '+n(s.audit_total)+'；审计SGD=0';e('gates').textContent='Smoke: '+(x.smoke?.decision||'未裁决')+'；Final: '+(x.final?.decision||'未解封/未裁决');e('checks').textContent=JSON.stringify(x.final?.checks||x.smoke?.checks||{},null,2);e('pack').textContent=(p.phase||'未开始')+' '+n(p.completed_files)+' / '+n(p.total_files)+' · ZIP '+((p.zip_bytes||0)/1e6).toFixed(2)+' MB';e('error').textContent=s.error||'无异常';e('updated').textContent='本地实时更新 '+new Date(x.updated_unix*1000).toLocaleString()}catch(err){e('status').textContent='连接恢复中'}}tick();setInterval(tick,5000)</script></html>'''

def optional(name):
 try:return read(ROOT/name)
 except (FileNotFoundError,json.JSONDecodeError):return None

class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  path=self.path.split('?',1)[0]
  if path in ('/','/index.html'):wire=HTML.encode('utf8');kind='text/html; charset=utf-8'
  elif path=='/api/status':
   wire=json.dumps(dict(state=optional('progress/progress_state.json'),workers=optional('progress/WORKER_STATUS.json'),packaging=optional('progress/PACKAGING_PROGRESS.json'),smoke=optional('SMOKE_GATE.json'),final=optional('NEW_FINAL_MAIN_GATE.json'),available_ram_bytes=psutil.virtual_memory().available,updated_unix=time.time()),ensure_ascii=False).encode('utf8');kind='application/json; charset=utf-8'
  else:
   receipt=optional('delivery/PACKAGE_VERIFICATION.json');mapping={'/artifacts/smoke-report':ROOT/'SMOKE_REPORT_ZH.md','/artifacts/final-report':ROOT/'NEW_FINAL_MAIN_REPORT_ZH.md','/artifacts/verification':ROOT/'delivery/PACKAGE_VERIFICATION.json'}
   if path=='/artifacts/return' and receipt and receipt['status']=='PASS':mapping[path]=Path(receipt['path'])
   p=mapping.get(path)
   if p is None or not p.is_file():self.send_error(404);return
   self.send_response(200);self.send_header('Content-Type','application/zip' if p.suffix=='.zip' else 'text/plain; charset=utf-8');self.send_header('Content-Length',str(p.stat().st_size));self.send_header('Content-Disposition',f'attachment; filename="{p.name}"');self.end_headers()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1024**2),b''):self.wfile.write(b)
   return
  self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
 def log_message(self,*a):pass

if __name__=='__main__':
 guard();p=psutil.Process();atomic(ROOT/'process/OBSERVER_IDENTITY.json',dict(pid=p.pid,create_time=p.create_time()))
 http.server.ThreadingHTTPServer(('127.0.0.1',8768),Handler).serve_forever()
