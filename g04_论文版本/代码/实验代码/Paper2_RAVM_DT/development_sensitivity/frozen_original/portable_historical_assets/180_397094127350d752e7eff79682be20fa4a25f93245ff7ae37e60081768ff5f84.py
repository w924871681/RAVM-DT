"""Frozen automatic resource gates. No search/retry after scientific STOP."""
from d3_common import *
import csv,collections,statistics,subprocess,psutil,itertools,threading,traceback
PHASES=('RootScreen','Pilot','DevCal','Confirm','ExpandedDev')
KEYS=dict(RootScreen='root_screen',Pilot='pilot',DevCal='devcal',Confirm='confirm',ExpandedDev='expanded_dev')
def seeds(p):return [7311] if p in ('RootScreen','Pilot','DevCal') else [7311,7312,7313]
def expected_counts(p):return dict(RootScreen=(24,576),Pilot=(84,2016),DevCal=(1272,30528),Confirm=(504,12096),ExpandedDev=(756,18144))[p]
def verify_phase_permission(p):
 verify_science_lock()
 if p=='RootScreen':return execution_lock_sha()
 prior=PHASES[PHASES.index(p)-1];v=read(ROOT/f'{prior.upper()}_GO_NO_GO.json');assert v['decision']=='GO_'+p.upper()
 assert v['science_lock_sha256']==sha(ROOT/'EXPERIMENT_LOCK.json') and v['execution_lock_sha256']==execution_lock_sha()
 ap=ROOT/f'results/{prior}/INDEPENDENT_AUDIT.json';assert sha(ap)==v['audit_sha256'];a=read(ap);assert a['status']=='VALIDATED' and a['verified_stage_rows']==expected_counts(prior)[1]
 for name,h in a['phase_results_sha256'].items():assert sha(ROOT/f'results/{prior}'/name)==h
 return sha(ROOT/f'{prior.upper()}_GO_NO_GO.json')
def readcsv(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def b(v):return v in (True,'True','true','1',1)
def summarize(rows):
 out={}
 for arm in sorted({r['arm'] for r in rows}):
  rr=[r for r in rows if r['arm']==arm];risk={};centers=sorted({int(r['center']) for r in rr});cm={}
  for s in ('clean','mixed','coherent_sticky'):
   sub=[r for r in rr if r['stream']==s];risk[s]=dict(harm=sum(b(r['harmful']) for r in sub)/len(sub),downside=statistics.mean(float(r['downside']) for r in sub),n=len(sub))
  for c in centers:cm[c]=statistics.mean(float(r['deployed_service_mse']) for r in rr if int(r['center'])==c and r['stream'] in ('clean','mixed'))
  out[arm]=dict(balanced_mse=statistics.mean(cm.values()),clean_mse=statistics.mean(float(r['deployed_service_mse']) for r in rr if r['stream']=='clean'),risk=risk,center_mse=cm,qualified=all(v['harm']<=.05 and v['downside']<=5.6036475532487204e-5 for v in risk.values()),commits=sum(b(r['commit']) for r in rr),cost=sum(float(r['maintenance_wall_seconds']) for r in rr))
 return out
def decide_root():
 rows=readcsv(ROOT/'results/RootScreen/MATCHED_CANDIDATE_RECORDS.csv');stats={}
 assert len(rows)==576
 for m in ('U_MSE','U_HUBER','U_LABEL_WEIGHT'):
  rr=[r for r in rows if r['method']==m];bycenter={}
  for c in sorted({int(r['center']) for r in rr}):
   q=[r for r in rr if int(r['center'])==c];bycenter[c]=dict(corrupt_mse=statistics.mean(float(r['candidate_service_mse']) for r in q if r['stream'] in ('mixed','coherent_sticky')),clean_mse=statistics.mean(float(r['candidate_service_mse']) for r in q if r['stream']=='clean'))
  stats[m]=dict(corrupt_mse=statistics.mean(v['corrupt_mse'] for v in bycenter.values()),clean_mse=statistics.mean(v['clean_mse'] for v in bycenter.values()),positive_delta_sum=sum(float(r['positive_delta']) for r in rr if r['stream'] in ('mixed','coherent_sticky')),cost=sum(float(r['training_seconds']) for r in rr),center_metrics=bycenter)
 base=stats['U_MSE'];eligible=[]
 for m in ('U_HUBER','U_LABEL_WEIGHT'):
  s=stats[m];s['checks']=dict(corrupt_1pct_improvement=s['corrupt_mse']<=.99*base['corrupt_mse'],clean_no_more_1pct=s['clean_mse']<=1.01*base['clean_mse'],harm_magnitude_not_worse=s['positive_delta_sum']<=base['positive_delta_sum'])
  if all(s['checks'].values()):eligible.append(m)
 chosen=min(eligible,key=lambda m:(stats[m]['corrupt_mse'],stats[m]['clean_mse'],stats[m]['cost'],m)) if eligible else None
 # Center-cluster intervals are descriptive, never override the frozen GO test.
 import numpy as np
 rng=np.random.default_rng(20261020);effects=[]
 for m in ('U_HUBER','U_LABEL_WEIGHT'):
  dif=np.array([stats[m]['center_metrics'][c]['corrupt_mse']-base['center_metrics'][c]['corrupt_mse'] for c in sorted(base['center_metrics'])]);boot=dif[rng.integers(8,size=(4000,8))].mean(1)
  effects.append(dict(method=m,n_centers=8,paired_corrupt_mse_delta=float(dif.mean()),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),scope='exploratory center cluster interval, not gate'))
 from d3_results import csvwrite
 csvwrite(ROOT/'results/RootScreen/CENTER_PAIRED_EFFECTS.csv',effects)
 return dict(decision='GO_PILOT' if chosen else 'STOP_ROOT_SCREEN',disposition=None if chosen else 'STOP_UPDATE_LABEL_HYPOTHESIS',selected_generator=chosen,selected_new_method=chosen+'+V02' if chosen else None,all_methods=stats,reason='Frozen matched-parent legal candidate quality resource gate',novelty='Huber is an existing robust method' if chosen=='U_HUBER' else 'Unconfirmed development prototype')
def decide_lifecycle(phase):
 rows=readcsv(ROOT/f'results/{phase}/STAGE_RECORDS.csv');ss=summarize(rows);root=read(ROOT/'ROOTSCREEN_GO_NO_GO.json');selected=root['selected_new_method'];s=ss[selected];base=ss['U_MSE+V02'];cfg=read(ROOT/'EXPERIMENT_LOCK.json')[KEYS[phase]];checks=dict(three_stream_risk=s['qualified'])
 doc=dict(selected_generator=root['selected_generator'],selected_new_method=selected,all_method_metrics=ss)
 if phase=='Pilot':checks.update(mse_improves_0_3pct=s['balanced_mse']<=.997*base['balanced_mse'],clean_not_degraded=s['clean_mse']<=1.01*base['clean_mse'])
 if phase=='DevCal':
  choices={};active=[];grid=read(ROOT/'RUN_MANIFEST.json')['baseline_grid'];scale=.00560364755324872
  for m in ('DUAL_MEAN','RECENT_MEAN','MWU_MEAN'):
   eligible=[a for a in ss if a.startswith(m+'__g') and ss[a]['qualified']]
   if not eligible:choices[m]=dict(fallback=True,threshold=None,selected_grid_arm=None)
   else:
    a=min(eligible,key=lambda a:(ss[a]['balanced_mse'],max(v['harm'] for v in ss[a]['risk'].values()),ss[a]['cost'],a));choices[m]=dict(fallback=False,threshold=grid[int(a[-2:])]*scale,selected_grid_arm=a)
    if ss[a]['commits']>0:active.append(a)
  best=min(active,key=lambda a:ss[a]['balanced_mse']) if active else None
  checks.update(coherent_buffer=s['risk']['coherent_sticky']['harm']<=.045,mse_improves_0_3pct=s['balanced_mse']<=.997*base['balanced_mse'],active_qualified_strong_exists=best is not None,within_0_5pct_best_strong=best is not None and s['balanced_mse']<=1.005*ss[best]['balanced_mse'])
  doc.update(baseline_selection=choices,best_active_qualified_strong=best)
 if phase=='Confirm':
  sel=read(ROOT/'DEVCAL_SELECTED_LOCK.json');active=[a for a in ss if a.endswith('_SELECTED') and ss[a]['qualified'] and ss[a]['commits']>0 and not sel['baseline_selection'][a.removesuffix('_SELECTED')]['fallback']];best=min(active,key=lambda a:ss[a]['balanced_mse']) if active else None
  wins=sum(s['center_mse'][c]<ss[best]['center_mse'][c] for c in s['center_mse']) if best else 0;ratio=s['cost']/ss['RAVM_ORIGINAL']['cost']
  checks.update(active_qualified_strong_exists=best is not None,beats_strong_0_2pct=best is not None and s['balanced_mse']<=.998*ss[best]['balanced_mse'],below_fixed_v02=s['balanced_mse']<base['balanced_mse'],four_centers_better=wins>=4,runtime_at_most_2_5x=ratio<=2.5);doc.update(best_active_qualified_strong=best,center_wins=wins,runtime_ratio=ratio)
 audit=read(ROOT/f'results/{phase}/INDEPENDENT_AUDIT.json');checks['independent_audit']=audit['status']=='VALIDATED'
 nxt=dict(Pilot='DEVCAL',DevCal='CONFIRM',Confirm='EXPANDEDDEV').get(phase)
 doc.update(decision='GO_'+nxt if all(checks.values()) else 'STOP_'+phase.upper(),checks=checks,failed_checks=[k for k,v in checks.items() if not v],disposition='CANDIDATE_IMPROVED_VERIFICATION_BLOCKER' if not s['qualified'] else None)
 return doc
def run_phase(phase,state,runid):
 from d3_engine import generate,IDS,initial,sync
 from d3_trajectory import run_trajectory
 from d3_runner import arms
 from d3_parallel import execute
 verify_phase_permission(phase);state.update(phase=phase,status='GENERATING_INPUTS',phase_started_unix=time.time(),trajectories_done=0,trajectories_total=expected_counts(phase)[0],stage_rows_done=0,stage_rows_total=expected_counts(phase)[1],candidate_adaptations_done=0,candidate_adaptations_total=576 if phase=='RootScreen' else None)
 atomic(ROOT/'progress/progress_state.json',state);generate(phase)
 # Common M0 is materialized serially by coordinator, before competing methods.
 def initialize(job):
  m=initial(*job);del m;sync();return True
 state.update(status='INITIALIZING',active_unit='DYNAMIC_COMMON_M0_INITIALIZATION');atomic(ROOT/'progress/progress_state.json',state)
 execute(itertools.product(IDS[phase],seeds(phase)),initialize,'INITIALIZE_'+phase)
 jobs=list(itertools.product(IDS[phase],seeds(phase),('clean','mixed','coherent_sticky'),arms(phase)));assert len(jobs)==expected_counts(phase)[0]
 counts={i:0 for i in range(len(jobs))};lock=threading.Lock()
 def work(job):
  c,seed,stream,arm=job;i=next(i for i,j in enumerate(jobs) if j is job)
  def progress(stage,**kw):
   with lock:counts[i]=stage
  return run_trajectory(phase,c,seed,stream,arm,runid,progress)
 def tick(z):
  with lock:n=sum(counts.values())
  state.update(status='RUNNING',stage_rows_done=n,trajectories_done=z['completed'],active_workers=z['active_workers'],parallel_compute_eta_seconds=z['eta_seconds'],active_unit=[dict(center=jobs[i][0],seed=jobs[i][1],stream=jobs[i][2],arm=jobs[i][3]['id'],stage=min(24,counts[i]+1)) for i in z['active_indices']],elapsed_phase_seconds=z['elapsed_seconds'])
  if phase=='RootScreen':
   state['candidate_adaptations_done']=sum(3*sum(t<=v for t in (2,3,4,5,9,10,19,20)) for v in counts.values());state['oracle_adaptations_done']=sum(3*sum(t<=v for t in (3,4)) for i,v in counts.items() if jobs[i][2]=='coherent_sticky');state['oracle_adaptations_total']=48
  if n>state.get('last_count',0):state.update(last_stage_commit_unix=time.time(),last_count=n)
  atomic(ROOT/'progress/progress_state.json',state)
 execute(jobs,work,'COMPUTE_'+phase,tick)
def call_audit(phase):
 with (ROOT/f'process/AUDIT_{phase}.stdout.log').open('ab') as out,(ROOT/f'process/AUDIT_{phase}.stderr.log').open('ab') as err:
  p=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(ROOT/'d3_audit_new.py'),phase],cwd=ROOT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));atomic(ROOT/f'process/AUDIT_{phase}_IDENTITY.json',dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),started_unix=time.time()));rc=p.wait()
 if rc:raise RuntimeError(f'Independent {phase} audit exit {rc}; see logs; no automatic scientific retry')
def main():
 protect_originals();verify_science_lock();runid='D3D1_'+str(time.time_ns());me=psutil.Process();atomic(ROOT/'process/RUNNER_IDENTITY.json',dict(pid=me.pid,create_time=me.create_time(),run_id=runid),exclusive=True)
 state=dict(run_id=runid,run_started_unix=time.time(),phase='Preflight',status='STARTING',auto_gate_decision=None,error=None);atomic(ROOT/'progress/progress_state.json',state);completed=[];code=0
 try:
  # Engineering acceptance uses only old development inputs. Its GPU process is
  # admitted separately, before any prospective pool is generated.
  while psutil.virtual_memory().available<3*1024**3:
   state.update(status='WAITING_FOR_MEMORY',active_unit='WAITING_FOR_SAFE_GPU_ENGINEERING_ACCEPTANCE',available_ram_bytes=psutil.virtual_memory().available);atomic(ROOT/'progress/progress_state.json',state);time.sleep(5)
  state.update(status='ENGINEERING_QA',active_unit='OLD_GPU_REGRESSION_AND_NEW_TRAINING_FORMULA_TESTS');atomic(ROOT/'progress/progress_state.json',state)
  with (ROOT/'engineering_test/ENGINEERING_TESTS.log').open('ab') as log:
   test=subprocess.Popen([sys.executable,'-X','utf8','-B','-u',str(ROOT/'test_engineering.py')],cwd=ROOT,stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));rc=test.wait()
  if rc or not read(ROOT/'engineering_test/ENGINEERING_QA.json')['passed']:raise RuntimeError('Engineering acceptance failed before scientific input generation; inspect preserved test evidence')
  for phase in PHASES:
   run_phase(phase,state,runid);state.update(status='AUDITING',active_unit='INDEPENDENT_FULL_CAUSAL_REAL_WEIGHT_AUDIT');atomic(ROOT/'progress/progress_state.json',state);call_audit(phase);completed.append(phase)
   if phase=='ExpandedDev':state.update(status='DONE',auto_gate_decision='STOP_AFTER_EXPANDED');break
   doc=decide_root() if phase=='RootScreen' else decide_lifecycle(phase);doc.update(science_lock_sha256=sha(ROOT/'EXPERIMENT_LOCK.json'),execution_lock_sha256=execution_lock_sha(),audit_sha256=sha(ROOT/f'results/{phase}/INDEPENDENT_AUDIT.json'),decision_unix=time.time());atomic(ROOT/f'{phase.upper()}_GO_NO_GO.json',doc,exclusive=True)
   if phase=='DevCal':atomic(ROOT/'DEVCAL_SELECTED_LOCK.json',doc,exclusive=True)
   state['auto_gate_decision']=doc['decision']
   if not doc['decision'].startswith('GO_'):state.update(status='STOPPED',active_unit=doc['decision']);break
  state['phases_completed']=completed
 except BaseException as ex:
  code=1;state.update(status='FAILED',error=str(ex),active_unit='SAFE_STOP_EVIDENCE_PRESERVED');atomic_text(ROOT/f'process/FAILURE_{time.time_ns()}.txt',traceback.format_exc())
 state['finished_unix']=time.time();atomic(ROOT/'progress/progress_state.json',state);atomic(ROOT/'process/RUNNER_RECEIPT.json',dict(exit_code=code,status=state['status'],run_id=runid,finished_unix=time.time(),phases_completed=completed,error=state.get('error')))
 return code
if __name__=='__main__':raise SystemExit(main())
