"""Post-seal scientific summaries and independent center-cluster intervals."""
from d3_common import *
import csv,math
import numpy as np
LIMIT=5.6036475532487204e-5
def csvwrite(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def aggregate(rows):
 return dict(opportunities=len(rows),service_mse=float(np.mean([r['deployed_service_mse'] for r in rows])),harmful_count=sum(r['harmful'] for r in rows),harmful_frequency=float(np.mean([r['harmful'] for r in rows])),mean_downside=float(np.mean([r['downside'] for r in rows])),commit_rate=float(np.mean([r['commit'] for r in rows])),beneficial_commits=sum(r['beneficial_commit'] for r in rows),maintenance_wall_seconds=sum(r['maintenance_wall_seconds'] for r in rows),late_score_seconds=sum(r['late_score_seconds'] for r in rows),training_seconds=sum(r['training_seconds'] for r in rows),retain_count=sum(not r['commit'] for r in rows),fallback_count=sum(r['fallback'] for r in rows),authenticated_feedback_active_count=sum(r['feedback_active'] for r in rows),bounded_release_count=sum(r['reason']=='BOUNDED_A_RELEASE' and r['commit'] for r in rows),bounded_release_beneficial_audit_only=sum(r['reason']=='BOUNDED_A_RELEASE' and r['beneficial_commit'] for r in rows),bounded_release_harmful_audit_only=sum(r['reason']=='BOUNDED_A_RELEASE' and r['harmful'] for r in rows))
def balanced(rows):
 center_stream={}
 for r in rows:
  if r['stream'] in ('clean','mixed'):center_stream.setdefault((r['center'],r['stream']),[]).append(r['deployed_service_mse'])
 return float(np.mean([np.mean(v) for v in center_stream.values()]))
def phase_summary(phase,rows):
 dest=ROOT/'results'/phase;dest.mkdir(parents=True,exist_ok=True)
 flat=[]
 for r in rows:
  q={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
  for role in ('parent','candidate','deployed'):q[role+'_state_hash']=r[role+'_entry']['state_hash'];q[role+'_file_sha256']=r[role+'_entry']['file_sha256']
  flat.append(q)
 csvwrite(dest/'STAGE_RECORDS.csv',flat)
 csvwrite(dest/'BOUNDED_RELEASE_AUDIT_ONLY.csv',[r for r in flat if r['reason']=='BOUNDED_A_RELEASE'])
 bycenter=[];bystream=[];byarm=[];byregime=[]
 arms=sorted(set(r['arm'] for r in rows));centers=sorted(set(r['center'] for r in rows))
 for arm in arms:
  rr=[r for r in rows if r['arm']==arm]
  for c in centers:
   cr=[r for r in rr if r['center']==c]
   bycenter.append(dict(phase=phase,arm=arm,method=rr[0]['method'],center=c,regime_audit_only=cr[0]['regime_audit_only'],balanced_clean_mixed_mse=balanced(cr),**aggregate(cr)))
  for regime in sorted(set(r['regime_audit_only'] for r in rr)):
   rg=[r for r in rr if r['regime_audit_only']==regime]
   byregime.append(dict(phase=phase,arm=arm,regime_audit_only=regime,independent_centers=len(set(r['center'] for r in rg)),balanced_clean_mixed_mse=balanced(rg),**aggregate(rg)))
  streams=[]
  for s in ('clean','mixed','coherent_sticky'):
   a=aggregate([r for r in rr if r['stream']==s]);streams.append(a);bystream.append(dict(phase=phase,arm=arm,method=rr[0]['method'],stream=s,**a))
  byarm.append(dict(phase=phase,arm=arm,method=rr[0]['method'],calibration_fallback=any(r.get('calibration_fallback',False) for r in rr),balanced_mse=balanced(rr),worst_harmful=max(a['harmful_frequency'] for a in streams),worst_downside=max(a['mean_downside'] for a in streams),balanced_commit=float(np.mean([r['commit'] for r in rr if r['stream'] in ('clean','mixed')])),feasible=all(a['harmful_frequency']<=.05 and a['mean_downside']<=LIMIT for a in streams),**aggregate(rr)))
 csvwrite(dest/'RESULTS_BY_CENTER.csv',bycenter);csvwrite(dest/'RESULTS_BY_STREAM.csv',bystream);csvwrite(dest/'RESULTS_BY_ARM.csv',byarm)
 csvwrite(dest/'RESULTS_BY_REGIME.csv',byregime)
 csvwrite(dest/'RESOURCE_COSTS.csv',[{k:r[k] for k in ('phase','center','seed','stream','arm','stage','maintenance_wall_seconds','training_seconds','late_score_seconds','post_seal_scientific_evaluation_seconds','gpu_peak_allocated_bytes')} for r in rows])
 if False:
  effects=[];rng=np.random.default_rng(20261019)
  for arm in arms:
   vals={r['center']:r['balanced_clean_mixed_mse'] for r in bycenter if r['arm']==arm}
   for baseline in arms:
    if arm!=read(ROOT/'DEVCAL_SELECTED_LOCK.json')['selected_new_method'] or baseline==arm:continue
    base={r['center']:r['balanced_clean_mixed_mse'] for r in bycenter if r['arm']==baseline}
    diff=np.array([vals[c]-base[c] for c in centers]);idx=rng.integers(len(diff),size=(4000,len(diff)));boot=diff[idx].mean(1)
    effects.append(dict(phase=phase,method=arm,baseline=baseline,n_centers=len(diff),paired_mse_delta=float(diff.mean()),lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),resampling='4000 paired center clusters; seeds/stages/streams stay clustered',scope='P0 exploratory only, no validation claim' if phase=='P0' else 'development confirmation, not final'))
  csvwrite(dest/'PAIRED_EFFECTS.csv',effects)
 return byarm,bystream
