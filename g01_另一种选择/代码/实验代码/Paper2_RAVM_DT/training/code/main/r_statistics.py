"""Center-unit CSV statistics, independent of the old analysis implementation."""
from d3_common import *
import csv
LIMIT=5.6036475532487204e-5
JOINT="JOINT__l3__m0"
STREAMS=("clean","mixed","coherent_sticky")
def csvwrite(p,rows):
 from d3_results import csvwrite as write
 return write(p,rows)
import numpy as np,collections,math
def load_csv(p):
 rr=list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
 for r in rr:
  for k in ('center','seed','stage'):r[k]=int(r[k])
  for k in ('commit','harmful','beneficial_commit','candidate_beneficial_audit_only','fallback','feedback_active'):r[k]=r[k]=='True'
  for k in ('deployed_service_mse','parent_service_mse','candidate_service_mse','downside','maintenance_wall_seconds','training_seconds','late_score_seconds','post_seal_scientific_evaluation_seconds'):r[k]=float(r[k])
 return rr
def metric(rr):
 n=len(rr);assert n
 for r in rr:
  d=r['candidate_service_mse']-r['parent_service_mse'];assert r['harmful']==bool(r['commit'] and d>0);assert r['downside']==(max(d,0.) if r['commit'] else 0.)
 return dict(opportunities=n,harmful_count=sum(r['harmful'] for r in rr),harmful_frequency=sum(r['harmful'] for r in rr)/n,downside=float(np.mean([r['downside'] for r in rr])),commits=sum(r['commit'] for r in rr),commit_rate=sum(r['commit'] for r in rr)/n,service_mse=float(np.mean([r['deployed_service_mse'] for r in rr])),maintenance_seconds=sum(r['maintenance_wall_seconds'] for r in rr),beneficial_commits=sum(r['beneficial_commit'] for r in rr))
def centers(rr,arm):
 groups=collections.defaultdict(list)
 for r in rr:
  if r['arm']==arm and r['stream'] in ('clean','mixed'):groups[r['center'],r['stream']].append(r['deployed_service_mse'])
 ids=sorted({c for c,s in groups});return {c:float(np.mean([np.mean(groups[c,s]) for s in ('clean','mixed')])) for c in ids}
def compare(rr,candidate,baseline):
 a=centers(rr,candidate);b=centers(rr,baseline);ids=sorted(a);assert ids==sorted(b)
 ds=np.array([a[c]-b[c] for c in ids]);idx=np.random.default_rng(20261020).integers(len(ids),size=(4000,len(ids)));boot=ds[idx].mean(axis=1)
 return dict(candidate=candidate,baseline=baseline,centers=len(ids),candidate_mse=float(np.mean(list(a.values()))),baseline_mse=float(np.mean(list(b.values()))),mse_delta=float(ds.mean()),relative_improvement=float(1-np.mean(list(a.values()))/np.mean(list(b.values()))),lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),center_wins=int((ds<0).sum()),center_ties=int((ds==0).sum()),bootstrap_draws=4000,bootstrap_seed=20261020,unit='center, all seeds/streams/stages clustered')
def summary(rr):
 rows=[]
 for a in sorted({r['arm'] for r in rr}):
  for s in STREAMS:
   m=metric([r for r in rr if r['arm']==a and r['stream']==s]);m.update(arm=a,stream=s,allowed_harm_count=math.floor(m['opportunities']/20),risk_qualified=m['harmful_frequency']<=.05 and m['downside']<=LIMIT);rows.append(m)
 return rows
