"""Decision-equivalent shared-parent execution; no service access before sealing.

Provider contract: initial(), maintenance(stage), service(stage, decisions),
xy(segment), initial_history(), stages, seed_for_adaptation(stage), metadata.
All gate arms maintain separate deployed chains. Only equal-state computations
within one stage are shared; no arm consumes another arm's outcome/decisions.
"""
import copy,hashlib,json,time
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from core.models import adapt,errors,state_hash,Batch,DEVICE
from core.engine import evidence_blocks

METHODS=['RecentMean','RecentMedian','DualMean','RAVM','CumulativeMean','MWUMean']
SCALES=[0.,.001,.002,.003,.005,.0075,.01,.015,.02,.03,.04,.05,.075,.1,.15,.2]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def arms(reference):
    return [dict(id=f'{m}__g{i:02}',method=m,threshold=s*reference,scale=s,fallback=False) for m in METHODS for i,s in enumerate(SCALES)]

def decision(method,scores,threshold,fallback=False):
    if fallback or method=='Frozen':return False
    if method=='AlwaysFT':return True
    if not np.isfinite(threshold) or threshold<0 or not np.isfinite(list(scores.values())).all():
        raise FloatingPointError('nonfinite/invalid gate evidence; never permit fail-open commitment')
    recent={'RecentMean':scores['recent_mean'],'RecentMedian':scores['recent_median'],
            'DualMean':scores['recent_mean'],'RAVM':scores['recent_median'],
            'CumulativeMean':scores['cumulative_mean'],'MWUMean':scores['recent_mean']}[method]
    if recent > -threshold:return False
    if method=='DualMean':return scores['history_mean']<=0
    if method=='RAVM':return scores['history_robust']<=0
    if method=='MWUMean':return scores['mwu_p']<.2
    # Boundary <= is common to all mean-margin gates in this controlled study.
    return True

def score_models(parent,candidate,blocks,n_recent,xy):
    batches=[xy(b) for b in blocks]
    counts=[len(b[0]) for b in batches]
    if not all(counts):raise ValueError('empty verification block after filtering')
    batch=Batch(torch.cat([b[0] for b in batches]),torch.cat([b[1] for b in batches]),'verification')
    old,_=errors(parent,batch);new,_=errors(candidate,batch)
    ds=new-old;offset=np.cumsum([0]+counts)
    blockmeans=np.array([float(ds[offset[i]:offset[i+1]].mean()) for i in range(len(counts))])
    start=offset[len(blocks)-n_recent]
    recent=ds[start:];historical=ds[:start]
    histmeans=blockmeans[max(0,len(blocks)-n_recent-12):len(blocks)-n_recent]
    hstart=offset[max(0,len(blocks)-n_recent-12)]
    histmedian=float(np.median(histmeans));histmad=float(np.median(abs(histmeans-histmedian)))
    return dict(recent_mean=float(recent.mean()),recent_median=float(np.median(blockmeans[-n_recent:])),
                history_mean=float(ds[hstart:start].mean()),history_robust=histmedian+.5*histmad,
                cumulative_mean=float(ds.mean()),mwu_p=float(mannwhitneyu(new[start:],old[start:],alternative='less',method='asymptotic').pvalue))

def run_graph(provider,policies,independent=False):
    started=time.time();initial=provider.initial();initial_hash=state_hash(initial)
    parents={a['id']:initial for a in policies}
    full=evidence_blocks(provider.initial_history())
    rows=[];training_nodes=0;naive_nodes=0
    for stage in range(1,provider.stages+1):
        update,verification,attack_metadata=provider.maintenance(stage)
        update_xy=provider.xy(update);recent=evidence_blocks(verification)
        verification_hash=verification.hash;update_hash=update.hash
        blocks=full+recent;node_cache={};chosen={};pending=[]
        for a in policies:
            parent=parents[a['id']];parent_hash=state_hash(parent)
            train_needed=a['method']!='Frozen' and not a.get('fallback',False)
            key=(parent_hash,a['id'] if independent else '',train_needed)
            naive_nodes+=int(train_needed)
            if key not in node_cache:
                if train_needed:
                    candidate=adapt(parent,update_xy,steps=50,seed=provider.seed_for_adaptation(stage));training_nodes+=1
                    scores=score_models(parent,candidate,blocks,len(recent),provider.xy)
                else:
                    candidate=parent;scores={k:0. for k in ['recent_mean','recent_median','history_mean','history_robust','cumulative_mean','mwu_p']}
                assert state_hash(parent)==parent_hash
                node_cache[key]=(candidate,scores,state_hash(candidate))
            candidate,scores,candidate_hash=node_cache[key]
            commit=decision(a['method'],scores,a['threshold'],a.get('fallback',False))
            record=dict(arm=a['id'],method=a['method'],threshold=a['threshold'],scale=a['scale'],fallback=a.get('fallback',False),
                        stage=stage,commit=int(commit),parent_state_hash=parent_hash,candidate_state_hash=candidate_hash,
                        initial_state_hash=initial_hash,update_data_hash=update_hash,verification_data_hash=verification_hash,
                        decision_ts=int(verification.ts[-1]),**scores,**attack_metadata,**provider.metadata)
            chosen[a['id']]=int(commit);pending.append((a,parent,candidate,record))
        # Seal ALL decisions and their inputs before constructing service data.
        sealed=digest_obj([dict(arm=r['arm'],commit=r['commit'],parent=r['parent_state_hash'],candidate=r['candidate_state_hash']) for _,_,_,r in pending])
        service=provider.service(stage,chosen)
        assert all(r['decision_ts']<int(service.ts[0]) for _,_,_,r in pending)
        service_xy=provider.xy(service);loss_cache={}
        for a,parent,candidate,r in pending:
            for model,h in [(parent,r['parent_state_hash']),(candidate,r['candidate_state_hash'])]:
                if h not in loss_cache:loss_cache[h]=float(errors(model,service_xy)[0].mean())
            old=loss_cache[r['parent_state_hash']];new=loss_cache[r['candidate_state_hash']];delta=new-old
            r.update(current_service_mse=old,candidate_service_mse=new,delta_srv=delta,
                     deployed_service_mse=new if r['commit'] else old,harmful_commit=int(r['commit'] and delta>0),
                     downside=max(delta,0)*r['commit'],beneficial_candidate=int(delta<0),
                     service_data_hash=service.hash,service_start_ts=int(service.ts[0]),decision_seal=sealed)
            rows.append(r);parents[a['id']]=candidate if r['commit'] else parent
        full=blocks
    return pd.DataFrame(rows),dict(training_nodes=training_nodes,naive_training_nodes=naive_nodes,seconds=time.time()-started)
