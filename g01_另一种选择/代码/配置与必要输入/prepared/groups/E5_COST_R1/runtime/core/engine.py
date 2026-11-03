import copy,time,json,hashlib,os
import numpy as np
import torch
from core.protocol import ROOT,Timeline,Segment,windows,block_indices,attack,digest,save_json
from core.models import Forecast,DEVICE,adapt,robust_candidate,tensors,errors,state_hash,sync,fisher_diag,Batch

DEFAULT={'B':5,'beta':.2,'kappa_rec':1.,'kappa_hist':1.,'delta_imp':0.,'delta_ret':0.,'capacity':24,'validation_threshold':0.,'ewc_lambda':200.,'retain_all':False,'clear_factor':.75,'clear_capacity':128}

def evidence_blocks(seg,H=1,B=6):
    w=windows(seg,H=H);chunks=block_indices(len(w[0]),B,H);result=[]
    for ids in chunks:
        if len(ids):
            a=int(ids[0]);b=int(ids[-1])+96+H
            result.append(Segment(seg.x[a:b].copy(),seg.y[a:b].copy(),seg.ts[a:b].copy(),'verification'))
    return result

def _atomic_torch_save(obj,path):
    tmp=path.with_name(f'{path.name}.{os.getpid()}.tmp')
    torch.save(obj,tmp)
    try:
        if not path.exists():os.replace(tmp,path)
    except PermissionError:
        if not path.exists():raise
    finally:tmp.unlink(missing_ok=True)

def score(candidate,parent,blocks,H,kappa,mean=False,cache=False):
    vals=[];weights=[];calls=0;cache_path=None
    if cache:
        key=json.dumps([state_hash(candidate),state_hash(parent),H,[b.hash for b in blocks]],separators=(',',':'))
        cache_dir=ROOT/'checkpoints/calibration_evidence';cache_dir.mkdir(parents=True,exist_ok=True)
        cache_path=cache_dir/(hashlib.sha256(key.encode()).hexdigest()+'.npz')
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path,allow_pickle=False) as z:
            vals=z['vals'].astype(np.float64).tolist();weights=z['weights'].astype(np.int64).tolist();calls=int(z['calls'])
    else:
        for block in blocks:
            xy=tensors(windows(block,H=H));a,_=errors(candidate,xy);b,_=errors(parent,xy)
            vals.append(float((a-b).mean()));weights.append(len(a));calls+=2*((len(a)+255)//256)
        if cache_path is not None:
            tmp=cache_path.with_name(f'{cache_path.name}.{os.getpid()}.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f,vals=np.asarray(vals,dtype=np.float64),weights=np.asarray(weights,dtype=np.int64),calls=np.int64(calls))
            try:
                if not cache_path.exists():os.replace(tmp,cache_path)
            except PermissionError:
                if not cache_path.exists():raise
            finally:tmp.unlink(missing_ok=True)
    if not vals:return float('inf'),float('nan'),float('nan'),calls
    median=float(np.median(vals));mad=float(np.median(np.abs(np.array(vals)-median)))
    return (float(np.average(vals,weights=weights)) if mean else median+kappa*mad),median,mad,calls

def initial_model(center,seed,backbone,H,smoke=False):
    p=ROOT/'checkpoints'/f'{"smoke_" if smoke else ""}M0_{backbone}_H{H}_{seed}_{center}.pt'
    model=Forecast(backbone,H).to(DEVICE)
    if p.exists():model.load_state_dict(torch.load(p,weights_only=True,map_location=DEVICE));return model
    prefix='alibaba_' if Timeline(center).meta.get('domain')=='alibaba' else ''
    src=ROOT/'checkpoints'/f'{prefix}{"smoke_" if smoke else ""}source_{backbone}_H{H}_{seed}.pt'
    model.load_state_dict(torch.load(src,weights_only=True,map_location=DEVICE))
    warm=Timeline(center).warmup();model=adapt(model,tensors(windows(warm,H=H)),seed=seed+center)
    torch.save(model.state_dict(),p)
    save_json(p.with_suffix('.json'),{'initial_state_hash':state_hash(model),'source_checkpoint_hash':__import__('hashlib').sha256(src.read_bytes()).hexdigest(),'warmup_segment_manifest':{'first_ts':int(warm.ts[0]),'last_ts':int(warm.ts[-1]),'hash':warm.hash},'smoke':smoke})
    return model

def run(center,seed,method,kind='clean',alpha=0.,location='both',cfg=None,stages=24,backbone='GRU',H=1,smoke=False,run_id=''):
    cfg={**DEFAULT,**(cfg or {})};timeline=Timeline(center);parent=initial_model(center,seed,backbone,H,smoke)
    history=evidence_blocks(timeline.initial_history(),H);replay=[];rows=[];fisher=None;anchor=None
    novelty=[];familiarity=[];clear_threshold=None
    # EWC importance is estimated exclusively from warmup, never initial verification.
    if method in ['EWC','CLeaR-Predictor']:
        fisher=fisher_diag(parent,tensors(windows(timeline.warmup(),H=H)),seed)
        anchor={k:p.detach().clone() for k,p in parent.named_parameters()}
        if method=='CLeaR-Predictor':clear_threshold=cfg['clear_factor']*float(errors(parent,tensors(windows(timeline.warmup(),H=H)))[0].mean())
    for t in range(1,stages+1):
        raw_u,raw_v=timeline.maintenance(t)
        u,um,us,actual=attack(raw_u,center,t,kind,alpha,location)
        v,vm,vs,_=attack(raw_v,center,t,kind,alpha,location)
        before=state_hash(parent);sync()
        if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();xy=tensors(windows(u,H=H));extra_fwd=0;extra_backward=0
        cand_start=time.perf_counter()
        robust=method in ['Proposed','MeanDelta','RecentOnly','MeanVerification','RobustAlways']
        clear_trigger=False
        cache_path=None;cache_hit=False
        if cfg.get('calibration_candidate_cache') and method in ['Proposed','ValidationCommit']:
            # Only training-produced weights are cached here. No service labels/losses
            # are saved in, or loaded from, this pre-decision cache.
            key=json.dumps([before,u.hash,seed,center,t,method,backbone,H,cfg['B'] if robust else None,cfg['beta'] if robust else None])
            cache_dir=ROOT/'checkpoints/calibration_candidates';cache_dir.mkdir(exist_ok=True)
            cache_path=cache_dir/(hashlib.sha256(key.encode()).hexdigest()+'.pt')
        if cache_path is not None and cache_path.exists():
            candidate=copy.deepcopy(parent);candidate.load_state_dict(torch.load(cache_path,map_location=DEVICE,weights_only=True));updates=50;cache_hit=True
        elif method=='CLeaR-Predictor':
            ex,_=errors(parent,xy);mask=ex>clear_threshold;extra_fwd+=(len(ex)+255)//256
            for ids,buffer in [(np.flatnonzero(mask),novelty),(np.flatnonzero(~mask),familiarity)]:
                if len(ids):buffer.append((xy[0][ids].clone(),xy[1][ids].clone()))
            clear_trigger=sum(len(a) for a,b in novelty)>=cfg['clear_capacity']
            if clear_trigger:
                train_xy=Batch(torch.cat([a for a,b in novelty]),torch.cat([b for a,b in novelty]),'replay_training')
                candidate=adapt(parent,train_xy,seed=seed+center+t*100,ewc=(anchor,fisher),lam=cfg.get('clear_ewc_lambda',200.));updates=50
                clear_threshold=cfg['clear_factor']*float(errors(candidate,train_xy)[0].mean())
            else:candidate=copy.deepcopy(parent);updates=0
        elif method=='Frozen':candidate=copy.deepcopy(parent);updates=0
        elif robust:
            candidate=robust_candidate(parent,xy,cfg['B'],cfg['beta'],seed+center+t*100,method=='MeanDelta',H);updates=50
        else:
            train_xy=xy
            if method=='ER' and replay:
                old=[tensors(windows(s,H=H)) for s in replay]
                train_xy=Batch(torch.cat([xy[0]]+[o[0] for o in old]),torch.cat([xy[1]]+[o[1] for o in old]),'replay_training')
            candidate=adapt(parent,train_xy,seed=seed+center+t*100,loss='huber' if method=='Huber-FT' else 'mse',ewc=(anchor,fisher) if method=='EWC' else None,lam=cfg['ewc_lambda']);updates=50
        if cache_path is not None and not cache_hit:
            _atomic_torch_save(candidate.state_dict(),cache_path)
        sync();update_time=time.perf_counter()-cand_start
        assert state_hash(parent)==before,'candidate modified deployed parent'
        ver_start=time.perf_counter();recent=evidence_blocks(v,H)
        ur=uh=mr=mh=rr=rh=float('nan');vf=0
        if method in ['Proposed','SinglePath','MeanDelta','RecentOnly','MeanVerification']:
            ur,mr,rr,n=score(candidate,parent,recent,H,cfg['kappa_rec'],method=='MeanVerification',cfg.get('calibration_evidence_cache',False));vf+=n
            if method!='RecentOnly':uh,mh,rh,n=score(candidate,parent,history,H,cfg['kappa_hist'],method=='MeanVerification',cfg.get('calibration_evidence_cache',False));vf+=n
            commit=(ur<=-cfg['delta_imp'] and (method=='RecentOnly' or uh<=cfg['delta_ret'])) and not cfg['retain_all']
        elif method=='ValidationCommit':
            verxy=tensors(windows(v,H=H));a,_=errors(candidate,verxy);b,_=errors(parent,verxy)
            ur=float((a-b).mean());vf+=2*((len(a)+255)//256);commit=ur<=-cfg['validation_threshold'] and not cfg.get('validation_retain_all',False)
        else:commit=clear_trigger if method=='CLeaR-Predictor' else method!='Frozen'
        sync();verification_time=time.perf_counter()-ver_start
        # Persist the deployable decision before opening the evaluation-only capability.
        timeline.record_decision(t,commit)
        if method=='EWC' or (method=='CLeaR-Predictor' and clear_trigger):
            fxy=xy
            if method=='CLeaR-Predictor':
                both=novelty+familiarity;fxy=Batch(torch.cat([a for a,b in both]),torch.cat([b for a,b in both]),'replay_training')
                novelty=[];familiarity=[]
            newer=fisher_diag(candidate,fxy,seed+t)
            fisher={k:.9*fisher[k]+newer[k] for k in fisher};anchor={k:p.detach().clone() for k,p in candidate.named_parameters()}
            extra_backward=min(32,len(fxy[0]));extra_fwd+=extra_backward
        history=(history+recent)[-cfg['capacity']:]
        if method=='ER':
            replay.append(u.copy('update'))
            # Raw replay budget approximately equals actual proposed history raw storage.
            max_bytes=sum(b.x.nbytes+b.y.nbytes+b.ts.nbytes for b in history)
            while len(replay)>1 and sum(s.x.nbytes+s.y.nbytes+s.ts.nbytes for s in replay)>max_bytes:replay.pop(0)
        sync();maintenance_time=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated() if DEVICE=='cuda' else 0
        # No service tensors exist in adaptation or verification scopes.
        service=timeline.service(t);service_hash=service.hash;sxy=tensors(windows(service,H=H))
        cur,_=errors(parent,sxy);can,_=errors(candidate,sxy)
        if method=='OracleCommit':commit=float(can.mean())<float(cur.mean())
        deployed=candidate if commit else parent
        mse,mae=errors(deployed,sxy);delta=float(can.mean()-cur.mean())
        assert service.hash==service_hash
        if not commit:assert state_hash(deployed)==before
        buffers=replay if method=='ER' else history if method in ['Proposed','SinglePath','MeanDelta','MeanVerification','RecentOnly','RobustAlways'] else []
        memory=sum(s.x.nbytes+s.y.nbytes+s.ts.nbytes for s in buffers)
        if method in ['EWC','CLeaR-Predictor']:memory+=sum(a.numel()*a.element_size() for a in list(anchor.values())+list(fisher.values()))
        if method=='CLeaR-Predictor':memory+=sum(a.numel()*a.element_size()+b.numel()*b.element_size() for a,b in novelty+familiarity)
        rows.append(dict(run_id=run_id,center_id=center,center_type=timeline.meta['center_type'],model_seed=seed,horizon=H,backbone=backbone,stage=t,drift_regime=timeline.meta['drift_regime'],attack_type=kind,actual_attack_type=actual,attack_alpha=alpha,attack_location=location,method=method,parent_state_hash=before,candidate_state_hash=state_hash(candidate),initial_state_hash=state_hash(initial_model(center,seed,backbone,H,smoke)) if t==1 else rows[0]['initial_state_hash'],commit=int(commit),current_service_mse=float(cur.mean()),candidate_service_mse=float(can.mean()),deployed_service_mse=float(mse.mean()),deployed_service_mae=float(mae.mean()),delta_srv=delta,harmful_commit=int(commit and delta>0),downside=float(commit)*max(0,delta),beneficial_candidate=int(delta<0),U_rec=ur,U_hist=uh,median_rec=mr,mad_rec=rr,median_hist=mh,mad_hist=rh,candidate_update_time_s=update_time,verification_time_s=verification_time,maintenance_time_s=maintenance_time,peak_gpu_memory_bytes=peak,persistent_buffer_bytes=memory,number_of_forward_passes=updates+vf+extra_fwd,number_of_backward_updates=updates,number_of_fisher_backward_passes=extra_backward,update_start_ts=int(u.ts[0]),update_end_ts=int(u.ts[-1]),verification_start_ts=int(v.ts[0]),verification_end_ts=int(v.ts[-1]),decision_ts=int(v.ts[-1]),service_start_ts=int(service.ts[0]),service_end_ts=int(service.ts[-1]),attack_mask_hash=digest(um,vm,us,vs),update_data_hash=u.hash,verification_data_hash=v.hash,service_data_hash=service_hash))
        parent=deployed
    return rows
