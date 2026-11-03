"""Residential-building temperature-maintenance study on one observed home.

Original complete observations are from the author's UCI-linked repository.
The experiment is retrospective; the three rooms are not independent buildings.
Only maintenance-role copies receive controlled monitoring perturbations.
"""
from __future__ import annotations
import argparse, copy, datetime, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from core.models import Forecast, Batch, DEVICE, adapt, errors, state_hash, tensors
from core.protocol import Segment, windows
from core.engine import evidence_blocks, score

BASE=Path(__file__).resolve().parent.parent
CSV=BASE/'data/energydata_complete.csv'
OUT=BASE/'results/building'
CHECK=OUT/'checkpoints'
CHECK.mkdir(parents=True,exist_ok=True)
SEEDS=[7311,7312,7313]
ROOMS=['T1','T2','T3']
CONDITIONS=['clean','bias','dropout','stale']
SCALES=[0.,.001,.005,.01,.05]
RHO=.05
L=96
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data():
    d=pd.read_csv(CSV)
    assert len(d)==19735 and not d.isna().any().any()
    ts=pd.to_datetime(d.date)
    assert not ts.duplicated().any() and (ts.diff().dropna()==pd.Timedelta(minutes=10)).all()
    cols=[item for n in range(1,10) for item in (f'T{n}',f'RH_{n}')]
    values=d[cols].to_numpy(dtype='float32')
    theta=2*np.pi*(ts.dt.hour.to_numpy()+ts.dt.minute.to_numpy()/60)/24
    values=np.column_stack([values,np.sin(theta),np.cos(theta)]).astype('float32')
    mean=values[:4000].mean(0);std=values[:4000].std(0)
    assert np.all(std>1e-6)
    x=(values-mean)/std
    x=np.concatenate([x,np.ones_like(x)],axis=1).astype('float32')
    t=np.arange(len(d),dtype='int64')
    return x,t,cols,mean,std,ts

X,TS,COLS,MEAN,STD,DATES=load_data()
DATA_HASH=sha(CSV)

def seg(start,end,room,role):
    assert 0<=start<end<=len(X)
    return Segment(X[start:end].copy(),X[start:end,ROOMS.index(room)*2].copy(),TS[start:end].copy(),role)

def xy(segment):return tensors(windows(segment,L=L,H=1))

def pretrain(room,seed):
    torch.manual_seed(seed+ROOMS.index(room)*1000)
    model=Forecast('GRU',H=1,features=X.shape[1]).to(DEVICE)
    model=adapt(model,xy(seg(0,3000,room,'warmup')),steps=200,seed=seed+ROOMS.index(room)*1000)
    evalerror,_=errors(model,xy(seg(3500,4000,room,'service')))
    p=CHECK/f'source_{room}_{seed}.pt'
    torch.save(model.state_dict(),p)
    return {'room':room,'seed':seed,'state_hash':state_hash(model),'checkpoint_hash':sha(p),'development_reference_mse':float(evalerror.mean())}

def initial_model(pool,room,seed):
    lo=4000 if pool=='calibration' else 9000
    model=Forecast('GRU',H=1,features=X.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(CHECK/f'source_{room}_{seed}.pt',map_location=DEVICE,weights_only=True))
    return adapt(model,xy(seg(lo,lo+300,room,'warmup')),steps=50,seed=seed+ROOMS.index(room)*1000+lo)

def attack(segment,room,condition,stage):
    assert segment.role in {'update','verification'}
    s=segment.copy()
    if condition=='clean':return s,0
    idx=ROOMS.index(room)*2
    rng=np.random.default_rng(np.random.SeedSequence([9487,ROOMS.index(room),stage,0 if s.role=='update' else 1]))
    n=len(s.y);picked=np.zeros(n,dtype=bool)
    for stratum in np.array_split(np.arange(n),10):
        k=int(np.floor(len(stratum)*.10))
        if k:picked[rng.choice(stratum,size=k,replace=False)]=True
    if condition=='bias':
        # The perturbation is a stress-test setting, not an observed fault rate.
        s.x[picked,idx]+=np.float32(2.0/STD[idx]);s.y[picked]+=np.float32(2.0/STD[idx])
    elif condition=='dropout':
        # Input-channel loss only. Labels remain evaluation references.
        s.x[picked,idx]=0;s.x[picked,X.shape[1]//2+idx]=0
    elif condition=='stale':
        ids=np.flatnonzero(picked);previous=np.maximum(ids-6,0)
        s.x[ids,idx]=s.x[previous,idx];s.y[ids]=s.y[previous]
    else:raise ValueError(condition)
    assert picked.sum()<=np.floor(n*.1+1e-8)
    return s,int(picked.sum())

def evaluate(pool,room,seed,condition,method,threshold=0.,stages=10):
    assert pool in {'calibration','final'}
    lo=4000 if pool=='calibration' else 9000
    assert stages==10 if pool=='calibration' else stages==24
    parent=initial_model(pool,room,seed)
    initial_hash=state_hash(parent)
    history=evidence_blocks(seg(lo+300,lo+500,room,'verification'))
    rows=[]
    for stage in range(1,stages+1):
        v0=lo+700+(stage-1)*360
        u_start=lo+500 if stage==1 else v0-200
        update_raw=seg(u_start,v0,room,'update')
        ver_raw=seg(v0,v0+160,room,'verification')
        update,n_update=attack(update_raw,room,condition,stage)
        verification,n_ver=attack(ver_raw,room,condition,stage)
        before=state_hash(parent)
        candidate=copy.deepcopy(parent) if method=='Frozen' else adapt(parent,xy(update),steps=50,seed=seed+ROOMS.index(room)+stage*100)
        assert state_hash(parent)==before
        if method=='RAVM-DT':
            recent=evidence_blocks(verification)
            urec,_,_,_=score(candidate,parent,recent,1,0)
            uhist,_,_,_=score(candidate,parent,history,1,.5)
            commit=bool(urec<=-threshold and uhist<=0)
        elif method=='Val-Mean':
            a,_=errors(candidate,xy(verification));b,_=errors(parent,xy(verification))
            urec=float((a-b).mean());uhist=float('nan');commit=bool(urec<=-threshold)
            recent=evidence_blocks(verification)
        elif method=='Always-FT':
            urec=uhist=float('nan');commit=True;recent=evidence_blocks(verification)
        elif method=='Frozen':
            urec=uhist=float('nan');commit=False;recent=evidence_blocks(verification)
        else:raise ValueError(method)
        # Seal the decision before the future service interval is constructed.
        decision={'commit':int(commit),'stage':stage,'decision_ts':int(ver_raw.ts[-1]),'parent_hash':before,'candidate_hash':state_hash(candidate)}
        history=(history+recent)[-12:]
        service=seg(v0+160,v0+360,room,'service')
        assert decision['decision_ts']<int(service.ts[0])
        cur,_=errors(parent,xy(service));can,_=errors(candidate,xy(service))
        current=float(cur.mean());candidate_loss=float(can.mean());delta=float((can.mean()-cur.mean()))
        rows.append({'pool':pool,'room':room,'seed':seed,'condition':condition,'method':method,'threshold':threshold,'stage':stage,'commit':int(commit),'parent_state_hash':before,'candidate_state_hash':decision['candidate_hash'],'initial_state_hash':initial_hash,'decision_ts':decision['decision_ts'],'service_start_ts':int(service.ts[0]),'current_service_mse':current,'candidate_service_mse':candidate_loss,'deployed_service_mse':candidate_loss if commit else current,'delta_srv':delta,'harmful_commit':int(commit and delta>0),'downside':max(delta,0)*int(commit),'beneficial_candidate':int(delta<0),'U_rec':urec,'U_hist':uhist,'n_corrupted_update':n_update,'n_corrupted_verification':n_ver})
        parent=candidate if commit else parent
    return rows

def write_rows(path,rows):
    d=pd.DataFrame(rows);temp=path.with_suffix(f'.{os.getpid()}.tmp');d.to_csv(temp,index=False);temp.replace(path)

def calibration_table():
    allrows=[];paths=list((OUT/'calibration').glob('*.csv'))
    assert len(paths)==2*5*3*3*4
    for p in paths:allrows.append(pd.read_csv(p))
    d=pd.concat(allrows,ignore_index=True)
    rows=[]
    cfg=json.loads((BASE/'audit/building_protocol_locked.json').read_text())
    for (method,threshold),g in d.groupby(['method','threshold']):
        by=g.groupby('condition')[['harmful_commit','downside','deployed_service_mse','commit']].mean()
        rows.append({'method':method,'threshold':threshold,'n':len(g),'balanced_mse':float(by.deployed_service_mse.mean()),'worst_harmful':float(by.harmful_commit.max()),'worst_downside':float(by.downside.max()),'balanced_commit':float(by.commit.mean()),'feasible':bool((by.harmful_commit<=.05).all() and (by.downside<=cfg['epsilon']).all())})
    table=pd.DataFrame(rows).sort_values(['method','threshold'])
    table.to_csv(OUT/'calibration_grid.csv',index=False)
    chosen={}
    for method,g in table.groupby('method'):
        eligible=g[g.feasible]
        if len(eligible):
            row=eligible.sort_values(['balanced_mse','worst_downside','threshold']).iloc[0]
            chosen[method]={'threshold':float(row.threshold),'fallback':False,'calibration_balanced_mse':float(row.balanced_mse)}
        else:chosen[method]={'threshold':None,'fallback':True}
    result={'status':'CALIBRATION_COMPLETE','selected':chosen,'grid_sha256':sha(OUT/'calibration_grid.csv'),'stages':int(len(d))}
    (OUT/'selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def run_all(pool):
    assert pool in {'calibration','final'}
    cfg=json.loads((BASE/'audit/building_protocol_locked.json').read_text())
    if pool=='final':selection=json.loads((OUT/'selection.json').read_text())['selected']
    dest=OUT/pool;dest.mkdir(exist_ok=True)
    methods=['RAVM-DT','Val-Mean'] if pool=='calibration' else ['RAVM-DT','Val-Mean','Always-FT','Frozen']
    count=0;start=time.time()
    for room in ROOMS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                for method in methods:
                    thresholds=[v*cfg['development_reference_mse'] for v in SCALES] if pool=='calibration' else [selection[method]['threshold'] if method in selection else 0.]
                    for threshold in thresholds:
                        fallback=pool=='final' and method in selection and selection[method]['fallback']
                        path=dest/f'{method}_{room}_{seed}_{condition}_{"fallback" if fallback else f"{threshold:.12g}"}.csv'
                        if not path.exists():
                            if fallback:
                                rows=evaluate(pool,room,seed,condition,'Frozen',0.,stages=24)
                                for row in rows:row.update(method=method,threshold=float('nan'))
                            else:rows=evaluate(pool,room,seed,condition,method,threshold,stages=10 if pool=='calibration' else 24)
                            write_rows(path,rows)
                        count+=1
                        if count%10==0:
                            (BASE/'logs/building_progress.json').write_text(json.dumps({'pool':pool,'completed':count,'elapsed_s':time.time()-start,'last':str(path),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),encoding='utf-8')
                            print(pool,count,round(time.time()-start,1),flush=True)
    if pool=='calibration':print(json.dumps(calibration_table()),flush=True)
    else:
        allrows=pd.concat([pd.read_csv(p) for p in dest.glob('*.csv')],ignore_index=True)
        result=allrows.groupby(['method','condition']).agg(mse=('deployed_service_mse','mean'),harmful=('harmful_commit','mean'),downside=('downside','mean'),commit=('commit','mean'),n=('stage','size')).reset_index()
        result.to_csv(OUT/'final_summary.csv',index=False)
        print(result.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','calibration','final']);args=ap.parse_args()
    if args.phase=='prepare':
        refs=[pretrain(room,seed) for room in ROOMS for seed in SEEDS]
        R=float(np.mean([x['development_reference_mse'] for x in refs]))
        assert R>0
        protocol={'status':'LOCKED_BEFORE_CALIBRATION','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data_hash':DATA_HASH,'n_observations':len(X),'single_building':True,'rooms':ROOMS,'seeds':SEEDS,'source_and_development':[0,4000],'source_training':[0,3000],'development_reference':[3500,4000],'calibration':[4000,9000],'final':[9000,len(X)],'target_warmup':300,'initial_history':200,'update':200,'verification':160,'service':200,'stage_stride':360,'calibration_stages':10,'final_stages':24,'lookback':L,'horizon':1,'fault_conditions':CONDITIONS,'fault_fraction':.1,'bias_celsius':2.,'stale_lag_observations':6,'scales':SCALES,'kappa_recent':0.,'kappa_history':.5,'history_capacity':12,'delta_ret':0.,'rho':RHO,'development_reference_mse':R,'epsilon':.01*R,'delta_imp_selection':'same 5 threshold scales for both methods','uncertainty':'one real home; rooms and seeds are not independent physical units','code_sha256':sha(__file__),'source_checkpoints':refs}
        (BASE/'audit/building_protocol_locked.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        print('LOCKED',R,flush=True)
    else:
        protocol=json.loads((BASE/'audit/building_protocol_locked.json').read_text())
        assert protocol['data_hash']==DATA_HASH and protocol['code_sha256']==sha(__file__)
        run_all(args.phase)
