"""Prospective ThermBuild gate study, isolated from all previous experiments.

Commands are explicit: inventory reads schema/coverage only; prepare trains source
checkpoints and a development Frozen reference; calibration evaluates the locked
grid; final evaluates only calibration-selected arms. Nothing runs on import.
Do not run prepare/calibration/final until the proposed protocol is reviewed.

TIME in delivered measured files is an ordinal, despite the generic publication
table saying hours. main.TIME + 1 == extra-temperature.TIME is the audited join.
Only raw measured data is used; no global KNN-imputed data is permitted.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from core.models import Forecast, Batch, DEVICE, adapt, state_hash
from core.protocol import Segment, windows
from core.engine import evidence_blocks
from gate_graph import METHODS, SCALES, arms, run_graph, sha

BASE=Path(__file__).resolve().parent.parent
RAW=BASE/'research'/'thermbuild'/'raw'
OUT=BASE/'results'/'thermal'
AUDIT=BASE/'audit'
CHECK=OUT/'checkpoints'
SEEDS=[7311,7312,7313]
BUILDINGS=['BSE1','BSE2']
CONDITIONS=['unmodified','offset_2C_1h','dropout_1h']
TARGET='room_5_living_110_AT'
L=96;WARMUP=800;INITIAL_HISTORY=200
UPDATE=192;VERIFICATION=192;SERVICE=192;STRIDE=384
MIN_ORIGINS=48;MIN_BLOCK_ORIGINS=8
RHO=.05;EPSILON_SCALE=.01
INPUT_NAMES=[TARGET,'wea_Tair_out','wea_IbeamHor','wea_IdiffHor',
             'room_5_living_ihs','hp_elP','hp_rps','dist_Tflow_store_ufh',
             'dist_Vol','mode_heat','mode_cool']
SOURCE_LAST_DAY=120
POOLS={'development':(121,180),'calibration':(181,270),'final':(271,444)}
LOCK=AUDIT/'thermal_protocol_locked.json'

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def plain_xy(segment):
    w=windows(segment,L=L,H=1)
    valid=np.isfinite(w[1]).all(axis=1)
    return Batch(torch.as_tensor(w[0][valid],device=DEVICE),
                 torch.as_tensor(w[1][valid],device=DEVICE),segment.role)

def valid_origins(segment):
    return int(np.isfinite(segment.y[L:]).sum())

def causal_fill(values,mean,std,max_lag=4):
    """Carry an observed value for <= four past intervals; never borrow future."""
    values=np.asarray(values,dtype=np.float64)
    observed=np.isfinite(values)
    z=np.zeros_like(values)
    for j in range(values.shape[1]):
        last=None;age=max_lag+1
        for i in range(len(values)):
            if observed[i,j]:last=values[i,j];age=0
            else:age+=1
            if last is not None and age<=max_lag:
                z[i,j]=(last-mean[j])/std[j]
    return np.concatenate([z,observed.astype(np.float64)],axis=1).astype('float32')

def load_building(building):
    paths=list(RAW.glob(f'*_{building}_*_raw.csv'))
    temperature_paths=list(RAW.glob(f'*_{building}_*_roomTemp.csv'))
    if len(paths)!=1 or len(temperature_paths)!=1:
        raise FileNotFoundError(f'exactly one raw and temperature CSV required for {building}')
    main=pd.read_csv(paths[0]);temperature=pd.read_csv(temperature_paths[0])
    main.columns=[c.strip() for c in main.columns]
    temperature.columns=[c.strip() for c in temperature.columns]
    assert np.array_equal(main.TIME.to_numpy(),np.arange(len(main)))
    assert np.array_equal(temperature.TIME.to_numpy(),np.arange(1,len(main)+1))
    main=main.copy();main[TARGET]=temperature[TARGET].to_numpy()
    assert main.consecutive_days.is_monotonic_increasing
    days=main.consecutive_days.to_numpy(dtype=np.int64)
    in_source=days<=SOURCE_LAST_DAY
    physical=main[INPUT_NAMES].to_numpy(dtype=np.float64)
    # Source-only normalization is a strict subset of the permitted development
    # information and never uses the later development reference/calibration/final.
    mean=np.nanmean(physical[in_source],axis=0)
    std=np.nanstd(physical[in_source],axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('a prespecified source input has no observed values')
    std=np.where(std>1e-6,std,1.)
    x=causal_fill(physical,mean,std,max_lag=4)
    # Infer within-day phase from the documented day groups and 15-min cadence.
    # Only the first day is partial at the beginning. No UTC claim is made.
    counts=main.groupby('consecutive_days').size()
    middle=counts.iloc[1:-1]
    if not (middle==96).all():
        raise ValueError('calendar phase reconstruction requires complete 96-row interior days')
    quarter=main.groupby('consecutive_days').cumcount().to_numpy().copy()
    quarter[days==days[0]]+=96-int(counts.iloc[0])
    phase=2*np.pi*quarter/96.
    annual=2*np.pi*(main.day_of_the_year.to_numpy()-1)/365.25
    x=np.column_stack([x,np.sin(phase),np.cos(phase),np.sin(annual),np.cos(annual)]).astype('float32')
    target=main[TARGET].to_numpy(dtype=np.float64)
    y=((target-mean[0])/std[0]).astype('float32')
    assert np.isfinite(x).all()
    assert np.all(np.isfinite(y)==np.isfinite(target))
    return dict(building=building,x=x,y=y,days=days,ts=np.arange(len(main),dtype=np.int64),
                target_scale=float(std[0]),target_mean=float(mean[0]),
                feature_names=INPUT_NAMES+[f'observed_{n}' for n in INPUT_NAMES]+['day_sin','day_cos','annual_sin','annual_cos'],
                feature_mean=mean.tolist(),feature_std=std.tolist(),
                files=[dict(path=p.relative_to(BASE).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in [paths[0],temperature_paths[0]]])

def pool_bounds(data,pool):
    a,b=POOLS[pool]
    ids=np.flatnonzero((data['days']>=a)&(data['days']<=b))
    if not len(ids):raise ValueError(f'no rows for {pool}')
    assert np.array_equal(ids,np.arange(ids[0],ids[-1]+1))
    return int(ids[0]),int(ids[-1]+1)

def segment(data,a,b,role):
    assert 0<=a<b<=len(data['y'])
    return Segment(data['x'][a:b].copy(),data['y'][a:b].copy(),data['ts'][a:b].copy(),role)

def stage_spans(lo,raw_stage):
    v=lo+WARMUP+INITIAL_HISTORY+UPDATE+(raw_stage-1)*STRIDE
    return (v-UPDATE,v),(v,v+VERIFICATION),(v+VERIFICATION,v+VERIFICATION+SERVICE)

def coverage(data,pool):
    lo,hi=pool_bounds(data,pool)
    history=segment(data,lo+WARMUP,lo+WARMUP+INITIAL_HISTORY,'verification')
    initial_counts=[valid_origins(b) for b in evidence_blocks(history)]
    initial_ok=len(initial_counts)==6 and min(initial_counts)>=MIN_BLOCK_ORIGINS
    warm=segment(data,lo,lo+WARMUP,'warmup')
    warm_count=valid_origins(warm)
    possible=max(0,(hi-lo-WARMUP-INITIAL_HISTORY-UPDATE)//STRIDE)
    kept=[];excluded=[]
    for raw_stage in range(1,possible+1):
        usp,vsp,ssp=stage_spans(lo,raw_stage)
        u=segment(data,*usp,'update');v=segment(data,*vsp,'verification');s=segment(data,*ssp,'service')
        uc=valid_origins(u);vc=[valid_origins(b) for b in evidence_blocks(v)];sc=valid_origins(s)
        reasons=[]
        if uc<MIN_ORIGINS:reasons.append('update_fewer_than_48_observed_target_origins')
        if len(vc)!=6 or min(vc)<MIN_BLOCK_ORIGINS:reasons.append('verification_block_fewer_than_8_observed_target_origins')
        if sc<MIN_ORIGINS:reasons.append('service_fewer_than_48_observed_target_origins')
        record=dict(raw_stage=raw_stage,update_observed_origins=uc,verification_block_observed_origins=vc,
                    service_observed_origins=sc,spans=[list(usp),list(vsp),list(ssp)])
        if reasons:excluded.append(dict(**record,reasons=reasons))
        else:kept.append(record)
    return dict(building=data['building'],pool=pool,row_bounds=[lo,hi],possible_stages=possible,
                retained_stages=len(kept),retained=kept,excluded=excluded,
                initial_history_block_observed_origins=initial_counts,initial_history_eligible=initial_ok,
                warmup_observed_origins=warm_count,pool_eligible=bool(initial_ok and warm_count>=MIN_ORIGINS and len(kept)>0))

class ThermalProvider:
    def __init__(self,data,pool,seed,condition,strict=True):
        assert condition in CONDITIONS
        self.data=data;self.pool=pool;self.seed=seed;self.condition=condition
        self.coverage=coverage(data,pool)
        if strict and not self.coverage['pool_eligible']:
            raise ValueError(f'fixed pool initial evidence/coverage ineligible: {data["building"]}/{pool}')
        self.lo,self.hi=self.coverage['row_bounds']
        self.raw_stages=[r['raw_stage'] for r in self.coverage['retained']]
        self.stages=len(self.raw_stages);self.sealed={}
        self.metadata=dict(building=data['building'],pool=pool,model_seed=seed,condition=condition,
                           target=TARGET,target_scale_celsius=data['target_scale'],target_mean_celsius=data['target_mean'])
    def initial(self):
        model=Forecast('GRU',H=1,features=self.data['x'].shape[1]).to(DEVICE)
        p=CHECK/f'source_{self.data["building"]}_{self.seed}.pt'
        model.load_state_dict(torch.load(p,map_location=DEVICE,weights_only=True))
        return adapt(model,self.xy(segment(self.data,self.lo,self.lo+WARMUP,'warmup')),
                     steps=50,seed=self.seed_for_adaptation(0))
    def initial_history(self):
        return segment(self.data,self.lo+WARMUP,self.lo+WARMUP+INITIAL_HISTORY,'verification')
    def xy(self,s):return plain_xy(s)
    def seed_for_adaptation(self,stage):
        raw=0 if stage==0 else self.raw_stages[stage-1]
        return self.seed+(1 if self.data['building']=='BSE1' else 2)*100000+self.lo+raw*100
    def _fault(self,s,raw_stage):
        assert s.role in {'update','verification'}
        out=s.copy();picked=np.zeros(len(out.y),dtype=bool)
        if self.condition!='unmodified':
            rng=np.random.default_rng(np.random.SeedSequence([27013,1 if self.data['building']=='BSE1' else 2,
                  self.lo,raw_stage,0 if s.role=='update' else 1]))
            a=int(rng.integers(0,len(out.y)-4+1));picked[a:a+4]=True
            if self.condition=='offset_2C_1h':
                out.x[picked,0]+=np.float32(2./self.data['target_scale'])
                out.y[picked]+=np.float32(2./self.data['target_scale'])
            elif self.condition=='dropout_1h':
                out.x[picked,0]=0.
                out.x[picked,len(INPUT_NAMES)]=0.
            else:raise ValueError(self.condition)
        return out,int(picked.sum())
    def maintenance(self,stage):
        raw=self.raw_stages[stage-1]
        us,vs,_=stage_spans(self.lo,raw)
        u,nu=self._fault(segment(self.data,*us,'update'),raw)
        v,nv=self._fault(segment(self.data,*vs,'verification'),raw)
        return u,v,dict(raw_stage=raw,n_fault_update_rows=nu,n_fault_verification_rows=nv,
                        fault_episode_samples=0 if self.condition=='unmodified' else 4)
    def service(self,stage,decisions):
        if not isinstance(decisions,dict) or not decisions or stage in self.sealed:
            raise AssertionError('all arm decisions must be sealed once before service access')
        self.sealed[stage]=dict(decisions)
        _,_,ss=stage_spans(self.lo,self.raw_stages[stage-1])
        return segment(self.data,*ss,'service')

def proposed_protocol():
    return dict(schema='ravm-thermal-round2-proposed-v1',status='PROPOSED_NOT_YET_TRAINED',
        buildings=BUILDINGS,physical_building_count=2,one_shared_laboratory_site=True,seeds=SEEDS,
        dataset_doi='10.24406/fordatis/445',license='CC BY-SA 4.0',
        target=TARGET,target_unit='Celsius',target_main_file_height_inconsistent=True,
        source_days=[1,120],development_reference_days=[121,180],calibration_days=[181,270],final_days=[271,444],
        normalization='each building source days1-120 only; observed-value mean/population SD; constant-feature SD=1',
        missing_inputs='causal past carry maximum4 samples then standardized0; observation masks retained',
        missing_targets='never imputed; windows with nonfinite target omitted equally for all arms',
        raw_time='ordinal only; main.TIME=temperature.TIME-1; quarterhour phase inferred from complete day groups; no UTC claim',
        L=L,H=1,warmup=WARMUP,initial_history=INITIAL_HISTORY,update=UPDATE,verification=VERIFICATION,
        service=SERVICE,stage_stride=STRIDE,minimum_update_and_service_origins=MIN_ORIGINS,
        minimum_origins_per_verification_block=MIN_BLOCK_ORIGINS,coverage_exclusions_before_model_outcomes=True,
        initial_history_failure='declare building/pool unavailable; no scan to alternate start',
        source_training_steps=1000,stage_and_warmup_steps=50,optimizer='SGD',learning_rate=.01,batch_size=64,gradient_clip=1.,
        methods=METHODS,scales=SCALES,fixed_ravm_kappa_recent=0.,fixed_ravm_kappa_history=.5,history_capacity=12,delta_ret=0.,
        reference='equal-building, equal-seed development Frozen lifecycle MSE after causal 800-row warmup',
        risk_frequency=RHO,risk_downside_scale=EPSILON_SCALE,
        risk_constraints='separately each of the3 condition streams, equal-building and equal-seed means',
        conditions=CONDITIONS,fault_episode='one uniformly located contiguous4-sample (1h) episode in each192-row update/verification role',
        offset_celsius=2.,fault_cadence_interpretation='controlled chosen stress cadence, not inferred field prevalence',
        service_reference_policy='unmodified observed physical temperature references; baseline acquisition gaps persist; no truth guarantee',
        selection=['lowest equal-condition/equal-building/equal-seed lifecycle MSE','lowest worst-condition downside',
                   'highest equal-condition beneficial-commit recall','manifest order'],
        none_feasible='retain all for that method',final_no_reselection=True,
        uncertainty='two real laboratory houses, same location; report per-building seed means, no population bootstrap CI',
        old_single_home_negative_result_preserved=True,
        disclaimer='No positive outcome or broader DT/control-safety guarantee is promised.')

def inventory():
    data={b:load_building(b) for b in BUILDINGS}
    cov=[coverage(d,p) for d in data.values() for p in POOLS]
    obj=dict(status='INVENTORY_ONLY_NO_MODEL_FIT',data=[{k:v for k,v in d.items() if k not in ['x','y','days','ts']} for d in data.values()],coverage=cov)
    dump(AUDIT/'thermal_data_inventory.json',obj)
    dump(AUDIT/'thermal_protocol_proposed.json',proposed_protocol())
    print(json.dumps([dict(building=c['building'],pool=c['pool'],possible=c['possible_stages'],retained=c['retained_stages'],initial_ok=c['initial_history_eligible']) for c in cov],indent=2))
    return data,cov

def prepare():
    if LOCK.exists():raise FileExistsError('prepare refuses to overwrite a locked thermal protocol')
    data,cov=inventory()
    if not all(c['pool_eligible'] for c in cov):raise ValueError('resolve fixed initial-evidence coverage before any training')
    CHECK.mkdir(parents=True,exist_ok=True);provenance=[]
    for building,d in data.items():
        ids=np.flatnonzero(d['days']<=SOURCE_LAST_DAY)
        train=segment(d,int(ids[0]),int(ids[-1]+1),'warmup')
        for seed in SEEDS:
            torch.manual_seed(seed+(1 if building=='BSE1' else 2)*100000)
            model=Forecast('GRU',H=1,features=d['x'].shape[1]).to(DEVICE)
            model=adapt(model,plain_xy(train),steps=1000,seed=seed)
            path=CHECK/f'source_{building}_{seed}.pt';torch.save(model.state_dict(),path)
            provenance.append(dict(building=building,seed=seed,source_state_hash=state_hash(model),checkpoint_sha256=sha(path),source_segment_hash=train.hash,source_training_origins=valid_origins(train)))
    # Development reference is a complete causally initialized Frozen lifecycle,
    # not a point estimate on the source-pretraining holdout interval.
    reference_rows=[]
    for building,d in data.items():
        for seed in SEEDS:
            provider=ThermalProvider(d,'development',seed,'unmodified')
            rows,cost=run_graph(provider,[dict(id='Frozen',method='Frozen',threshold=0.,scale=0.,fallback=False)])
            rows.to_csv(OUT/f'development_frozen_{building}_{seed}.csv',index=False)
            reference_rows.append(rows)
    reference_data=pd.concat(reference_rows,ignore_index=True)
    unit=reference_data.groupby(['building','model_seed']).deployed_service_mse.mean()
    reference=float(unit.groupby('building').mean().mean())
    if not np.isfinite(reference) or reference<=0:raise ValueError('invalid development reference')
    protocol=proposed_protocol()
    protocol.update(status='LOCKED_BEFORE_CALIBRATION_AND_FINAL',locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        development_frozen_lifecycle_reference=reference,epsilon=EPSILON_SCALE*reference,rho=RHO,
        code_sha256=sha(Path(__file__)),gate_graph_sha256=sha(Path(__file__).parent/'gate_graph.py'),
        data_inventory_sha256=sha(AUDIT/'thermal_data_inventory.json'),source_provenance=provenance)
    protocol['dependency_sha256']={p.relative_to(BASE/'code').as_posix():sha(p) for folder in ['core','vendor'] for p in (BASE/'code'/folder).glob('*.py')}
    dump(LOCK,protocol)
    print('THERMAL_LOCKED',reference,flush=True)

def verify_lock():
    p=json.loads(LOCK.read_text())
    assert p['status']=='LOCKED_BEFORE_CALIBRATION_AND_FINAL'
    assert p['code_sha256']==sha(Path(__file__))
    assert p['gate_graph_sha256']==sha(Path(__file__).parent/'gate_graph.py')
    assert p['data_inventory_sha256']==sha(AUDIT/'thermal_data_inventory.json')
    for name,value in p['dependency_sha256'].items():assert sha(BASE/'code'/name)==value
    for row in p['source_provenance']:
        assert sha(CHECK/f'source_{row["building"]}_{row["seed"]}.pt')==row['checkpoint_sha256']
    for row in json.loads((AUDIT/'thermal_data_inventory.json').read_text())['data']:
        for f in row['files']:assert sha(BASE/f['path'])==f['sha256']
    for row in p['source_provenance']:
        checkpoint=CHECK/f'source_{row["building"]}_{row["seed"]}.pt'
        assert sha(checkpoint)==row['checkpoint_sha256']
    return p

def balanced_metrics(g):
    by=g.groupby(['condition','building','model_seed']).agg(mse=('deployed_service_mse','mean'),
        harmful=('harmful_commit','mean'),downside=('downside','mean'),commit=('commit','mean'),
        good_commits=('commit','sum'),beneficial_candidates=('beneficial_candidate','sum'))
    # Recall denominator counts only actually beneficial candidate stages.
    recalls=g.assign(good=(g.commit*g.beneficial_candidate)).groupby(['condition','building','model_seed']).agg(good=('good','sum'),den=('beneficial_candidate','sum'))
    by['recall']=recalls.good/recalls.den.replace(0,np.nan)
    return by.groupby(['condition','building']).mean(numeric_only=True).groupby('condition').mean(numeric_only=True)

def select_calibration(d,p):
    rows=[]
    for (arm,method,scale),g in d.groupby(['arm','method','scale']):
        by=balanced_metrics(g)
        feasible=bool((by.harmful<=p['rho']).all() and (by.downside<=p['epsilon']).all())
        recall=float(by.recall.mean())
        rows.append(dict(arm=arm,method=method,scale=float(scale),threshold=float(g.threshold.iloc[0]),
             balanced_mse=float(by.mse.mean()),worst_harmful=float(by.harmful.max()),worst_downside=float(by.downside.max()),
             balanced_commit=float(by.commit.mean()),beneficial_commit_recall=recall if np.isfinite(recall) else 0.,feasible=feasible))
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'calibration_grid.csv',index=False)
    chosen=[]
    for method in METHODS:
        eligible=table[(table.method==method)&table.feasible]
        if len(eligible):
            row=eligible.sort_values(['balanced_mse','worst_downside','beneficial_commit_recall','arm'],ascending=[True,True,False,True],kind='stable').iloc[0]
            chosen.append(dict(id=method,method=method,threshold=float(row.threshold),scale=float(row.scale),fallback=False,selected_calibration_arm=row.arm))
        else:chosen.append(dict(id=method,method=method,threshold=0.,scale=0.,fallback=True,selected_calibration_arm=None))
    selection=dict(status='CALIBRATION_SELECTED_BEFORE_FINAL',policies=chosen,calibration_grid_sha256=sha(OUT/'calibration_grid.csv'),protocol_sha256=sha(LOCK))
    dump(OUT/'selection.json',selection)
    return selection

def evaluate(pool):
    p=verify_lock();dest=OUT/pool;dest.mkdir(parents=True,exist_ok=True)
    if pool=='calibration':policies=arms(p['development_frozen_lifecycle_reference'])
    elif pool=='final':
        chosen=json.loads((OUT/'selection.json').read_text())
        assert chosen['status']=='CALIBRATION_SELECTED_BEFORE_FINAL' and chosen['protocol_sha256']==sha(LOCK)
        assert chosen['calibration_grid_sha256']==sha(OUT/'calibration_grid.csv')
        policies=chosen['policies']+[dict(id=m,method=m,threshold=0.,scale=0.,fallback=False) for m in ['Frozen','AlwaysFT']]
    else:raise ValueError(pool)
    completed=[];costs=[];start=time.time()
    for building in BUILDINGS:
        d=load_building(building)
        for seed in SEEDS:
            for condition in CONDITIONS:
                path=dest/f'{building}_{seed}_{condition}.csv'
                if path.exists():raise FileExistsError(f'refuses silent reuse: {path}')
                provider=ThermalProvider(d,pool,seed,condition)
                rows,cost=run_graph(provider,policies)
                rows['protocol_sha256']=sha(LOCK)
                rows['data_inventory_sha256']=p['data_inventory_sha256']
                rows['deployed_service_mse_celsius2']=rows.deployed_service_mse*d['target_scale']**2
                rows['downside_celsius2']=rows.downside*d['target_scale']**2
                rows.to_csv(path,index=False);completed.append(rows)
                costs.append(dict(building=building,seed=seed,condition=condition,**cost))
                dump(OUT/f'{pool}_progress.json',dict(completed=len(completed),total=18,seconds=time.time()-start,costs=costs))
                print(pool,building,seed,condition,round(time.time()-start,1),flush=True)
    allrows=pd.concat(completed,ignore_index=True)
    if pool=='calibration':print(json.dumps(select_calibration(allrows,p)),flush=True)
    else:
        summary=allrows.groupby(['method','condition','building','model_seed']).agg(
            mse=('deployed_service_mse','mean'),mse_celsius2=('deployed_service_mse_celsius2','mean'),
            harmful=('harmful_commit','mean'),downside=('downside','mean'),downside_celsius2=('downside_celsius2','mean'),commit=('commit','mean'),n=('stage','size')).reset_index()
        summary.to_csv(OUT/'final_building_seed_summary.csv',index=False)
        summary.groupby(['method','condition','building']).mean(numeric_only=True).to_csv(OUT/'final_building_summary.csv')
    dump(OUT/f'{pool}_complete.json',dict(pool=pool,rows=len(allrows),costs=costs,seconds=time.time()-start,
                                        protocol_sha256=sha(LOCK),code_sha256=sha(Path(__file__))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['inventory','prepare','calibration','final'])
    args=parser.parse_args()
    if args.phase=='inventory':inventory()
    elif args.phase=='prepare':prepare()
    else:evaluate(args.phase)
