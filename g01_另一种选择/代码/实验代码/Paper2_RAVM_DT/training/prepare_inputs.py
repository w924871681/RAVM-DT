"""Explicit no-optimizer input preparation from included bytes or provider raw files."""
from pathlib import Path
import argparse,sys,ast,json,hashlib,tarfile,io
B=Path(__file__).resolve().parent;sys.path[:0]=[str(B),str(B/'code/main/runtime')]
import numpy as np,pandas as pd
import run as r
def save(path,x,y,ts,meta):
 path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,x=x,y=y,ts=ts,meta=json.dumps(meta))
def main(a):
 assert not a.output.exists(),'New output path required';a.output.mkdir(parents=True)
 if a.mode=='frozen':
  r.verify();s=r.stage(r.read(B/'groups'/f'{a.group}.json'),a.output);r.atomic(a.output/'INPUT_PREPARATION_RECEIPT.json',dict(status='PASS_EXACT_INCLUDED_BYTES',group=a.group,root=s['root'],optimizer_steps=0));return
 if a.mode=='synthetic-source':
  from vendor.main_cfg import MainCfg
  from vendor.sim import simulate_centers
  from core.protocol import digest
  env=dict(ROOT=a.output,MainCfg=MainCfg,simulate_centers=simulate_centers,POOLS=dict(source=list(range(10000,10020))),REGIMES=['stationary','gradual','abrupt','recurring'],np=np,json=json,digest=digest)
  (a.output/'data').mkdir();r.selected_functions(B/'code/recipes/historical_bootstrap.py',['generate'],env);env['generate']('source')
 elif a.mode=='alibaba':
  assert r.sha(a.archive)=='3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a'
  membership=r.read(B/'source_inputs/alibaba_pool_manifest.json')['machines'];wanted={x['machine_id_hash']:x for x in membership};parts={h:[]for h in wanted}
  # Pandas' TextIOWrapper queries seekable(); tarfile stream mode does not
  # expose that method on Python 3.13. Seekable gzip mode still reads CSV in
  # bounded chunks and does not extract the large archive to disk.
  with tarfile.open(a.archive,'r:gz')as tf:
   mem=next(m for m in tf if m.isfile()and Path(m.name).name=='machine_usage.csv')
   for chunk in pd.read_csv(tf.extractfile(mem),header=None,chunksize=1000000):
    mapping={mid:hashlib.sha256(str(mid).encode()).hexdigest()for mid in chunk[0].unique()};ids=chunk[0].map(mapping);sub=chunk[ids.isin(wanted)]
    for mid,g in sub.groupby(0):parts[mapping[mid]].append(g)
  for h,meta in wanted.items():
   assert parts[h],'Missing registered machine; never substitute another machine'
   d=pd.concat(parts[h]).drop_duplicates(1).sort_values(1);d=d[d[2].between(0,100)&d[1].notna()].iloc[:17000];assert len(d)>=16760
   raw=d.iloc[:,2:9].to_numpy(dtype='float32');mask=np.isfinite(raw)&(raw>=0)&(raw<=100);mask[:,3]=False;x=np.zeros((len(d),12),np.float32);m=x.copy();x[:,:7]=np.where(mask,raw/100,0);m[:,:7]=mask
   save(a.output/'data'/f'center_{meta["center_id"]}.npz',np.concatenate([x,m],1),raw[:,0]/100,d[1].to_numpy('int64'),meta)
 elif a.mode=='uci':
  env=dict(np=np,pd=pd,CSV=a.csv);r.selected_functions(B/'code/recipes/uci.py',['load_data'],env);x,ts,cols,mean,std,dates=env['load_data']()
  for i,task in enumerate(['T1','T2','T3']):save(a.output/'data'/f'center_{9000+i*1000}.npz',x,x[:,i*2],ts,dict(pool='historical_external_final',domain='uci',task=task,target_scale=float(std[i*2])))
 elif a.mode=='thermbuild':
  names=['room_5_living_110_AT','wea_Tair_out','wea_IbeamHor','wea_IdiffHor','room_5_living_ihs','hp_elP','hp_rps','dist_Tflow_store_ufh','dist_Vol','mode_heat','mode_cool']
  env=dict(np=np,pd=pd,RAW=a.raw_dir,BASE=a.raw_dir.parent,INPUT_NAMES=names,TARGET=names[0],SOURCE_LAST_DAY=120,sha=r.sha)
  r.selected_functions(B/'code/recipes/thermbuild.py',['causal_fill','load_building'],env)
  for i,b in enumerate(['BSE1','BSE2']):
   d=env['load_building'](b);save(a.output/'data'/f'center_{70001+i}.npz',d['x'],d['y'],d['ts'],{k:v for k,v in d.items()if k not in ['x','y','ts','days']})
 r.atomic(a.output/'INPUT_PREPARATION_RECEIPT.json',dict(status='COMPLETED_INPUT_PREPARATION',mode=a.mode,optimizer_steps=0,warning='Generated file-container hashes may differ; compare array identities and units, not timestamps in compressed containers'))
if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['frozen','synthetic-source','alibaba','uci','thermbuild']);p.add_argument('--group');p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path);p.add_argument('--csv',type=Path);p.add_argument('--raw-dir',type=Path);main(p.parse_args())
