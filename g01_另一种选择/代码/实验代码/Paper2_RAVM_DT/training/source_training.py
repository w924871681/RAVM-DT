"""PATH_B source fitting. Full mode is an explicit reader command; not automatic.

--max-optimizer-steps 2 is a labelled minimal entry check, never a frozen source.
"""
from pathlib import Path
import argparse,ast,sys,json,hashlib,time,inspect
B=Path(__file__).resolve().parent;sys.path[:0]=[str(B),str(B/'code/main/runtime')]
import numpy as np,torch
from core.models import Forecast,DEVICE,tensors,state_hash,adapt,errors,Batch
from core.protocol import Segment,windows
from run import read,atomic,selected_functions
def source_fit(args):
 out=args.output.resolve();assert not out.exists(),'Fresh output path required';out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'logs').mkdir()
 counter=0;captured=None;started=time.monotonic();full=True
 class EntryBudgetReached(Exception):pass
 def wrap(fn):
  def step(opt,*a,**kw):
   nonlocal counter,captured
   counter+=1
   if args.validation_counter:
    p=args.validation_counter;old=read(p)if p.exists()else dict(attempted_steps=0)
    if old['attempted_steps']>=50000:raise RuntimeError('B4 optimizer budget exhausted')
    atomic(p,dict(attempted_steps=old['attempted_steps']+1,last_case='SOURCE_ENTRY_MINIMAL',case_attempted_steps=counter,kind=opt.__class__.__name__))
   value=fn(opt,*a,**kw)
   if args.max_optimizer_steps and counter>=args.max_optimizer_steps:
    frame=inspect.currentframe().f_back
    while frame and 'model'not in frame.f_locals:frame=frame.f_back
    if frame:captured=frame.f_locals['model']
    raise EntryBudgetReached('Minimal entry check stopped at declared optimizer budget')
   return value
  return step
 torch.optim.Adam.step=wrap(torch.optim.Adam.step);torch.optim.SGD.step=wrap(torch.optim.SGD.step)
 manifest=read(B/'source_inputs/SOURCE_INPUTS.json')
 for row in manifest['rows']:
  p=B/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 if args.domain in ('uci','thermbuild'):assert args.architecture=='GRU'and args.horizon==1
 try:
  if args.domain in ('synthetic','alibaba'):
   class SourceTimeline:
    def __init__(self,c):self.c=c
    def source(self):
     root=args.source_data_root if args.source_data_root else B/'source_inputs'
     with np.load(root/f'center_{self.c}.npz')as z:return Segment(z['x'][:1000],z['y'][:1000],z['ts'][:1000],'source')
   env=dict(ROOT=out,POOLS=dict(source=list(range(10000,10020))),Timeline=SourceTimeline,windows=windows,Forecast=Forecast,DEVICE=DEVICE,np=np,torch=torch,tensors=tensors,state_hash=state_hash,time=time,save_json=atomic)
   selected_functions(B/'code/recipes/historical_pretrain.py',['pretrain'],env)
   env['pretrain'](args.seed,args.architecture,args.horizon,False,args.domain)
  elif args.domain=='uci':
   spec=read(B/'groups'/f'E4_uci_{args.task}.json');asset=B/spec['input_mapping'][f'data/center_{spec["centers"][0]}.npz']
   if args.source_data_root:asset=args.source_data_root/f'center_{spec["centers"][0]}.npz'
   with np.load(asset)as z:x,y,ts=z['x'],z['y'],z['ts']
   idx=['T1','T2','T3'].index(args.task);torch.manual_seed(args.seed+idx*1000);captured=Forecast('GRU',H=1,features=40).to(DEVICE)
   captured=adapt(captured,tensors(windows(Segment(x[:3000],y[:3000],ts[:3000],'warmup'))),steps=200,seed=args.seed+idx*1000)
  else:
   spec=read(B/'groups'/f'E4_thermbuild_{args.task}.json');asset=B/spec['input_mapping'][f'data/center_{spec["centers"][0]}.npz']
   if args.source_data_root:asset=args.source_data_root/f'center_{spec["centers"][0]}.npz'
   with np.load(asset)as z:x,y,ts=z['x'],z['y'],z['ts']
   a,z=manifest['thermal_source_bounds'][args.task];w=windows(Segment(x[a:z],y[a:z],ts[a:z],'warmup'));valid=np.isfinite(w[1]).all(1);xy=Batch(torch.as_tensor(w[0][valid],device=DEVICE),torch.as_tensor(w[1][valid],device=DEVICE),'warmup')
   torch.manual_seed(args.seed+(1 if args.task=='BSE1'else 2)*100000);captured=Forecast('GRU',H=1,features=26).to(DEVICE);captured=adapt(captured,xy,steps=1000,seed=args.seed)
 except EntryBudgetReached:full=False
 if captured is not None:
  torch.save(captured.state_dict(),out/'checkpoints'/('SOURCE_COMPLETE.pt'if full else'ENTRY_CHECK_ONLY_NOT_FROZEN.pt'))
 receipt=dict(status='SOURCE_TRAINING_COMPLETE'if full else'PASS_MINIMAL_SOURCE_TRAINING_ENTRY',domain=args.domain,task=args.task,architecture=args.architecture,horizon=args.horizon,seed=args.seed,optimizer_steps=counter,full_source_training_complete=full,full_matrix_rerun=False,not_paper_scientific_result=True,elapsed_seconds=time.monotonic()-started,environment=dict(torch=str(torch.__version__),python=sys.version),warning='A minimal entry checkpoint must not be substituted for the original frozen source')
 atomic(out/'SOURCE_TRAINING_RECEIPT.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--domain',choices=['synthetic','alibaba','uci','thermbuild'],required=True);p.add_argument('--task',default='');p.add_argument('--architecture',choices=['GRU','MLP','TCN'],default='GRU');p.add_argument('--horizon',type=int,choices=[1,4],default=1);p.add_argument('--seed',type=int,default=7311);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-data-root',type=Path);p.add_argument('--max-optimizer-steps',type=int);p.add_argument('--validation-counter',type=Path);source_fit(p.parse_args())
