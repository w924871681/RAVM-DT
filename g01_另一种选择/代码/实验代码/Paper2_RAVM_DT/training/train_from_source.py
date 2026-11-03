"""PATH_B continuation: complete supplied-source fitting -> fresh common warm-up -> fixed lifecycle.

One locked synthetic path has completed source fitting and lifecycle acceptance. Minimal entry
checkpoints are explicitly rejected. It never replaces frozen paper evidence.
"""
from pathlib import Path
import argparse,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as r
def main(a):
 source=a.source_checkpoint.resolve();proof=r.read(a.source_receipt)
 assert proof['full_source_training_complete'],'A truncated source-entry check is not a complete source model'
 assert proof['seed']==a.seed
 assert not a.output.exists(),'Fresh output directory required'
 r.verify();spec=r.read(r.B/'groups'/f'{a.group}.json');assert a.center in spec['centers']and a.seed in spec['seeds']and a.stream in spec['stream_ids']
 assert (proof['architecture'],proof['horizon'])==(spec['architecture'],spec['horizon'])
 if spec['domain']in ('uci','thermbuild'):assert proof['task']==spec['building']
 assert proof['domain']==spec['domain']
 b=r.bootstrap(a.output);spec=r.attach(r.stage(spec,a.output),b)
 with b.context(spec):
  model=b.Forecast().to(b.DEVICE);model.load_state_dict(b.torch.load(source,map_location=b.DEVICE,weights_only=True))
  spec['sources'][str(a.seed)]=dict(path=str(source),sha256=r.sha(source),state_hash=b.state_hash(model))
  warmup=b.Timeline(a.center).warmup();seed=b.training_seed(a.seed,a.center,0)
  initial=b.adapt(model,b.tensors(b.windows(warmup)),steps=50,seed=seed)
  p=Path(spec['root'])/'initial'/f'{a.center}_{a.seed}.pt';b.torch.save(initial.state_dict(),p)
  meta=r.read(p.with_suffix('.json'));meta.update(sha256=r.sha(p),state_hash=b.state_hash(initial),source_sha256=r.sha(source),warmup_hash=warmup.hash,warmup_steps=50,warmup_seed=seed,path_b_source_receipt=str(a.source_receipt));r.atomic(p.with_suffix('.json'),meta)
  arm=next(x for x in spec['arms']if x['id']==a.arm);fn=b.run_legacy if arm['id']in b.LEGACY_NAMES else b.run_trajectory
  fn(spec['phase'],a.center,a.seed,a.stream,arm,'PATH_B_EXTERNAL_REPRODUCTION_NOT_NEW_FINAL')
  td=Path(spec['root'])/'science'/spec['phase']/str(a.center)/str(a.seed)/a.stream/a.arm
  audit=b.audit_trajectory(spec['phase'],td)
  r.atomic(a.output/'PATH_B_RECEIPT.json',dict(status='COMPLETED',source_receipt=str(a.source_receipt),warmup_steps=50,audit=audit,not_frozen_paper_results=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__)
 for n in ['group','center','seed','stream','arm']:p.add_argument('--'+n,required=True,type=int if n in ('center','seed')else str)
 for n in ['source-checkpoint','source-receipt','output']:p.add_argument('--'+n,required=True,type=Path)
 main(p.parse_args())
