"""Regenerate all eight DevCal inputs using the exact frozen generator body.

This is input regeneration only, with zero model/optimizer execution. Historical
phase permission is replaced by an explicit new reproduction authorization;
all generator functions, centers, seeds and regime parameters stay unchanged.
"""
import argparse,ast,hashlib,json,os,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent;D=B/'development_sensitivity';F=D/'frozen_original'
def read(path):return json.loads(path.read_text('utf-8-sig'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify():
    receipt=read(D/'COLLECTION_RECEIPT.json')
    for row in receipt['files']:
        if sha(F/row['file'])!=row['sha256']:raise RuntimeError('Frozen DevCal file changed: '+row['file'])
    return receipt
def generate(out):
    verify();out=out.resolve()
    if out.exists():raise RuntimeError('Fresh generation directory required')
    if out.is_relative_to(D) or out.is_relative_to(B/'training'):raise RuntimeError('Output overlaps frozen inputs')
    out.mkdir(parents=True)
    sys.path[:0]=[str(F),str(F/'runtime')];sys.dont_write_bytecode=True
    import numpy as np
    import d3_common as common
    import f2_plan
    from d2c_core import AuditLedger
    from vendor.main_cfg import MainCfg
    from vendor.sim import simulate_centers
    auth=out/'REPRODUCTION_AUTHORIZATION.json'
    auth.write_text(json.dumps(dict(scope='New independent DevCal input generation',frozen_execution_sha256=sha(F/'D3F2_EXECUTION.lock.json'),old_go_stop_not_reused_as_new_permission=True,changes_to_scientific_generator=False),indent=2),'utf-8')
    f2_plan.verify_phase_permission=lambda phase: sha(auth) if phase=='DevCal' else (_ for _ in ()).throw(ValueError('Only DevCal authorized by this entry'))
    env=dict(ROOT=out,IDS=read(F/'RUN_MANIFEST.json')['centers'],REGIMES=('stationary','gradual','abrupt','recurring'),np=np,MainCfg=MainCfg,simulate_centers=simulate_centers,AuditLedger=AuditLedger,read=read,sha=sha,atomic=common.atomic,canonical=common.canonical,hashlib=hashlib,json=json,os=os,time=time,execution_lock_sha=lambda:sha(F/'D3F2_EXECUTION.lock.json'))
    tree=ast.parse((F/'d3_engine.py').read_text('utf-8'))
    names={'_logit','_sigmoid','generate'};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:raise RuntimeError('Generator AST mismatch')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(F/'d3_engine.py'),'exec'),env)
    env['generate']('DevCal');checks=[]
    for row in read(F/'data/DevCal_IDENTITY.json')['rows']:
        new=out/row['path'];old=F/row['path']
        with np.load(new,allow_pickle=False) as a,np.load(old,allow_pickle=False) as b:
            if a.files!=b.files:raise RuntimeError('Input schema changed')
            equal={key:a[key].dtype==b[key].dtype and a[key].shape==b[key].shape and bool(np.array_equal(a[key],b[key])) for key in a.files}
        checks.append(dict(center=row['center'],arrays_exact=equal,original_sha256=sha(old),generated_sha256=sha(new),serialized_bytes_exact=sha(old)==sha(new)))
    passed=all(all(c['arrays_exact'].values()) for c in checks)
    result=dict(status='PASS_ALL_EIGHT_DEVCAL_GENERATED_INPUTS' if passed else 'FAIL',generator_sha256=sha(F/'d3_engine.py'),checks=checks,optimizer_steps=0,source_training_verified=False,full_development_reproduction=False)
    (out/'INPUT_COMPARISON.json').write_text(json.dumps(result,indent=2),'utf-8')
    if not passed:raise RuntimeError('Generated data differ; preserve output')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(generate(a.output)))
