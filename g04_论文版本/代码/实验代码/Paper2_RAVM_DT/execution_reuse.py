"""Verified local reuse and bounded read caches; no scientific result cache.

Only completed source training with matching recipe, environment and exact model
state is reusable. Material is copied into each new output, never symlinked to an
old experiment. Frozen scientific implementations and tolerances are unchanged.
"""
import hashlib,json,shutil
from pathlib import Path

B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def state(path):
    import torch
    value=torch.load(path,map_location='cpu',weights_only=True);h=hashlib.sha256()
    for k,v in value.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def copy_verified(source,target):
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    before=sha(source)
    if target.exists():
        if sha(target)!=before:raise RuntimeError('Conflicting immutable copy: '+str(target))
    else:shutil.copy2(source,target)
    if sha(target)!=before or sha(source)!=before:raise RuntimeError('Copy changed: '+str(source))
    return dict(file=str(target),bytes=target.stat().st_size,sha256=before)

def reuse_source(source_dir,target,spec,seed):
    """Validate the original successful run, then materialize a private copy."""
    source_dir=Path(source_dir);target=Path(target)
    receipt=read(source_dir/'SOURCE_TRAINING_RECEIPT.json')
    expected=dict(domain=spec['domain'],architecture=spec['architecture'],horizon=spec['horizon'],seed=seed,task=spec.get('building',''))
    if any(receipt.get(k)!=v for k,v in expected.items()):raise RuntimeError('Source recipe identity differs')
    if not receipt.get('full_source_training_complete') or receipt['status']!='SOURCE_TRAINING_COMPLETE' or receipt.get('optimizer_steps',0)<=0:raise RuntimeError('Incomplete source cannot be reused')
    env=receipt['environment']
    if env['torch']!='2.5.1+cu121' or not env['python'].startswith('3.11.15 '):raise RuntimeError('Source environment differs')
    models=list((source_dir/'checkpoints').glob('*.pt'))
    if len(models)!=1:raise RuntimeError('Ambiguous source model')
    expected_state=spec['sources'][str(seed)]['state_hash']
    if state(models[0])!=expected_state:raise RuntimeError('Reusable source state differs from frozen identity')
    # A copy includes its original training log/receipt. The destination is a
    # complete local resource, not a link or a runtime dependency on the donor.
    if target.exists():raise RuntimeError('Fresh cache destination required')
    target.mkdir(parents=True)
    files=[]
    for f in sorted(source_dir.rglob('*')):
        if f.is_symlink():raise RuntimeError('No linked cache material')
        if f.is_file():files.append(copy_verified(f,target/f.relative_to(source_dir)))
    result=dict(status='PASS_REUSED_COMPLETED_SOURCE_EXACT_STATE',recipe=expected,state_hash=expected_state,
                donor=str(source_dir),donor_receipt_sha256=sha(source_dir/'SOURCE_TRAINING_RECEIPT.json'),
                scientific_asset_manifest_sha256=sha(B/'training/ASSET_MANIFEST.json'),files=files,
                optimizer_executed_this_invocation=False,original_full_training_receipt_preserved=True)
    (target/'REUSE_RECEIPT.json').write_text(json.dumps(result,indent=2),'utf-8')
    return target/'checkpoints'/models[0].name,target/'SOURCE_TRAINING_RECEIPT.json'

def install_read_cache(b):
    """Cache decoding only under Windows deny-write/delete file handles."""
    import immutable_cache,model_store
    counters=dict(initial_loads=0,initial_cached_loads=0)
    def initial(center,seed):
        path=b.ROOT/'initial'/f'{center}_{seed}.pt';meta=b.read(path.with_suffix('.json'))
        existed=str(path.resolve()) in immutable_cache.WEIGHTS
        model=b.Forecast().to(b.DEVICE)
        model.load_state_dict(immutable_cache.state_dict(path,meta['sha256']))
        if b.state_hash(model)!=meta['state_hash']:raise RuntimeError('Cached initialization state differs')
        counters['initial_loads']+=1;counters['initial_cached_loads']+=int(existed)
        return model
    b.initial=initial
    # ModelStore writes a blob once before hashing. Pins deny subsequent writes;
    # audit and final output comparison remain mandatory and uncached.
    model_store.sha=immutable_cache.hash_file
    return counters

def reuse_initial(cache,spec,center,seed,target,source,receipt,b):
    """Warmup is reusable only with exact source, data, seed and final state."""
    old=Path(cache)/'groups'/spec['group']/'initial'/f'{center}_{seed}.pt'
    if not old.exists() or not old.with_suffix('.json').exists():return False
    meta=read(old.with_suffix('.json'));frozen=read(target.with_suffix('.json'))
    if not meta.get('path_b_source_receipt'):return False
    if meta['sha256']!=sha(old) or meta['state_hash']!=frozen['state_hash']:raise RuntimeError('Initialization cache identity differs')
    if meta.get('source_sha256')!=sha(source):return False
    warmup=b.Timeline(center).warmup()
    if meta.get('warmup_steps')!=50 or meta.get('warmup_seed')!=b.training_seed(seed,center,0) or meta.get('warmup_hash')!=warmup.hash:return False
    if state(old)!=frozen['state_hash']:raise RuntimeError('Cached initial tensor state differs')
    shutil.copy2(old,target)
    meta.update(path_b_source_receipt=str(receipt),reused_verified_warmup=True,actual_new_optimizer_steps_this_invocation=0)
    target.with_suffix('.json').write_text(json.dumps(meta,sort_keys=True),'utf-8')
    return True

def install_devcal_read_cache(engine,trajectory,audit):
    import sys
    sys.path.insert(0,str(B/'training/code/meta'))
    import immutable_cache
    counters=dict(model_loads=0,initial_loads=0)
    def load(self,entry):
        path=self.root/entry['file_sha256']
        if path.stat().st_size!=entry['bytes']:raise RuntimeError('Cached model size differs')
        model=engine.Forecast().to(engine.DEVICE)
        model.load_state_dict(immutable_cache.state_dict(path,entry['file_sha256']))
        if engine.state_hash(model)!=entry['state_hash']:raise RuntimeError('Cached model state differs')
        counters['model_loads']+=1;return model
    def initial(center,seed):
        path=engine.ROOT/'initial'/f'{center}_{seed}.pt';meta=engine.read(path.with_suffix('.json'))
        model=engine.Forecast().to(engine.DEVICE);model.load_state_dict(immutable_cache.state_dict(path,meta['sha256']))
        if engine.state_hash(model)!=meta['state_hash']:raise RuntimeError('Initial cache mismatch')
        counters['initial_loads']+=1;return model
    engine.CAS.load=load
    for module in (engine,trajectory,audit):module.initial=initial
    return counters
