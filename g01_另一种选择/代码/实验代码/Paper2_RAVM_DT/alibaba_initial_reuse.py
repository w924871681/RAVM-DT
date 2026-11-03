"""Reuse completed original-environment Alibaba M0 with exact checks; no GPU work."""
from pathlib import Path
import json
from execution_reuse import read,sha,state,copy_verified

B=Path(__file__).resolve().parent

def reuse(cache,spec,seed,source,target):
    donor=Path(cache)/'historical_initials'/f'{spec["group"]}_{seed}'
    receipt=donor/'INITIAL_COMPARISON.json'
    if not receipt.exists():return False
    result=read(receipt)
    records=result.get('records',[])
    expected={(c,seed) for c in spec['centers']}
    if not result.get('all_match') or len(records)!=len(expected) or {(r['center'],r['seed']) for r in records}!=expected:
        raise RuntimeError('Alibaba initial receipt coverage differs')
    source_sha=sha(source)
    if state(source)!=spec['sources'][str(seed)]['state_hash']:raise RuntimeError('Alibaba source tensor identity differs')
    validated=[]
    for center in spec['centers']:
        p=donor/f'{center}_{seed}.pt';meta=read(p.with_suffix('.json'))
        original=read(B/'training'/spec['input_mapping'][f'initial/{center}_{seed}.json'])
        row=next(r for r in records if r['center']==center and r['seed']==seed)
        if row.get('status')!='PASS' or row.get('state_hash')!=original['state_hash'] or row.get('expected_state_hash')!=original['state_hash']:
            raise RuntimeError('Alibaba comparison differs')
        if meta.get('torch')!='2.5.1+cu121' or not meta.get('python','').startswith('3.11.15 '):raise RuntimeError('Alibaba initial environment differs')
        if meta.get('warmup_steps')!=50 or meta.get('warmup_seed')!=seed+center or meta.get('warmup_hash')!=original['warmup_hash']:
            raise RuntimeError('Alibaba warmup recipe differs')
        if meta.get('source_sha256')!=source_sha or meta.get('sha256')!=sha(p) or meta.get('state_hash')!=original['state_hash'] or state(p)!=original['state_hash']:
            raise RuntimeError('Alibaba initial/source bytes or tensors differ')
        validated.append(p)
    target=Path(target)
    if target.exists():raise RuntimeError('Fresh Alibaba reuse destination required')
    target.mkdir(parents=True)
    files=[]
    for p in validated:
        files.extend([copy_verified(p,target/p.name),copy_verified(p.with_suffix('.json'),target/p.with_suffix('.json').name)])
    files.append(copy_verified(receipt,target/receipt.name))
    (target/'REUSE_RECEIPT.json').write_text(json.dumps(dict(status='PASS_ORIGINAL_ALIBABA_INITIAL_REUSE',
        count=len(validated),donor=str(donor),source_sha256=source_sha,comparison_sha256=sha(receipt),files=files,
        optimizer_steps_this_invocation=0,scientific_tolerance_changed=False),indent=2),encoding='utf-8')
    return True
