"""Compare scientific stage outputs, separately from machine timing/provenance."""
import argparse, gzip, hashlib, json, math
from pathlib import Path

NON_SCIENTIFIC = {'decision_sha256', 'label_authority_public_key', 'storage_path',
                  'file_sha256', 'bytes', 'gpu_peak_allocated_bytes'}
# These exact nested fields are assigned by core.engine.run from the caller's
# run label, perf_counter timings and CUDA peak allocation. Keep them in raw
# evidence and cost tables, but do not require historical machine timings to
# equal a new run. Scientific counters, buffers, states and values stay checked.
NATIVE_RUNTIME_FIELDS={'run_id','candidate_update_time_s','verification_time_s','maintenance_time_s','peak_gpu_memory_bytes'}

def compare_records(expected_rows, actual_rows):
    differences, checked, ignored = [], 0, set()
    provenance=[]
    def split_recovery(row,stage,side):
        fields={'recovery_measurement_status','recovery_original_engine_sha256'}
        found=fields.intersection(row)
        if not found:return row
        engine=Path(__file__).resolve().parent/'training/code/main/runtime/core/engine.py'
        if (found!=fields or row.get('recovery_measurement_status')!='SEALED_WEIGHTS_REEVALUATED_ZERO_SGD_ORIGINAL_TIMINGS_UNAVAILABLE'
            or row.get('recovery_original_engine_sha256')!=sha(engine)):
            differences.append(dict(field=f'result_{stage:03}.json.recovery_provenance',side=side,error='Invalid recovery identity'))
        provenance.append(dict(stage=stage,side=side,metadata={k:row[k] for k in found},
            classification='Historical execution provenance; not a claim about the other run'))
        ignored.update(found)
        return {k:v for k,v in row.items() if k not in fields}
    def visit(a, b, path):
        nonlocal checked
        key = path.rsplit('.', 1)[-1]
        if key in NON_SCIENTIFIC or key.endswith('_seconds') or ('.native_original_row.' in path and key in NATIVE_RUNTIME_FIELDS):
            ignored.add(key)
            return
        if isinstance(a, dict) and isinstance(b, dict):
            ak = {k for k in a if k not in NON_SCIENTIFIC and not k.endswith('_seconds')}
            bk = {k for k in b if k not in NON_SCIENTIFIC and not k.endswith('_seconds')}
            ignored.update((a.keys() | b.keys()) - (ak | bk))
            if ak != bk:
                differences.append({'field': path, 'missing': sorted(ak-bk), 'extra': sorted(bk-ak)})
            for k in a.keys() & b.keys(): visit(a[k], b[k], path+'.'+k)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b): differences.append({'field': path, 'lengths': [len(a),len(b)]})
            for i,(x,y) in enumerate(zip(a,b)): visit(x,y,path+f'[{i}]')
        else:
            checked += 1
            if isinstance(a, (float,int)) and isinstance(b,(float,int)) and not isinstance(a,bool):
                equal = math.isclose(a,b,rel_tol=0,abs_tol=1e-7)
            else: equal = a == b
            if not equal: differences.append({'field':path,'expected':a,'actual':b})
    expected_stages=len(expected_rows)
    actual_stages=len(actual_rows)
    if expected_stages!=actual_stages: differences.append({'field':'stages','expected':expected_stages,'actual':actual_stages})
    for stage in range(1,expected_stages+1):
        name=f'result_{stage:03}.json'
        if stage<=actual_stages:visit(split_recovery(expected_rows[stage-1],stage,'reference'),split_recovery(actual_rows[stage-1],stage,'actual'),name)
    return {'status':'PASS_SINGLE_LIFECYCLE' if not differences else 'FAIL',
            'scientific_values_checked':checked,'stages':expected_stages,'absolute_numeric_tolerance':1e-7,
            'ignored_machine_or_storage_fields':sorted(ignored), 'differences':differences,
            'validated_historical_recovery_provenance':provenance,
            'scope':'One lifecycle scientific comparison; upstream training is assessed separately.'}

def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def reference_records(package,group):
    package=Path(package).resolve(); project=package.parents[3]
    manifest=read(package/'evidence/frozen_trajectories/MANIFEST.json')
    row=next(r for r in manifest['groups'] if r['group']==group)
    total=0
    for item in row['parts']:
        relative=Path(item['file'])
        if relative.is_absolute():raise RuntimeError('Absolute reference path refused')
        path=(package/relative).resolve()
        if not path.is_relative_to(project) or sha(path)!=item['sha256']:raise RuntimeError('Reference path or hash mismatch')
        count=0
        with path.open('r',encoding='utf-8') as f:
            for line in f:
                count+=1;total+=1;yield json.loads(line)
        if count!=item['trajectories']:raise RuntimeError('Reference split count mismatch')
    if total!=row['trajectories']:raise RuntimeError('Reference group count mismatch')

def load_actual(td):
    done=read(td/'DONE.json');rows=[]
    for stage in range(1,done['stages']+1):
        path=td/f'result_{stage:03}.json'
        if sha(path)!=done['result_sha256'][stage-1]:raise RuntimeError('New result hash mismatch: '+str(path))
        rows.append(read(path))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--matrix',required=True,type=Path);p.add_argument('--group',action='append');p.add_argument('--partial-entry-check',action='store_true')
    a=p.parse_args();package=Path(__file__).resolve().parent;records=[];missing=[];failures=[]
    specs=[read(f) for f in sorted((package/'training/groups').glob('*.json')) if not a.group or f.stem in a.group]
    if a.group and {s['group'] for s in specs}!=set(a.group):raise ValueError('Unknown group')
    for spec in specs:
        for original in reference_records(package,spec['group']):
            k=original['key'];td=a.matrix/'groups'/k['group']/'science'/spec['phase']/str(k['center'])/str(k['seed'])/k['stream']/k['arm']
            if not (td/'DONE.json').exists():missing.append(k);continue
            result=compare_records(original['results'],load_actual(td))
            (td/'SCIENTIFIC_COMPARISON.json').write_text(json.dumps(result,indent=2),'utf-8')
            records.append(dict(key=k,status=result['status'],scientific_values_checked=result['scientific_values_checked']))
            if result['differences']:failures.append(dict(key=k,differences=result['differences']))
    full=not missing and len(specs)==17 and len(records)==17118 and not a.partial_entry_check
    passed=bool(records) and not failures and (not missing or a.partial_entry_check)
    result=dict(status=('PASS_FULL_SCIENTIFIC_MATRIX' if full else 'PASS_PARTIAL_ENTRY_CHECK') if passed else 'FAIL',completed=len(records),full_matrix=full,missing=missing,failures=failures,records=records,execution=read(a.matrix/'EXECUTION_RECEIPT.json'))
    (a.matrix/'MATRIX_COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'records','missing','failures','execution'}},ensure_ascii=False))
    raise SystemExit(0 if passed else 1)
