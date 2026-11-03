"""Run original E0 regime and same-state gate calculations with relocated inputs."""
import argparse,ast,csv,hashlib,itertools,json,math,threading
from pathlib import Path
B=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def generate_supplements(science_root,stage_records,output,saved_results_check=False):
    science_root,stage_records,output=map(lambda p:Path(p).resolve(),(science_root,stage_records,output))
    if output.exists():raise RuntimeError('Fresh supplement output directory required')
    output.mkdir(parents=True)
    bindings=json.loads((B/'statistics_source/SOURCE_BINDINGS.json').read_text('utf-8-sig'))
    for row in bindings:
        if sha(B/'statistics_source'/row['file'])!=row['sha256']:raise RuntimeError('Frozen statistics source changed')
    original=B/'statistics_source/readonly_e0.py'
    tree=ast.parse(original.read_text('utf-8'))
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    def assigned(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    first=next(i for i,n in enumerate(run.body) if assigned(n,'agg'))
    last=next(i for i,n in enumerate(run.body) if assigned(n,'dev'))
    body=ast.parse("tables=ROOT/'E0_MAIN_TABLES';fig=ROOT/'E0_FIGURE_SOURCE_DATA';tables.mkdir();fig.mkdir()").body+run.body[first:last]
    replacements={"PRIOR / 'results/NewFinal/STAGE_RECORDS.csv'":'STAGE_RECORDS',
                  "PRIOR / 'f2_gate.py'":'GATE_SOURCE',"PRIOR / 'DEVCAL_SELECTED_LOCK.json'":'SELECTION_LOCK'}
    changed=[]
    class Paths(ast.NodeTransformer):
        def visit_BinOp(self,node):
            key=ast.unparse(node)
            if key in replacements:
                changed.append(key);return ast.copy_location(ast.Name(id=replacements[key],ctx=ast.Load()),node)
            return self.generic_visit(node)
    run.body=body;run.decorator_list=[]
    module=Paths().visit(ast.Module(body=[run],type_ignores=[]));ast.fix_missing_locations(module)
    if sorted(changed)!=sorted(replacements):raise RuntimeError('Unexpected source path adaptations')
    inputs={str(stage_records):sha(stage_records)}
    def read(p):
        p=Path(p);data=p.read_bytes();inputs[str(p)]=hashlib.sha256(data).hexdigest();return json.loads(data.decode('utf-8-sig'))
    env=dict(ROOT=output,PRIOR=science_root,STAGE_RECORDS=stage_records,
             GATE_SOURCE=B/'training/code/main/f2_gate.py',SELECTION_LOCK=B/'training/identities/DEVCAL_SELECTED_LOCK.json',
             ast=ast,csv=csv,itertools=itertools,threading=threading,read=read,sha=sha,Path=Path,
             state=lambda **kw:None,execute=lambda planned,fn,*args,**kwargs:[fn(x) for x in planned])
    common=ast.parse((B/'statistics_source/coverage_common.py').read_text('utf-8'))
    save=next(n for n in common.body if isinstance(n,ast.FunctionDef) and n.name=='save_csv')
    exec(compile(ast.Module(body=[save],type_ignores=[]),str(B/'statistics_source/coverage_common.py'),'exec'),env)
    exec(compile(module,str(original),'exec'),env);env['run']()
    comparisons=[]
    for name in ['BENIGN_REGIME_ACTIVITY_RECALL.csv','SAME_STATE_DISAGREEMENT_SUMMARY.csv']:
        actual=output/'E0_MAIN_TABLES'/name;ref=B/'expected_results/data/E0_MAIN_TABLES'/name
        def rows(path):
            with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
        left,right=rows(ref),rows(actual);diff=[]
        if len(left)!=len(right):diff.append({'row_counts':[len(left),len(right)]})
        for i,(a,b) in enumerate(zip(left,right)):
            if a.keys()!=b.keys():diff.append({'row':i,'columns':True});continue
            for k in a:
                if k.endswith('_seconds') or a[k]==b[k]:continue
                try:equal=math.isclose(float(a[k]),float(b[k]),rel_tol=0,abs_tol=1e-7)
                except ValueError:equal=False
                if not equal:diff.append({'row':i,'column':k,'expected':a[k],'actual':b[k]})
        comparisons.append({'file':name,'status':'FAIL' if diff else 'PASS','differences':diff,'sha256':sha(actual)})
    for path,expected in inputs.items():
        if sha(path)!=expected:raise RuntimeError('Input changed while recomputing: '+path)
    result={'status':'PASS' if all(x['status']=='PASS' for x in comparisons) else 'FAIL',
            'scope':'SAVED_RESULT_RECOMPUTATION_ONLY' if saved_results_check else 'NEW_MATRIX_SUPPLEMENT_RECOMPUTATION',
            'source_sha256':sha(original),'path_only_adaptations':changed,'comparisons':comparisons,
            'input_sha256':inputs,'optimizer_steps':0,'development_sensitivity_complete':False,'full_package_accepted':False}
    (output/'SUPPLEMENT_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if result['status']!='PASS':raise RuntimeError('Supplement mismatch')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--science-root',type=Path,required=True);p.add_argument('--stage-records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--saved-results-check',action='store_true',required=True);a=p.parse_args()
    result=generate_supplements(a.science_root,a.stage_records,a.output,a.saved_results_check)
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}))
