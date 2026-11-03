"""Recompute final tables with the original aggregation/bootstrap functions."""
import argparse,ast,collections,csv,hashlib,itertools,json,math,shutil,statistics,sys,threading,time,types
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;T=B/'training'
sys.path.insert(0,str(T));import run as r
from compare_outputs import load_actual

def functions(path,names,env):
    tree=ast.parse(path.read_text('utf-8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise RuntimeError('Frozen functions missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)

def td_path(spec,job):return Path(spec['root'])/'science'/spec['phase']/str(job['center'])/str(job['seed'])/job['stream']/job['arm']['id']
def jobs(spec):return [dict(center=c,seed=s,stream=st,arm=a) for c,s,st,a in itertools.product(spec['centers'],spec['seeds'],spec['stream_ids'],spec['arms'])]

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--matrix',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--group',action='append');a=p.parse_args()
    out=a.output.resolve();matrix=a.matrix.resolve()
    if out.exists():raise RuntimeError('Fresh statistics directory required')
    out.mkdir(parents=True);control=out/'_control';control.mkdir();shutil.copy2(T/'identities/SCIENCE_EXECUTION.lock.json',control/'SCIENCE_EXECUTION.lock.json')
    specs=[r.read(f) for f in sorted((T/'groups').glob('*.json')) if not a.group or f.stem in a.group]
    if a.group and {s['group'] for s in specs}!=set(a.group):raise ValueError('Unknown group')
    env=dict(Path=Path,np=np,collections=collections,csv=csv,itertools=itertools,sys=sys,threading=threading,statistics=statistics,time=time,math=math,
             ROOT=control,STORAGE=out,sha=r.sha,read=r.read,atomic=r.atomic,jobs=jobs,
             JOINT='JOINT__l3__m0',METRICS=('mse','harm_frequency','mean_downside','commit_rate','beneficial_recall'),
             state=lambda **kw:None,execute=lambda planned,fn,*args,**kw:[fn(job) for job in planned])
    functions(B/'statistics_source/coverage_common.py',['save_csv'],env)
    functions(B/'statistics_source/coverage_statistics.py',['mean','aggregate','group_stats'],env)
    sys.modules['stage_worker']=types.SimpleNamespace(td_path=td_path)
    groups=[]
    for spec in specs:
        spec=dict(spec,root=str(matrix/'groups'/spec['group']))
        expected=jobs(spec)
        for job in expected:
            td=td_path(spec,job)
            if not (td/'AUDIT.json').exists() or r.read(td/'AUDIT.json')['status']!='VALIDATED':raise RuntimeError('Missing/failed audit: '+str(td))
        if spec['group']!='Main':
            receipt=env['group_stats'](spec)
            if receipt['status']!='PASS':raise RuntimeError(receipt)
            groups.append(receipt);continue
        rows=[]
        # Original main analysis uses lexicographically sorted stage paths.
        for td in sorted(td_path(spec,job) for job in expected):rows.extend(load_actual(td))
        main_env=dict(env,ROOT=out/'main',LIMIT=5.6036475532487204e-5)
        functions(T/'code/main/d3_results.py',['csvwrite','aggregate','balanced','phase_summary'],main_env)
        main_env['phase_summary'](spec['phase'],rows)
        stat_env=dict(main_env,STREAMS=('clean','mixed','coherent_sticky'))
        functions(T/'code/main/r_statistics.py',['metric','centers','compare','summary'],stat_env)
        functions(T/'code/main/new_analysis.py',['risk_contrasts'],stat_env)
        tables=out/'main/results'/spec['phase'];write=main_env['csvwrite'];joint='JOINT__l3__m0'
        baselines=[x for x in sorted({r['arm'] for r in rows}) if x!=joint]
        write(tables/'RISK_BY_METHOD_STREAM.csv',stat_env['summary'](rows))
        write(tables/'PAIRED_MSE_ALL_BASELINES.csv',[stat_env['compare'](rows,joint,b) for b in baselines])
        write(tables/'PAIRED_RISK_ALL_BASELINES.csv',stat_env['risk_contrasts'](rows,baselines))
        from rebuild_main_supplements import generate_supplements
        supplement_dir=out/'main_supplements'
        supplements=generate_supplements(Path(spec['root']),tables/'STAGE_RECORDS.csv',supplement_dir)
        for table in (supplement_dir/'E0_MAIN_TABLES').glob('*.csv'):shutil.copyfile(table,tables/table.name)
        groups.append(dict(group='Main',status='PASS_CORE_AND_SAME_STATE_FINAL_TABLE_RECOMPUTATION',stages=len(rows),trajectories=len(expected),same_state_and_regime_tables=supplements['status'],development_sensitivity='PENDING_SOURCE_REPRODUCTION'))
    comparisons=[]
    for group in groups:
        name=group['group'];folder=out/'main/results/NewFinal' if name=='Main' else out/'statistics'/name
        expected_dir=B/'expected_results/data'/('E0_MAIN_TABLES' if name=='Main' else name)
        for actual in sorted(folder.glob('*.csv')):
            if actual.name=='UNIFIED_COST.csv':continue
            ref=expected_dir/actual.name
            if not ref.exists():continue
            with ref.open(encoding='utf-8-sig',newline='') as f:left=list(csv.DictReader(f))
            with actual.open(encoding='utf-8-sig',newline='') as f:right=list(csv.DictReader(f))
            def normalized(rows):
                ids=('phase','group','arm','method','candidate','baseline','stream','center','seed','regime','metric','stage')
                return sorted([{k:v for k,v in x.items() if not k.endswith('_seconds')} for x in rows],key=lambda x:tuple(x.get(k,'') for k in ids))
            lhs,rhs=normalized(left),normalized(right);differences=[]
            if len(lhs)!=len(rhs):differences.append(dict(row_counts=[len(lhs),len(rhs)]))
            for index,(x,y) in enumerate(zip(lhs,rhs)):
                if x.keys()!=y.keys():differences.append(dict(row=index,columns=True));continue
                for key in x:
                    if x[key]==y[key]:continue
                    try:equal=math.isclose(float(x[key]),float(y[key]),rel_tol=0,abs_tol=1e-7)
                    except (ValueError,TypeError):equal=False
                    if not equal:differences.append(dict(row=index,column=key,expected=x[key],actual=y[key]))
            comparisons.append(dict(group=name,file=actual.name,status='PASS' if not differences else 'FAIL',differences=differences))
    result=dict(status='PASS_GENERATED_TABLE_COMPARISON' if comparisons and all(x['status']=='PASS' for x in comparisons) else 'FAIL',groups=groups,comparisons=comparisons,complete_paper_table_coverage=False)
    (out/'STATISTICS_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(status=result['status'],groups=len(groups),tables=len(comparisons))))
    if result['status']=='FAIL':raise RuntimeError('Statistics differ from frozen values')
if __name__=='__main__':main()
