"""Recompute only g01-dependent statistics with original scientific functions.

The accepted g04 statistics and other method outputs are reused. Differences
are reported, never used to retune selection or to change an acceptance limit.
"""
import argparse,ast,collections,copy,csv,hashlib,itertools,json,math,statistics,sys
from pathlib import Path
import numpy as np
from compare_outputs import load_actual
from reproduce_selection_branch import read,write,sha
B=Path(__file__).resolve().parent;R=B.parent
G04='MWU_MEAN__g04';G01='MWU_MEAN__g01';JOINT='JOINT__l3__m0'
LIMIT=5.6036475532487204e-5

def functions(path,names,env):
    tree=ast.parse(path.read_text('utf-8'))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    if {x.name for x in nodes}!=set(names):raise RuntimeError('Original function missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)

def csvrows(p,typed=True):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not typed:return rows
    for row in rows:
        for k,v in row.items():
            if v=='':row[k]=None
            elif v in ('True','False'):row[k]=v=='True'
            else:
                try:row[k]=int(v)
                except ValueError:
                    try:row[k]=float(v)
                    except ValueError:pass
    return rows

def save(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:write(p.with_suffix('.empty.json'),dict(rows=0));return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def assign_name(node,name):return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)

def coverage_original_parts():
    path=B/'statistics_source/coverage_statistics.py';tree=ast.parse(path.read_text('utf-8'))
    group=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='group_stats')
    worker=next(x for x in ast.walk(group) if isinstance(x,ast.FunctionDef) and x.name=='worker')
    row=next(x for x in ast.walk(worker) if assign_name(x,'row'))
    wanted={'n','benef','harm','result'}
    body=next(x for x in worker.body if isinstance(x,ast.Try)).body
    summary=[x for x in body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in x.targets)]
    assert len(summary)==4
    loops=[x for x in group.body if isinstance(x,ast.For)]
    arms=[x for x in loops if isinstance(x.target,ast.Name) and x.target.id=='arm']
    pairs=[x for x in loops if isinstance(x.target,ast.Name) and x.target.id=='stream']
    assert len(arms)==2 and len(pairs)==1
    compile_nodes=lambda x:compile(ast.Module(body=x,type_ignores=[]),str(path),'exec')
    return compile_nodes([row]),compile_nodes(summary),compile_nodes(arms),compile_nodes(pairs)

def policy_signature(pair,risk,active):
    qualified=active and len(risk)==3 and all(x['risk_qualified'] for x in risk)
    return dict(qualified_active=qualified,mse_direction='joint_lower' if pair['mse_delta']<0 else 'joint_higher' if pair['mse_delta']>0 else 'equal',
        center_ci_excludes_zero=pair['upper95']<0 or pair['lower95']>0,
        joint_improvement_ci_supported=pair['upper95']<0,center_wins=pair['center_wins'],
        original_relative_improvement_threshold_pass=pair['relative_improvement']>=.002,
        original_center_wins_threshold_pass=pair['center_wins']>=17)

def changed_table_cells(old_path,new_path):
    ids=('phase','group','arm','method','candidate','baseline','stream','center','seed','regime','regime_audit_only','metric','stage')
    def selected(path,arm):
        rows=[x for x in csvrows(path) if x.get('arm')==arm or x.get('baseline')==arm]
        for x in rows:
            for k,v in x.items():
                if v==G01:x[k]=G04  # Treatment label alignment only; values stay unchanged.
        return sorted(rows,key=lambda x:tuple(str(x.get(k,'')) for k in ids))
    left,right=selected(old_path,G04),selected(new_path,G01)
    if len(left)!=len(right):raise RuntimeError('Changed table coverage: '+str(new_path))
    differences=[];timing=[];max_abs=0.
    for index,(x,y) in enumerate(zip(left,right)):
        if x.keys()!=y.keys():raise RuntimeError('Changed table schema: '+str(new_path))
        for key in x:
            av,bv=x[key],y[key]
            if any(type(v) in (float,int) and not math.isfinite(v) for v in (av,bv)):raise RuntimeError('Non-finite table value')
            if av==bv:continue
            numeric=type(av) in (float,int) and type(bv) in (float,int)
            if numeric and (not math.isfinite(av) or not math.isfinite(bv)):raise RuntimeError('Non-finite table value')
            if key.endswith('_seconds') or key.endswith('_bytes'):
                timing.append(dict(row=index,column=key,g04=av,g01=bv));continue
            if numeric:
                delta=abs(av-bv);max_abs=max(max_abs,delta)
                if delta<=1e-7:continue
            differences.append(dict(row=index,column=key,g04=av,g01=bv))
    return dict(file=str(new_path),g04_file=str(old_path),rows=len(left),scientific_differing_cells=len(differences),
        max_absolute_numeric_difference=max_abs,differences=differences,timing_or_storage_differences=timing,
        label_projection='g01->g04 only in comparison; saved CSVs retain actual labels',absolute_numeric_tolerance=1e-7)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    c=read(a.contract)
    for x in c['frozen_bindings']:
        if sha(x['path'])!=x['sha256']:raise RuntimeError('Frozen analysis/claim input changed: '+x['path'])
    g04stats=Path(c['g04_statistics']);g04matrix=Path(c['g04_matrix']);out=a.output.resolve()
    receipt=read(g04stats/'STATISTICS_RECEIPT.json')
    if receipt['status']!='PASS_GENERATED_TABLE_COMPARISON':raise RuntimeError('Original g04 tables not accepted')
    paired=read(Path(c['paired_summary'])/'RECEIPT.json')
    if paired['status']!='PASS_COMPLETE_PAIRED_G01_G04_SENSITIVITY_REPORT':raise RuntimeError('Paired comparison incomplete')
    if out.exists():raise RuntimeError('Fresh output required')
    out.mkdir(parents=True)
    input_manifest=[];routes={}
    for root in map(Path,c['g01_outputs']):
        pair=read(root/'PAIRED_COMPARISON.json')
        if pair['status']!='PASS_PAIRED_COMPARISON_COMPLETE_DIFFERENCES_RETAINED':raise RuntimeError('Unaccepted branch comparison')
        input_manifest.append(dict(path=str(root/'PAIRED_COMPARISON.json'),sha256=sha(root/'PAIRED_COMPARISON.json')))
        for row in pair['rows']:
            key=tuple(row['key'][x] for x in ('group','center','seed','stream'))
            if key in routes:raise RuntimeError('Duplicate branch trajectory')
            routes[key]=(root,row)
    if len(routes)!=2097:raise RuntimeError('Incomplete branch matrix')
    new=read(c['observed_lock']);source=B/'training/code/main'
    env=dict(np=np,collections=collections,math=math,Path=Path,ROOT=out/'main',LIMIT=LIMIT,JOINT=JOINT,STREAMS=('clean','mixed','coherent_sticky'))
    functions(source/'r_statistics.py',['metric','centers','compare','summary'],env)
    functions(source/'new_analysis.py',['risk_contrasts'],env)
    functions(source/'d3_results.py',['csvwrite','aggregate','balanced','phase_summary'],env)
    oldmain=g04stats/'main/results/NewFinal';mainrows=[];supplement_changes=[];risk_qualification_changes=[];cost_rows=[]
    stage_code,trajectory_code,center_code,pair_code=coverage_original_parts()
    for specpath in sorted((B/'training/groups').glob('*.json')):
        spec=read(specpath);group=spec['group'];results=[]
        if group=='Main':pass
        for key in sorted(k for k in routes if k[0]==group):
            root,binding=routes[key];_,center,seed,stream=key
            td=root/'groups'/group/'science'/spec['phase']/str(center)/str(seed)/stream/G01
            if sha(td/'AUDIT.json')!=binding['audit_sha256'] or sha(td/'DONE.json')!=binding['done_sha256']:raise RuntimeError('Accepted output changed')
            rows=load_actual(td)  # Integrity only, no repeated scientific audit/forward.
            if group=='Main':mainrows.extend(rows);continue
            if spec.get('cost_only'):
                for p in sorted(td.glob('COST_UNIFIED_*.json')):cost_rows.append(dict(group=group,arm=G01,**read(p)))
                continue
            e=dict(statistics=statistics,spec=spec,group=group,job=dict(center=center,seed=seed,stream=stream,arm=new['baseline_selection']['MWU_MEAN']['arm_spec']),
                key=dict(group=group,center=center,seed=seed,stream=stream,arm=G01),rows=[],scale=spec.get('target_scale'))
            functions(B/'statistics_source/coverage_statistics.py',['mean','aggregate'],e)
            for t,r in enumerate(rows,1):e.update(t=t,r=r);exec(stage_code,e);e['rows'].append(e['row'])
            exec(trajectory_code,e);results.append(e['result'])
        if group=='Main' or spec.get('cost_only'):continue
        prior=g04stats/'statistics'/group;dest=out/'statistics'/group
        e=dict(np=np,statistics=statistics,spec=dict(spec,arms=[new['baseline_selection']['MWU_MEAN']['arm_spec']]),
            group=group,valid=results,center_rows=[],summary=[],pairs=[],JOINT=JOINT,METRICS=('mse','harm_frequency','mean_downside','commit_rate','beneficial_recall'))
        functions(B/'statistics_source/coverage_statistics.py',['mean','aggregate'],e)
        exec(center_code,e)
        newcenters=list(e['center_rows']);newsummary=list(e['summary'])
        oldcenters=csvrows(prior/'CENTERS.csv');e['center_rows']+= [x for x in oldcenters if x['arm']==JOINT]
        e['spec']['arms']=[dict(id=JOINT),dict(id=G01)];exec(pair_code,e)
        for name,newrows,key in [('TRAJECTORIES.csv',results,'arm'),('CENTERS.csv',newcenters,'arm'),('METHOD_STREAM.csv',newsummary,'arm'),('PAIRED_ALL_BASELINES.csv',e['pairs'],'baseline')]:
            oldrows=csvrows(prior/name);merged=[x for x in oldrows if x[key]!=G04]+newrows;save(dest/name,merged)
        oldpairs=csvrows(prior/'PAIRED_ALL_BASELINES.csv');lookup={(x['stream'],x['metric']):x for x in oldpairs if x['baseline']==G04}
        old_qual={x['stream']:x for x in csvrows(prior/'METHOD_STREAM.csv') if x['arm']==G04}
        for row in newsummary:
            old=old_qual[row['stream']]
            risk_qualification_changes.append(dict(group=group,stream=row['stream'],
                g04_qualified=old['source_empirical_risk_pass'],g01_qualified=row['source_empirical_risk_pass'],
                same=old['source_empirical_risk_pass']==row['source_empirical_risk_pass']))
        for row in e['pairs']:
            old=lookup[row['stream'],row['metric']]
            signature=lambda x:(x['delta_joint_minus_baseline']<0, None if x['ci_high'] is None else x['ci_high']<0,None if x['ci_low'] is None else x['ci_low']>0)
            supplement_changes.append(dict(group=group,stream=row['stream'],metric=row['metric'],g04=old,g01=row,qualitative_pair_result_same=signature(old)==signature(row)))
        write(out/'PROGRESS.json',dict(phase='NEW_MWU_DEPENDENT_STATISTICS_ONLY',last_group=group,training=0))
    if len(mainrows)!=6912:raise RuntimeError('Incomplete Main g01 stages')
    env['phase_summary']('NewFinal',mainrows)
    generated=out/'main/results/NewFinal'
    # Merge the newly computed MWU rows with accepted unchanged methods.
    for p in list(generated.glob('*.csv')):
        if (oldmain/p.name).exists():save(p,[x for x in csvrows(oldmain/p.name) if x.get('arm')!=G04]+csvrows(p))
    joint=[];mainspec=read(B/'training/groups/Main.json')
    for center,seed,stream in itertools.product(mainspec['centers'],mainspec['seeds'],mainspec['stream_ids']):
        joint.extend(load_actual(g04matrix/'groups/Main/science/NewFinal'/str(center)/str(seed)/stream/JOINT))
    pair=env['compare'](joint+mainrows,JOINT,G01)
    risk=env['summary'](mainrows);risk_pairs=env['risk_contrasts'](joint+mainrows,[G01])
    oldpairs=csvrows(oldmain/'PAIRED_MSE_ALL_BASELINES.csv');oldpair=next(x for x in oldpairs if x['baseline']==G04)
    effects=[x for x in oldpairs if x['baseline']!=G04]+[pair]
    oldrisk=csvrows(oldmain/'RISK_BY_METHOD_STREAM.csv');allrisk=[x for x in oldrisk if x['arm']!=G04]+risk
    save(generated/'PAIRED_MSE_ALL_BASELINES.csv',effects);save(generated/'RISK_BY_METHOD_STREAM.csv',allrisk)
    save(generated/'PAIRED_RISK_ALL_BASELINES.csv',[x for x in csvrows(oldmain/'PAIRED_RISK_ALL_BASELINES.csv') if x['baseline']!=G04]+risk_pairs)
    arms=csvrows(oldmain/'RESULTS_BY_ARM.csv');newarms=csvrows(generated/'RESULTS_BY_ARM.csv')
    def primary(pairs,risk,arms):
        active={x['arm']:x['balanced_commit']>0 for x in arms}
        qualified=[x for x in pairs if active[x['baseline']] and len([z for z in risk if z['arm']==x['baseline']])==3 and all(z['risk_qualified'] for z in risk if z['arm']==x['baseline'])]
        return min(qualified,key=lambda x:(x['baseline_mse'],x['baseline'])) if qualified else None
    oldprimary=primary(oldpairs,oldrisk,arms);newprimary=primary(effects,allrisk,newarms)
    same_primary=(oldprimary or {}).get('baseline')==(newprimary or {}).get('baseline')
    oldsig=policy_signature(oldpair,[x for x in oldrisk if x['arm']==G04],next(x for x in arms if x['arm']==G04)['balanced_commit']>0)
    newsig=policy_signature(pair,risk,any(x['commit'] and x['stream'] in ('clean','mixed') for x in mainrows))
    # Same-state sensitivity uses accepted saved d/p and Joint decisions, no model forwards.
    samefile=g04stats/'main_supplements/E0_MAIN_TABLES/SAME_STATE_FROZEN_GATE_DECISIONS.csv'
    same=csvrows(samefile);assert len(same)==6912
    gate={};functions(source/'d3_gate.py',['baseline_decision'],gate)
    threshold=new['baseline_selection']['MWU_MEAN']['arm_spec']['threshold']
    for row in same:row[G01+'_on_joint_same_state']=gate['baseline_decision']('MWU_MEAN',dict(recent_mean=row['d'],mwu_p=row['p']),threshold)
    changed=sum(x[G01+'_on_joint_same_state']!=x[G04+'_on_joint_same_state'] for x in same)
    summary=lambda arm:dict(different=sum(x[JOINT+'_on_joint_same_state']!=x[arm+'_on_joint_same_state'] for x in same),joint_only=sum(x[JOINT+'_on_joint_same_state'] and not x[arm+'_on_joint_same_state'] for x in same),mwu_only=sum(not x[JOINT+'_on_joint_same_state'] and x[arm+'_on_joint_same_state'] for x in same))
    same_state=dict(states=6912,mwu_g01_vs_g04_different=changed,g04_vs_joint=summary(G04),g01_vs_joint=summary(G01),independent_causal_effect_claim=False)
    save(out/'same_state/G01_G04_ON_JOINT_STATES.csv',same);write(out/'same_state/SUMMARY.json',same_state)
    save(out/'cost/G01_COST_UNIFIED.csv',cost_rows)
    costs=[]
    for g in ('E5_COST_R1','E5_COST_R2','E5_COST_R3'):
        old=[x for x in csvrows(g04stats/f'statistics/{g}/UNIFIED_COST.csv') if x['arm']==G04];fresh=[x for x in cost_rows if x['group']==g]
        assert len(old)==len(fresh)==24
        costs.append(dict(group=g,g04_seconds=sum(x['maintenance_common_seconds'] for x in old),g01_seconds=sum(x['maintenance_common_seconds'] for x in fresh)))
    write(out/'cost/REPEATS.json',dict(repeats=costs,scope='separate measured three-run ranges, not population CIs or expected exact timing equality'))
    table_changes=[]
    for path in sorted((out/'statistics').glob('*/*.csv')):
        table_changes.append(changed_table_cells(g04stats/'statistics'/path.parent.name/path.name,path))
    for path in sorted(generated.glob('*.csv')):
        if (oldmain/path.name).exists():table_changes.append(changed_table_cells(oldmain/path.name,path))
    write(out/'TABLE_CHANGES.json',table_changes)
    result=dict(status='PASS_DUAL_BRANCH_CONCLUSION_ANALYSIS_COMPLETE',ordinary_pairs=2094,cost_pairs=3,
        lifecycle_scientific_outputs_equal=paired['scientifically_equal_with_original_tolerance'],
        same_primary_comparator=same_primary,g04_primary=oldprimary,g01_primary=newprimary,
        main_primary_quantitative_conclusion_same=same_primary and oldprimary==newprimary,
        mwu_risk_and_utility_claims_same=oldsig==newsig,g04_mwu_claims=oldsig,g01_mwu_claims=newsig,
        same_state_claims_same=same_state['g04_vs_joint']==same_state['g01_vs_joint'],same_state=same_state,
        supplementary_mwu_qualitative_results_same=all(x['qualitative_pair_result_same'] for x in supplement_changes),
        supplementary_mwu_risk_qualification_same=all(x['same'] for x in risk_qualification_changes),
        scientific_table_values_equal_at_original_tolerance=all(x['scientific_differing_cells']==0 for x in table_changes),
        scientific_table_differing_cells=sum(x['scientific_differing_cells'] for x in table_changes),
        unchanged_primary_claims=['Joint method, DevCal Joint choice, Joint risk rates and activity','Supplementary fixed DualMean contrasts, severe corruption/H4 and external transfer limitations'],
        frozen_manuscript_parameter_text_identical=False,parameter_text_reason='g04 margin2.8018237766243602e-5 versus observed g01 margin5.60364755324872e-6; alternative branch must be labelled',
        source_manuscript_sha256=c['paper_sha256'],source_bindings=c['frozen_bindings'],inputs=input_manifest,
        raw_failures_and_timing_selection_difference_retained=True,training=0,forward_passes=0,
        whole_independent_zip_accepted=False,post_hoc_fixed_input_sensitivity_not_blind=True)
    write(out/'SUPPLEMENTARY_MWU_CONTRASTS.json',supplement_changes);write(out/'SUPPLEMENTARY_MWU_RISK.json',risk_qualification_changes);write(out/'RECEIPT.json',result)
    (out/'CONCLUSION_REPORT_ZH.md').write_text('两条选择路径的结果与论文结论\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\n数值、同状态图表及方法参数差异必须逐项披露；不能将原稿不改解释为g01完整复现。\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if isinstance(v,(str,bool,int))},ensure_ascii=False))
if __name__=='__main__':main()
