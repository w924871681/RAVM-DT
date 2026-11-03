"""Paired g01/g04 differences are outcomes, not reasons to relax acceptance."""
import argparse,collections,copy,json
from pathlib import Path
from compare_outputs import reference_records,load_actual,compare_records,read,sha
from reproduce_selection_branch import validated_plan,write
B=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);a=p.parse_args()
    plan,_=validated_plan(a.plan);tasks=read(a.tasks)
    execution=read(a.matrix/'BRANCH_EXECUTION.json')
    if execution['status']!='PASS_G01_EXECUTION_AND_ORIGINAL_CAUSAL_AUDITS' or execution['tasks_sha256']!=sha(a.tasks):raise RuntimeError('Unaccepted g01 execution')
    accepted_plans={sha(a.plan)}
    if plan.get('previous_plan'):
        prev=Path(plan['previous_plan']);old=read(prev)
        if sha(prev)!=plan['previous_plan_sha256']:raise RuntimeError('Previous plan changed')
        for key in ['historical_lock','observed_lock','groups','tasks','reuse_sources','reuse_initials','source_python']:
            if old[key]!=plan[key]:raise RuntimeError('Scientific plan changed across packaging repair')
        accepted_plans.add(sha(prev))
    if execution['plan_sha256'] not in accepted_plans:raise RuntimeError('Wrong execution plan')
    selected_arm=read(Path(plan['observed_lock']))['baseline_selection']['MWU_MEAN']['arm_spec']
    wanted={(x['group'],x['center'],x['seed'],x['stream']) for x in tasks};seen=set();paired=[]
    for group in sorted({x['group'] for x in tasks}):
        spec=read(B/'training/groups'/f'{group}.json')
        for original in reference_records(B,group):
            k=original['key'];key=tuple(k[x] for x in ('group','center','seed','stream'))
            if key not in wanted or k['arm']!='MWU_MEAN__g04':continue
            if key in seen:raise RuntimeError('Duplicate paired reference')
            seen.add(key);td=a.matrix/'groups'/group/'science'/spec['phase']/str(k['center'])/str(k['seed'])/k['stream']/'MWU_MEAN__g01'
            if read(td/'AUDIT.json')['status']!='VALIDATED':raise RuntimeError('Missing independent causal audit')
            actual=load_actual(td);expected=original['results']
            if len(actual)!=len(expected):raise RuntimeError('Different coverage')
            for stage,row in enumerate(actual,1):
                s=read(td/f'seal_{stage:03}.json')
                if s['arm_spec']!=selected_arm:raise RuntimeError('Wrong actual arm')
                if row['arm']!='MWU_MEAN__g01':raise RuntimeError('Wrong result label')
            # Explicit projection of the treatment ID only; every scientific value
            # retains the original numeric/discrete comparator and tolerance.
            projected=copy.deepcopy(actual)
            for row in projected:row['arm']='MWU_MEAN__g04'
            comparison=compare_records(expected,projected)
            actions=sum(x['commit']!=y['commit'] for x,y in zip(expected,actual))
            states=sum(x['deployed_entry']['state_hash']!=y['deployed_entry']['state_hash'] for x,y in zip(expected,actual))
            metric=lambda rows,name:sum(float(x[name]) for x in rows)/len(rows)
            metrics={name:dict(g04=metric(expected,name),g01=metric(actual,name),delta=metric(actual,name)-metric(expected,name)) for name in ['deployed_service_mse','harmful','downside','commit']}
            row=dict(key=dict(zip(('group','center','seed','stream'),key)),stages=len(actual),different_actions=actions,different_deployed_states=states,
                all_scientific_outputs_equal=comparison['status']=='PASS_SINGLE_LIFECYCLE',scientific_values_checked=comparison['scientific_values_checked'],
                differing_fields=len(comparison['differences']),metrics=metrics,audit_sha256=sha(td/'AUDIT.json'),done_sha256=sha(td/'DONE.json'))
            write(td/'PAIRED_G01_G04.json',dict(**row,comparison=comparison,projection='Only arm identity g01->g04 for comparison; raw outputs unchanged'))
            paired.append(row)
    if seen!=wanted:raise RuntimeError('Incomplete paired coverage')
    result=dict(status='PASS_PAIRED_COMPARISON_COMPLETE_DIFFERENCES_RETAINED',trajectories=len(paired),stages=sum(x['stages'] for x in paired),
        same_trajectories=sum(x['all_scientific_outputs_equal'] for x in paired),different_actions=sum(x['different_actions'] for x in paired),
        different_deployed_states=sum(x['different_deployed_states'] for x in paired),absolute_numeric_tolerance=1e-7,
        scope='Post hoc fixed-input sensitivity; observed differences are preserved, not a failure or proof of equality',
        all_followup_equality_established=False,rows=paired)
    write(a.matrix/'PAIRED_COMPARISON.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
