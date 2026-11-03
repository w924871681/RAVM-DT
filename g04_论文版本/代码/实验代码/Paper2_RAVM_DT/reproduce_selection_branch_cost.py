"""Only the three missing g01 MWU timing repeats, under project exclusivity."""
import argparse,json,sys
from pathlib import Path
import psutil
from reproduce_selection_branch import validated_plan,read,write,sha,install,core,B
R=B.parent
def exclusive(state,name):
    return state.get('status')=='ACTIVE' and state.get('existing_paper3_slot')==0 and [j['name'] for j in state['jobs'] if j['status']=='RUNNING']==[name]
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--job',required=True);a=p.parse_args()
    plan,choice=validated_plan(a.plan);plan['_path']=str(a.plan.resolve());tasks=read(a.tasks)
    if len(tasks)!=3 or {x['group'] for x in tasks}!={'E5_COST_R1','E5_COST_R2','E5_COST_R3'} or any(x['arm']!='MWU_MEAN__g01' for x in tasks):raise RuntimeError('Only original three g01 repeats allowed')
    if any(x not in plan['tasks'] for x in tasks):raise RuntimeError('Unregistered cost key')
    state=read(R/'validation/parallel_reproduction_scheduler_008.json')
    if not exclusive(state,a.job):raise RuntimeError('Cost requires no other active project job')
    own=psutil.Process()
    other=[]
    for proc in psutil.process_iter(['pid','cmdline']):
        cmd=proc.info['cmdline'] or []
        if proc.pid==own.pid:continue
        if any(Path(x).name in {'reproduce_matrix.py','reproduce_devcal.py','reproduce_selection_branch.py'} for x in cmd):other.append(dict(pid=proc.pid,cmd=cmd))
    if other:raise RuntimeError('Other production trajectory processes remain: '+str(other))
    install(plan,choice)
    # No --read-cache, no changed cost instrumentation, no fresh source training.
    sys.argv=[str(B/'reproduce_matrix.py'),'--mode','source-to-final','--source-python',plan['source_python'],
        '--output',str(a.output),'--task-list',str(a.tasks),'--reuse-sources',plan['reuse_sources'],'--reuse-initials',plan['reuse_initials']]
    for g in sorted({x['group'] for x in tasks}):sys.argv.extend(['--group',g])
    core.main()
    write(a.output/'BRANCH_EXECUTION.json',dict(status='PASS_G01_EXECUTION_AND_ORIGINAL_CAUSAL_AUDITS',trajectories=3,
        plan_sha256=sha(a.plan),tasks_sha256=sha(a.tasks),comparison_pending=True,not_original_g04_reproduction=True,
        project_exclusivity_verified=True,extra_read_cache=False,original_cost_instrumentation=True))
if __name__=='__main__':main()
