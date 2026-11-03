"""Execute the observed g01 selection on the fixed g04 evaluation design.

This is a post hoc paired sensitivity run, never a new blind Final claim.
Original science and audits are called without changing their formulae.
Only the MWU arm and its private selection provenance differ.
"""
import argparse,copy,hashlib,json,shutil,sys,time
from pathlib import Path
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
import reproduce_matrix as core

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):core.r.atomic(Path(p),v)

def validated_plan(path):
    plan=read(path)
    for row in plan['bindings']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Branch input/code identity changed: '+row['path'])
    old=read(plan['historical_lock']);new=read(plan['observed_lock'])
    if old['baseline_selection']['MWU_MEAN']['selected_grid_arm']!='MWU_MEAN__g04':raise RuntimeError('Wrong historical arm')
    if new['baseline_selection']['MWU_MEAN']['selected_grid_arm']!='MWU_MEAN__g01':raise RuntimeError('Wrong observed arm')
    for key in ['candidate','all_qualified_joint_ids','best_active_strong']:
        if old[key]!=new[key]:raise RuntimeError('Unregistered non-MWU choice change')
    for key in ['RECENT_MEAN','DUAL_MEAN']:
        if old['baseline_selection'][key]!=new['baseline_selection'][key]:raise RuntimeError('Unregistered baseline change')
    return plan,new

def branch_spec(original,choice):
    spec=copy.deepcopy(original)
    selected=choice['baseline_selection']['MWU_MEAN']['arm_spec']
    old=[a for a in spec['arms'] if a['method']=='MWU_MEAN']
    if len(old)!=1 or old[0]['id']!='MWU_MEAN__g04':raise RuntimeError('Ambiguous baseline')
    spec['arms']=[copy.deepcopy(selected) if a['method']=='MWU_MEAN' else a for a in spec['arms']]
    return spec

def install(plan,choice):
    raw_read,raw_stage,raw_boot=core.r.read,core.r.stage,core.r.bootstrap
    spec_paths={str((core.T/'groups'/f'{g}.json').resolve()) for g in plan['groups']}
    def route_read(p):
        val=raw_read(p)
        return branch_spec(val,choice) if str(Path(p).resolve()) in spec_paths else val
    def boot(out):
        b=raw_boot(out)
        lock=Path(out)/'_control/SCIENCE_EXECUTION.lock.json'
        shutil.copy2(lock,lock.with_name('ORIGINAL_SCIENCE_EXECUTION.lock.json'))
        write(lock,dict(status='SEALED_G01_FIXED_INPUT_SENSITIVITY',original_lock_sha256=sha(lock),
            branch_plan_sha256=sha(plan['_path']),selection_sha256=sha(plan['observed_lock']),
            original_scientific_code_unchanged=True,new_blind_final_claim=False))
        return b
    def stage(spec,out):
        root=Path(out)/'groups'/spec['group'];root.mkdir(parents=True,exist_ok=True)
        # Original control records are preserved as history, never relabelled.
        hist=root/'branch_provenance/original_controls';hist.mkdir(parents=True)
        excluded={'DEVCAL_SELECTED_LOCK.json','SELECTION_SEAL_RECEIPT.json','process/DATA_ROLE_EVENTS.jsonl',f"data/{spec['phase']}_IDENTITY.json"}
        private=copy.deepcopy(spec)
        for name in excluded:
            if name in private['input_mapping']:
                source=core.T/private['input_mapping'].pop(name)
                dest=hist/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        shutil.copy2(plan['observed_lock'],root/'DEVCAL_SELECTED_LOCK.json')
        from d2c_core import AuditLedger
        ledger=AuditLedger(root/'process/DATA_ROLE_EVENTS.jsonl')
        scope=('G01_FIXED_INPUT_SENSITIVITY','ALL','ALL')
        choice_sha=sha(root/'DEVCAL_SELECTED_LOCK.json');sealed=time.time()
        ev=ledger.append('DEVCAL_SELECTION_DURABLY_FROZEN',scope,1,dict(selection_sha256=choice_sha,
            provenance='Observed new calibration; post hoc fixed-input sensitivity, not historical selection'))
        write(root/'SELECTION_SEAL_RECEIPT.json',dict(selection_sha256=choice_sha,role_event_seq=ev['seq'],sealed_unix=sealed,
            phase_inputs_existed_before_seal=False,scope='Only private materialization. Original evaluation inputs already existed and are reused; not fresh blind generation.'))
        begin=ledger.append('INPUT_GENERATION_STARTED',scope,1,dict(phase=spec['phase'],selection_sha256=choice_sha,
            operation='MATERIALIZE_EXISTING_FROZEN_INPUTS',new_data_generated=False))
        result=raw_stage(private,out)
        end=ledger.append('INPUT_GENERATION_COMPLETED',scope,1,dict(phase=spec['phase'],selection_sha256=choice_sha,
            operation='MATERIALIZE_EXISTING_FROZEN_INPUTS',new_data_generated=False))
        identity=raw_read(hist/f"data/{spec['phase']}_IDENTITY.json") if (hist/f"data/{spec['phase']}_IDENTITY.json").exists() else {}
        identity.update(devcal_lock_sha256=choice_sha,generation_start_seq=begin['seq'],generation_end_seq=end['seq'],
            created_unix=time.time(),materialization_only=True,new_blind_evaluation=False)
        write(root/f"data/{spec['phase']}_IDENTITY.json",identity)
        write(root/'branch_provenance/BRANCH.json',dict(plan_sha256=sha(plan['_path']),observed_selection_sha256=choice_sha,
            historical_selection_sha256=sha(plan['historical_lock']),fixed_existing_evaluation_inputs=True,
            original_control_records='original_controls',algorithm_or_seed_changed=False,
            changed_arm=result['arms'][next(i for i,x in enumerate(result['arms']) if x['method']=='MWU_MEAN')],
            scope='Post hoc paired sensitivity; original runtime scope strings are historical labels, not new blind Final claims'))
        return result
    core.r.read=route_read;core.r.bootstrap=boot;core.r.stage=stage
    # Missing reusable upstream material must fail, never silently retrain it.
    import execution_reuse,alibaba_initial_reuse
    for module,name in [(execution_reuse,'reuse_initial'),(alibaba_initial_reuse,'reuse')]:
        prior=getattr(module,name)
        def strict(*a,_prior=prior,**k):
            result=_prior(*a,**k)
            if not result:raise RuntimeError('Accepted initialization missing; new source/warmup training prohibited')
            return result
        setattr(module,name,strict)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan,choice=validated_plan(a.plan);plan['_path']=str(a.plan.resolve())
    tasks=read(a.tasks);keys={(x['group'],x['center'],x['seed'],x['stream'],x['arm']) for x in tasks}
    allowed={tuple(x[k] for k in ('group','center','seed','stream','arm')) for x in plan['tasks']}
    if not keys or len(keys)!=len(tasks) or not keys<=allowed:raise RuntimeError('Unregistered/duplicate branch task')
    groups=sorted({x['group'] for x in tasks})
    for g in groups:
        spec=read(core.T/'groups'/f'{g}.json')
        if spec.get('cost_only'):raise RuntimeError('Cost sensitivity requires separate exclusive execution; ordinary runner refuses it')
        for seed in {x['seed'] for x in tasks if x['group']==g}:
            if not (Path(plan['reuse_sources'])/core.source_key(spec,seed)/'SOURCE_TRAINING_RECEIPT.json').is_file():raise RuntimeError('Accepted source missing')
    install(plan,choice)
    sys.argv=[str(B/'reproduce_matrix.py'),'--mode','source-to-final','--source-python',plan['source_python'],
        '--output',str(a.output),'--task-list',str(a.tasks),'--reuse-sources',plan['reuse_sources'],'--reuse-initials',plan['reuse_initials'],'--read-cache']
    for g in groups:sys.argv.extend(['--group',g])
    core.main()
    write(a.output/'BRANCH_EXECUTION.json',dict(status='PASS_G01_EXECUTION_AND_ORIGINAL_CAUSAL_AUDITS',trajectories=len(tasks),
        plan_sha256=sha(a.plan),tasks_sha256=sha(a.tasks),comparison_pending=True,not_original_g04_reproduction=True))
if __name__=='__main__':main()
