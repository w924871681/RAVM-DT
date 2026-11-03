"""Run three original g01 timing repeats with an isolated local guard."""
import argparse,csv,datetime,json,subprocess,sys,time
from pathlib import Path
import psutil
here=Path(__file__).resolve().parent
repo=here/'实验代码'/'Paper2_RAVM_DT'
sys.path.insert(0,str(repo))
from reproduce_selection_branch import validated_plan

def telemetry():
 output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True,timeout=10)
 result=[]
 for row in csv.reader(output.splitlines()):
  if not row or not any(x.strip() for x in row):continue
  if len(row)<2 or not row[0].strip().isdigit():raise RuntimeError('Unknown GPU process telemetry')
  result.append(int(row[0]))
 return result
def gpu_quiet():
 if telemetry():return False
 values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
 if not values:raise RuntimeError('No GPU telemetry')
 return all(int(x.strip())==0 for x in values)
def worker(plan,output):
 import reproduce_selection_branch_cost as cost
 control=output/'控制';(control/'validation').mkdir(parents=True)
 cost.R=control
 name='g01_standalone_cost'
 (control/'validation/parallel_reproduction_scheduler_008.json').write_text(json.dumps(dict(status='ACTIVE',existing_paper3_slot=0,jobs=[dict(name=name,status='RUNNING')])),encoding='utf8')
 taskfile=output/'COST_TASKS.json'
 data=validated_plan(plan)[0];tasks=[x for x in data['tasks'] if x['group'].startswith('E5_COST_')]
 if len(tasks)!=3:raise RuntimeError('Expected exactly three cost groups')
 taskfile.write_text(json.dumps(tasks,sort_keys=True),encoding='utf8')
 sys.argv=[str(repo/'reproduce_selection_branch_cost.py'),'--plan',str(plan),'--tasks',str(taskfile),'--output',str(output/'计时结果'),'--job',name]
 cost.main()
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--plan',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--check-only',action='store_true')
 p.add_argument('--worker',action='store_true')
 a=p.parse_args()
 if a.worker:
  worker(a.plan.resolve(strict=True),a.output.resolve(strict=True));return
 plan,_=validated_plan(a.plan.resolve(strict=True))
 if len([x for x in plan['tasks'] if x['group'].startswith('E5_COST_')])!=3:raise RuntimeError('Expected three cost tasks')
 if a.check_only:
  print(json.dumps(dict(status='PLAN_VALID',gpu_quiet=gpu_quiet(),training_started=False)));return
 if a.output.exists():raise RuntimeError('Use a fresh output directory')
 a.output.mkdir(parents=True)
 quiet=None
 while True:
  try:ok=gpu_quiet()
  except Exception:ok=False
  quiet=(time.monotonic() if quiet is None else quiet) if ok else None
  if quiet is not None and time.monotonic()-quiet>=60 and gpu_quiet():break
  time.sleep(10)
 cmd=[sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--plan',str(a.plan.resolve()),'--output',str(a.output.resolve()),'--worker']
 child=subprocess.Popen(cmd)
 errors=[];samples=0
 with (a.output/'GPU_SAMPLES.jsonl').open('x',encoding='utf8') as f:
  while child.poll() is None:
   try:
    observed=telemetry()
    own={child.pid}
    try:own.update(x.pid for x in psutil.Process(child.pid).children(recursive=True))
    except psutil.NoSuchProcess:pass
    external=[x for x in observed if x not in own]
    if external:errors.append(dict(reason='external GPU processes',pids=external))
    sample=dict(time=datetime.datetime.now().astimezone().isoformat(),gpu_pids=observed,external=external)
   except Exception as e:
    sample=dict(error=repr(e));errors.append(sample)
   f.write(json.dumps(sample)+'\n');f.flush();samples+=1
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:pass
 status='PASS_SAMPLED_COST_EXCLUSIVITY' if child.returncode==0 and not errors else 'COST_REQUIRES_REVIEW'
 (a.output/'COST_GUARD_RECEIPT.json').write_text(json.dumps(dict(status=status,exit_code=child.returncode,samples=samples,errors=errors,scope='Sampled check, not a system GPU reservation')),encoding='utf8')
 if status!='PASS_SAMPLED_COST_EXCLUSIVITY':raise RuntimeError('Cost requires review')
if __name__=='__main__':main()
