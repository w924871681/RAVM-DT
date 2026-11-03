"""Prepare a portable g01 branch plan and optionally execute it."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path,PureWindowsPath
here=Path(__file__).resolve().parent
repo=here/'实验代码'/'Paper2_RAVM_DT'
inputs=here/'配置与必要输入'
version_root=here.parent.resolve()
def read(path):return json.loads(path.read_text('utf-8-sig'))
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--source-python',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--run',action='store_true')
 a=parser.parse_args()
 if a.output.exists():raise RuntimeError('Use a new output directory')
 if not a.source_python.is_file():raise RuntimeError('Source Python is missing')
 plan=read(inputs/'PLAN_TEMPLATE.json')
 bound=[]
 for row in plan['bindings']:
  target=(inputs/row['path']).resolve()
  if not target.is_relative_to(version_root):raise RuntimeError('Binding outside version directory')
  if not target.is_file() or digest(target)!=row['sha256']:raise RuntimeError('Frozen input changed: '+str(target))
  bound.append(dict(path=str(target),sha256=row['sha256']))
 plan.update(bindings=bound,historical_lock=str((inputs/'g04_lock.json').resolve()),observed_lock=str((inputs/'g01_lock.json').resolve()),reuse_sources=str((inputs/'sources').resolve()),reuse_initials=str((inputs/'prepared').resolve()),source_python=str(a.source_python.resolve()))
 plan.pop('previous_plan',None);plan.pop('previous_plan_sha256',None)
 tasks=[x for x in plan['tasks'] if not x['group'].startswith('E5_COST_')]
 assert len(tasks)==2094
 a.output.mkdir(parents=True)
 planfile=a.output/'PORTABLE_BRANCH_PLAN.json';taskfile=a.output/'ORDINARY_TASKS.json'
 planfile.write_text(json.dumps(plan,ensure_ascii=False,sort_keys=True),encoding='utf8')
 taskfile.write_text(json.dumps(tasks,ensure_ascii=False,sort_keys=True),encoding='utf8')
 print(json.dumps(dict(status='PREPARED',ordinary_tasks=len(tasks),plan=str(planfile),training_started=a.run)))
 if a.run:
  command=[sys.executable,'-X','utf8','-B',str(repo/'reproduce_selection_branch.py'),'--plan',str(planfile),'--tasks',str(taskfile),'--output',str(a.output/'new_result')]
  subprocess.run(command,check=True,cwd=repo)
if __name__=='__main__':main()
