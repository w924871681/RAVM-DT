"""One local owner; mechanically gated, no automatic scientific replay, no periodic LLM."""
from r_common import *
import subprocess,psutil,traceback

def call(script,role,args=()):
 freeze_check();state(current_role=role)
 with (ROOT/f'process/{role}.stdout.log').open('ab') as out,(ROOT/f'process/{role}.stderr.log').open('ab') as err:
  p=subprocess.Popen([sys.executable,'-B','-u','-X','utf8',str(ROOT/script),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 atomic(ROOT/f'process/{role}_IDENTITY.json',dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),script=script,args=args))
 while p.poll() is None:
  atomic(ROOT/'progress/SUPERVISION_LATEST.json',dict(role=role,pid=p.pid,alive=True,at_unix=time.time(),automatic_scientific_retry=False));time.sleep(5)
 atomic(ROOT/f'process/{role}_EXIT.json',dict(exit_code=p.returncode,at_unix=time.time()));assert p.returncode==0,f'{role} failed; no scientific replay'

def main():
 guard();freeze_check();me=psutil.Process();atomic(ROOT/'process/OWNER_IDENTITY.json',dict(pid=me.pid,create_time=me.create_time()))
 for phase in ('Smoke','NewFinal'):
  if phase=='NewFinal' and read(ROOT/'SMOKE_GATE.json')['decision']!='SMOKE_GO':break
  while psutil.virtual_memory().available<2*1024**3:state(status='WAITING_FOR_2GIB_RAM',phase=phase,work='StartupAdmission',eta_seconds=None);time.sleep(5)
  if phase=='NewFinal':assert read(ROOT/'FINAL_ASSET_MATRIX.json')['main_ready']
  call('new_runner.py',phase+'_TRAINING',(phase,));call('new_audit.py',phase+'_AUDIT',(phase,));call('new_analysis.py',phase+'_ANALYSIS',(phase,))
  if phase=='NewFinal':
   if read(ROOT/'NEW_FINAL_MAIN_GATE.json')['decision']=='FINAL_MAIN_PASS':
    # This package explicitly permits Main Final while missing supplementary specifications are reported.
    ready=read(ROOT/'FINAL_ASSET_MATRIX.json')
    if not ready['coverage_ready']:
     atomic(ROOT/'FINAL_COVERAGE_STATUS.json',dict(decision='FINAL_COVERAGE_INCOMPLETE',main_final_pass=True,uncompleted=ready['coverage_blockers'],no_external_or_generalization_completion_claim=True))
     textfile(ROOT/'FINAL_COVERAGE_REPORT_ZH.md','# 论文实验覆盖状态\n\nFINAL_COVERAGE_INCOMPLETE。主Final通过不等于全文覆盖。以下项目未执行，不宣称泛化/外部验证已完成：\n\n'+'\n'.join('- '+x for x in ready['coverage_blockers'])+'\n\n补充实验没有足够精确且事前锁定的角色/预算定义，不能从已观察主Final结果后临时编造补实验方案。原论文与回复信未修改。\n')
  if phase=='Smoke' and read(ROOT/'SMOKE_GATE.json')['decision']!='SMOKE_GO':
   assert not (ROOT/'data/NewFinal_IDENTITY.json').exists() and not any((ROOT/'data').glob('center_63*.npz'))
   atomic(ROOT/'process/FINAL_POOL_UNOPENED_RECEIPT.json',dict(no_new_final_data=True,no_new_final_sgd=True,stop=read(ROOT/'SMOKE_GATE.json')['decision']))
 call('new_delivery.py','DELIVERY')
 atomic(ROOT/'process/OWNER_RECEIPT.json',dict(status='COMPLETE',at_unix=time.time(),no_paper_edits=True,no_retuning=True))

if __name__=='__main__':
 try:main()
 except BaseException:
  error=traceback.format_exc();atomic(ROOT/'process/OWNER_FAILURE.json',dict(error=error,at_unix=time.time(),no_automatic_scientific_retry=True));state(status='FAILED',error=error,active_workers=0,eta_seconds=None)
  textfile(ROOT/'ENGINEERING_STOP_ZH.md','# 本轮工程停止\n\n全部已有工作和实际优化器账本保留，不重训已消费工作；科学完成状态不能补造。\n\n'+error)
  raise
