from r_common import *
import traceback,psutil

def main():
 guard();freeze_check();assert read(ROOT/'INPUT_CONTRACT_PREFLIGHT.json')['status']=='PASS'
 assert not (ROOT/'process/CORRECTION_OWNER_STARTED.json').exists()
 atomic(ROOT/'process/CORRECTION_OWNER_STARTED.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),at_unix=time.time()),exclusive=True)
 import owner
 original=owner.call
 def call(script,role,args=()):
  result=original('correction_runner.py' if script=='new_runner.py' and tuple(args)==('Smoke',) else script,role,args)
  if script=='new_analysis.py':
   report=ROOT/('SMOKE_REPORT_ZH.md' if args[0]=='Smoke' else 'NEW_FINAL_MAIN_REPORT_ZH.md')
   with report.open('a',encoding='utf8') as f:f.write('\n\n## 工程修正来源与成本\n\n原失败轮次完整保留。Smoke复用同一已暴露开发池的312条产物，修正24条clean原CMC基线；新增24,000步SGD，包含失败轮次的Smoke实际成本为418,400步。此修正Smoke不是新增独立确认。完整来源见CORRECTION_PROTOCOL.json、REUSE_PROVENANCE.json、COST_ACCOUNTING.json。Final仅按原门控在新池执行，不改参数与选择规则。\n')
  return result
 owner.call=call;owner.main()
 atomic(ROOT/'process/CORRECTION_OWNER_RECEIPT.json',dict(status='COMPLETE',at_unix=time.time()),exclusive=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  error=traceback.format_exc();atomic(ROOT/'process/OWNER_FAILURE.json',dict(error=error,at_unix=time.time(),automatic_retry=False));state(status='FAILED',error=error,active_workers=0,eta_seconds=None);raise
