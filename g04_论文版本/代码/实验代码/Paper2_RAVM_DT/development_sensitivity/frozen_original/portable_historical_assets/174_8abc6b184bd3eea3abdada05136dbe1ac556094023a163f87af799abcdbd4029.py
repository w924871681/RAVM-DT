"""Complete terminal return with a conservative 512MB upload cap."""
from d3_common import *
import gzip,zipfile,csv,shutil
def main():
 protect_originals();s=read(ROOT/'progress/progress_state.json');dest=ROOT/'delivery';lines=['# D3d-1 根因修复与分级验证执行报告','','## Material Passport','','真实冻结神经权重上的新合成开发研究；非现场验证、非新Final。','',f"终态：**{s['status']}**；裁决：**{s.get('auto_gate_decision')}**。",'',f"工程异常：{s.get('error') or '无'}。",'','## 冻结范围','','只比较 U_MSE、既有 U_HUBER 与唯一原型 U_LABEL_WEIGHT；固定门控、50步、原阈值未修改。Huber属于已有鲁棒方法，不能据此宣称原创。所有阶段按冻结GO/STOP执行，禁止重复Confirm或再调参。']
 for phase in ('RootScreen','Pilot','DevCal','Confirm','ExpandedDev'):
  gate=ROOT/f'{phase.upper()}_GO_NO_GO.json';audit=ROOT/f'results/{phase}/INDEPENDENT_AUDIT.json'
  lines+=['',f'## {phase}','','裁决与完整方法指标：','',json.dumps(read(gate),ensure_ascii=False,indent=2) if gate.exists() else '未到该阶段；无科学结果。','','独立审计：',json.dumps(read(audit),ensure_ascii=False,indent=2) if audit.exists() else '未执行或尚未通过，不能声称VALIDATED。']
 lines+=['','## 证据等级与负结果','','RootScreen是固定同父局部候选质量比较，不是24阶段新方法生命周期效果；参考轨迹单独计数。Oracle仅在合法候选和参考动作封存并打开Service后作AUDIT_ONLY诊断，从不进入在线权重/门控或GO选型。中心级区间保留在结果CSV，种子、阶段、窗口不作为独立中心。','','逐中心、流、工况及候选权重信息、全部失败与成功记录、原始Seal/Journal、独立审计及15分钟进度史随包保留。若候选改善但门控风险失败，标记 CANDIDATE_IMPROVED_VERIFICATION_BLOCKER，不增加新规则。','','## 资产与复现','','源权重、原CMC逐块评分器和旧D3c回传在Preflight认证。旧论文、PADV、Final和历史工程全程只读。完整CAS权重保留本地，路径/字节/SHA256索引随包。归档不是自包含重训练包。标签成熟源为合成模拟来源，不冒充外部现场认证。','','最终ZIP必须不超过512,000,000字节，发布前核验CRC、所有成员SHA256和实际大小。','',f"执行耗时（含审计，不含终端打包）：{s.get('finished_unix',time.time())-s['run_started_unix']:.2f}秒。"]
 report=ROOT/'D3d1_EXECUTION_REPORT_ZH.md';atomic_text(report,'\n'.join(lines)+'\n');shutil.copyfile(report,dest/report.name)
 inv=dest/'D3d1_LOCAL_CAS_INVENTORY.jsonl.gz';count=0
 with gzip.open(inv,'wt',encoding='utf8') as f:
  for p in (ROOT/'science').rglob('*'):
   if p.is_file() and p.parent.name=='models':f.write(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),expected=p.name))+'\n');count+=1
 post=[]
 for v in read(ROOT/'PREFLIGHT_ASSETS.json')['source_asset_files']:post.append(dict(path=v['path'],sha256=sha(v['path']),expected=v['sha256']))
 atomic(ROOT/'ORIGINAL_ASSET_POSTRUN_AUDIT.json',dict(passed=all(v['sha256']==v['expected'] for v in post),files=post))
 from d3_zip512 import build
 packed=build(ROOT,dest,inv);archive=packed.pop('archive');h=sha(archive);atomic_text(dest/'D3d1_EXECUTION_RETURN.zip.sha256',h+'  D3d1_EXECUTION_RETURN.zip\n');atomic(dest/'DELIVERY_RECEIPT.json',dict(zip=str(archive),sha256=h,**packed,status=s['status'],decision=s.get('auto_gate_decision'),omitted_local_cas_files=count))
 print(json.dumps(read(dest/'DELIVERY_RECEIPT.json'),ensure_ascii=False))
if __name__=='__main__':main()
