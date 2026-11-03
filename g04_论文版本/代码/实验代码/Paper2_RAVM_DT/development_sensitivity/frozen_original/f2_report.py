"""Produce complete positive/negative phase report, no model imports or tuning."""
from d3_common import *
import csv
def rows(p):
 if not p.exists():return []
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(data,cols):
 return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(str(r.get(k,'')) for k in cols)+' |' for r in data])
def main():
 verify_science_lock();state=read(ROOT/'progress/progress_state.json');manifest=read(ROOT/'RUN_MANIFEST.json');sections=[];candidate_steps=warmup_steps=0
 for phase,centers in manifest['centers'].items():
  dest=ROOT/'results'/phase
  if not (dest/'GO_STOP.json').exists():
   partial=len(list((ROOT/'science'/phase).glob('*/*/*/*/result_*.json')))
   sections.append(f'## {phase}\n\n未形成完整科学阶段裁决。已存在阶段记录{partial}；未启动或工程中止不得当作科学失败或补造结果。');continue
  decision=read(dest/'GO_STOP.json');audit=read(dest/'INDEPENDENT_AUDIT.json');training=read(ROOT/f'process/{phase}_TRAINING_RECEIPT.json');candidate_steps+=training['actual_candidate_steps'];warmup_steps+=training['actual_warmup_steps']
  aa=rows(dest/'RESULTS_BY_ARM.csv');tab=[]
  for a in aa:
   tab.append(dict(配置=a['arm'],MSE=f"{float(a['balanced_mse']):.9g}",最大harm=f"{float(a['worst_harmful']):.4%}",最大downside=f"{float(a['worst_downside']):.7g}",提交率=f"{float(a['balanced_commit']):.4%}",风险合格=a['feasible'],开发含缓冲合格=a['dev_qualified'],维护秒=f"{float(a['maintenance_wall_seconds']):.3f}"))
  detail=''
  if phase=='DevCal':detail='\n\n配置选择与所有落选配置见 `results/DevCal/SELECTION_EVIDENCE.json`；只有合格且活跃的强基线能够成为主比较者。λ=0不能颁发联合算法GO。'
  effects=rows(dest/'PAIRED_CENTER_EFFECTS.csv')
  sections.append(f'''## {phase}

中心{centers}；种子{manifest['seeds'][phase]}；完成{audit['trajectories']}条真实计算生命周期、{audit['stage_rows']}条阶段记录，每条均实际运行24阶段。独立审计 **{audit['status']}**。裁决 **{decision['decision']}**。

```json
{json.dumps(decision,ensure_ascii=False,indent=2)}
```

{table(tab,list(tab[0]))}
{detail}

训练/生命周期Runner墙钟{training['wall_seconds']:.3f}秒，含共享M0与资源等待；独立审计{audit['wall_seconds']:.3f}秒，审计SGD为0。实际M0 SGD {training['actual_warmup_steps']}步，候选SGD {training['actual_candidate_steps']}步。全部逐中心/流/工况MSE、harm、downside、提交率以及计时/模型缓存见本阶段CSV；不能把候选局部有益损失幅度累加当部署收益。

配对中心bootstrap（4000次、种子20261020，所有种子/流/阶段留在中心簇内）完整对照：

{table(effects,['candidate','baseline','centers','center_wins','paired_mse_delta','lower95','upper95','relative_improvement']) if effects else '无可比较的已选联合配置，不能造配对结论。'}
''')
 end=state.get('stop_decision','UNFINISHED');description=''
 if end.startswith('STOP_P0'):description='固定P0未达到工程/行为烟测条件，停止此具体联合分支；P0不是机制选型或算法有效性证据。'
 elif end.startswith('STOP_DEV_CAL'):description='固定DevCal未找到满足风险、coherent缓冲及相对合格活跃强基线效用要求的正λ联合配置；本轮不支持推进该联合规则。保留所有配置，禁止再调参或重抽中心。'
 elif end.startswith('STOP_CONFIRM'):description='已锁定联合配置在新开发Confirm未满足全部预定条件；当前联合规则未建立所要求的强基线优势，停止此具体分支。'
 elif end.startswith('STOP_AFTER_EXPANDED'):description='已按预定GO链完成扩展开发，结果以ExpandedDev实测为准；仍不是新Final或现场验证。若没有风险合格的实质强基线优势，应收敛此具体分支。'
 else:description='本轮存在工程/资产/审计停止，不能伪装成完整科学结果；已完成与部分证据均保留。'
 text=f'''# D3f2 联合门控有限自动实验报告

## Material Passport

协议原字节SHA256 `{sha(ROOT/'EXPERIMENT_LOCK.json')}`；执行锁SHA256 `{execution_lock_sha()}`。冻结RAVM/CMC系统、原U_MSE 50步SGD（lr=.01、batch64有放回、clip1）、每(center,seed)仅一次50步M0，三流/方法共享M0。真实训练与前向计算使用原冻结**合成开发生成器**；真实计算轨迹不代表真实现场数据。

## 最终停止与科学结论

**{end}**。{description} D3f1 INFO_GO仅支持判别信息研究，不能作为联合算法成功证明。

唯一新公式：`d<0 and d+lambda*max(parent_recent_verification_mean_loss,1e-12)*(p-.2)<=-margin`。三个当前Verification标量通过原Stage3逐窗口路径计算；p是经验排序信号，不是校准风险概率。没有新增候选生成器、历史权重或延迟反馈；历史证据/成熟反馈仅保留于原历史对照的冻结执行路径，不能用于联合门控。

## 预算、覆盖与公平性

全部四阶段都运行的条件性预算上限为2736轨迹、65664方法阶段、3283200候选SGD+3600 M0 SGD=3286800步，**不是已执行数量**。完整阶段收据累计实际候选{candidate_steps}、M0 {warmup_steps}、合计{candidate_steps+warmup_steps}步。若有部分失败，以`process/SGD_STEP_ACCOUNTING.jsonl`及最后进度中的实际成功调用补充，不把未完成调用报为零。原源模型预训练0步，审计0步；前向工程验收0步。已完成SGD全局计数 `{state.get('actual_sgd_steps',0)}`。

DevCal联合4×4配置含λ=0控制；RecentMean/MWU-Mean/DualMean各自16项原校准网格。所有配置逐条独立运行24阶段，不以旧D3f1结局模拟、不从别的轨迹复用候选训练。MSE按中心等权clean/mixed平衡均值。harm是有害提交次数除全部阶段机会；downside是提交后的正损失差在全部机会上的平均。风险要求对每流分别核验，不用合并流掩盖coherent风险。活跃定义为clean/mixed至少一次提交，不能拿retain-all当最强活跃对照。

成本为同口径每条方法维护路径实际训练、Verification、模型缓存/封存及原v0.2必要成熟反馈前向；Service科学评价与独立审计另计。为完整审计，所有臂保留原统一评分仪器的计算成本，不能把本轮共享GPU并发测时解释成隔离硬件能耗或最终优化实现性能。分母/种子/簇均不改变。

{''.join(sections)}

## 因果与资产审计边界

阶段决策持久Seal后才允许同阶段Service；独立进程逐条验证事件哈希链、签名、M0公平性、24阶段父状态继承、原生分块/Stage3评分、联合公式、全部Service真实权重前向、源权重/脚本和实际预算。未来阶段数据仅在前阶段GO与所需配置锁之后生成。权限由受信提供器与审计顺序实现，不宣称是操作系统级恶意代码隔离或第三方现场时间戳。

中心身份60000–60031在生成前统一冻结；旧D3f1、旧开发/校准/Final身份与47000–47015均排除。原论文、PADV、旧Final、历史实验和权重保持原封存。没有自动进入新Final或修改论文。

## 完整复现与交付索引

`EXPERIMENT_LOCK.json`/`D3F2_EXECUTION.lock.json`/`ANALYSIS_IMPLEMENTATION_LOCK.json`：科学配置与源码身份；`SOURCE_CHANGES.diff`及本轮源码：必要工程差异；`science/`：每阶段seal/result、完整ledger、模型身份和DONE；`results/`：全部正负配置、逐中心/流/工况统计、成本和独立审计；`progress/`：每15分钟实际进度、动态worker、资源/异常与最终快照；`reference/`：原冻结资产及D3f1证据身份。`CAS_EXTERNAL_INDEX.json`若非空，列出因体积外置且完整保留在本地的CAS路径、大小及SHA256，不能视作已包含这些权重字节。原输入包设计文档全部保留。

局限：合成开发、单一冻结生成系统，开发Confirm仍非新Final；分组/种子/时间窗相关，bootstrap按中心整簇重采样，不使用阶段或流当独立中心。分流/工况区间描述性、不作额外选择；有限网格失败不等于一般不可能，成功不证明原创性、现场有效性或总体风险保证。所有负结果与未运行阶段明确保留，无第二套机制、无阈值重调、无中心替换。
'''
 atomic_text(ROOT/'D3f2_EXECUTION_REPORT_ZH.md',text)
if __name__=='__main__':main()
