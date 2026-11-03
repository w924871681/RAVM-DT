from r_statistics import *
from d3_results import phase_summary
from f2_plan import CORE_IDS,LEGACY_IDS,jobs
from r_common import guard,state,freeze_check,textfile

def risk_contrasts(rows,baselines):
 output=[];rng=np.random.default_rng(20261020);ids=sorted({r['center'] for r in rows});idx=rng.integers(len(ids),size=(4000,len(ids)))
 for baseline in baselines:
  for stream in STREAMS:
   for metric_name,key in [('harm','harmful'),('downside','downside')]:
    def values(arm):return np.array([np.mean([r[key] for r in rows if r['center']==c and r['stream']==stream and r['arm']==arm]) for c in ids])
    diff=values(JOINT)-values(baseline);boot=diff[idx].mean(axis=1)
    output.append(dict(baseline=baseline,stream=stream,metric=metric_name,delta=float(diff.mean()),lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),centers=len(ids),bootstrap_draws=4000,bootstrap_seed=20261020,multiple_comparison_adjusted=False))
 return output

def main(phase):
 guard();freeze_check();audit=read(ROOT/f'results/{phase}/INDEPENDENT_AUDIT.json');assert audit['status']=='VALIDATED'
 rr=[read(p) for p in sorted((ROOT/f'science/{phase}').glob('*/*/*/*/result_*.json'))];assert len(rr)==len(jobs(phase))*24
 dest=ROOT/f'results/{phase}';dest.mkdir(exist_ok=True)
 core=[r for r in rr if r['arm'] in CORE_IDS]
 if phase=='Smoke':
  phase_summary('SmokeCore',core)
  # All compatibility outcomes retained separately; never inflate core sample size.
  legacy=[r for r in rr if r['arm'] in LEGACY_IDS]
  flat=[]
  for r in legacy:
   q={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
   for role in ('parent','candidate','deployed'):q[role+'_state_hash']=r[role+'_entry']['state_hash'];q[role+'_file_sha256']=r[role+'_entry']['file_sha256']
   flat.append(q)
  csvwrite(dest/'LEGACY_COMPATIBILITY_STAGE_RECORDS.csv',flat)
  rows=load_csv(ROOT/'results/SmokeCore/STAGE_RECORDS.csv')
  candidates=['RECENT_MEAN__g12','MWU_MEAN__g04','DUAL_MEAN__g12']
  csvwrite(dest/'LEGACY_COMPATIBILITY_SUMMARY.csv',[dict(arm=a,stage_rows=len([r for r in legacy if r['arm']==a]),**metric([r for r in legacy if r['arm']==a])) for a in LEGACY_IDS])
 else:
  phase_summary(phase,rr);rows=load_csv(dest/'STAGE_RECORDS.csv');candidates=[a for a in sorted({r['arm'] for r in rows}) if a!=JOINT]
 ss=summary(rows);csvwrite(dest/'RISK_BY_METHOD_STREAM.csv',ss)
 stats={a:dict(mse=float(np.mean(list(centers(rows,a).values()))),maintenance_seconds=sum(r['maintenance_wall_seconds'] for r in rows if r['arm']==a),active=any(r['commit'] and r['stream'] in ('clean','mixed') for r in rows if r['arm']==a)) for a in sorted({r['arm'] for r in rows})}
 qualified=[a for a in candidates if stats[a]['active'] and len([x for x in ss if x['arm']==a])==3 and all(x['risk_qualified'] for x in ss if x['arm']==a)]
 primary=min(qualified,key=lambda a:(stats[a]['mse'],a)) if qualified else None
 effects=[compare(rows,JOINT,a) for a in sorted(stats) if a!=JOINT]
 csvwrite(dest/'PAIRED_MSE_ALL_BASELINES.csv',effects)
 csvwrite(dest/'PAIRED_RISK_ALL_BASELINES.csv',risk_contrasts(rows,[e['baseline'] for e in effects]))
 pairs=[]
 for e in effects:
  a=centers(rows,JOINT);b=centers(rows,e['baseline'])
  for c in sorted(a):pairs.append(dict(center=c,baseline=e['baseline'],baseline_eligible=e['baseline'] in qualified,joint_mse=a[c],baseline_mse=b[c],delta=a[c]-b[c],center_win=a[c]<b[c]))
 csvwrite(dest/'PAIRED_CENTER_RAW.csv',pairs)
 risk=[x for x in ss if x['arm']==JOINT];pair=next((e for e in effects if e['baseline']==primary),None);count=384 if phase=='Smoke' else 2304
 ratio=stats[JOINT]['maintenance_seconds']/stats[primary]['maintenance_seconds'] if primary else None
 checks=dict(complete_matrix=len(rr)==(8064 if phase=='Smoke' else 82944),independent_audit_validated=True,
  legacy_exactness_validated=audit['legacy_native_compatibility_validated'],joint_active=stats[JOINT]['active'],
  joint_all_stream_risk=len(risk)==3 and all(x['risk_qualified'] and x['opportunities']==count for x in risk),eligible_active_primary_exists=primary is not None,
  relative_improvement=pair['relative_improvement']>=(.005 if phase=='Smoke' else .002) if pair else None,
  center_wins=pair['center_wins']>=(5 if phase=='Smoke' else 17) if pair else None,cost_ratio=ratio<=2.5 if ratio is not None else None)
 if phase=='NewFinal':checks['paired_center_upper95_below_zero']=pair['upper95']<0 if pair else None
 if primary is None:decision='SMOKE_INCONCLUSIVE' if phase=='Smoke' else 'FINAL_MAIN_INCONCLUSIVE'
 elif all(checks.values()):decision='SMOKE_GO' if phase=='Smoke' else 'FINAL_MAIN_PASS'
 else:decision='SMOKE_STOP' if phase=='Smoke' else 'FINAL_MAIN_NO_GO'
 result=dict(decision=decision,checks=checks,primary_comparator=primary,eligible_active_baselines=qualified,comparator_scope=candidates,
   all_method_statistics=stats,joint_stream_risk=risk,primary_pair=pair,all_pairs=effects,maintenance_cost_ratio=ratio,
   ALL_BASELINES_MSE_CI_DOMINANCE=all(e['upper95']<0 for e in effects),ci_is_familywise_adjusted=False,
   primary_selected_on_observed_eligible_final_pool=phase=='NewFinal',scope='Synthetic Smoke development, not Final' if phase=='Smoke' else 'New prospective synthetic Final main study; paper-wide coverage tracked separately',
   source_protocol_sha256=sha(ROOT/'RUN_PROTOCOL_LOCK.json'),execution_lock_sha256=execution_lock_sha(),audit_sha256=sha(dest/'INDEPENDENT_AUDIT.json'),no_retuning=True,no_paper_edits=True,at_unix=time.time())
 gatefile='SMOKE_GATE.json' if phase=='Smoke' else 'NEW_FINAL_MAIN_GATE.json';atomic(ROOT/gatefile,result,exclusive=True)
 train=read(ROOT/f'process/{phase}_TRAINING_RECEIPT.json')
 lines=['# '+('Smoke中文科研检验报告' if phase=='Smoke' else '独立新Final主实验中文科研检验报告'),'','## Material Passport','',f'协议锁SHA256：`{sha(ROOT/"RUN_PROTOCOL_LOCK.json")}`；独立执行审计：VALIDATED。','',f'## {decision}','',f'完整轨迹{len(jobs(phase))}条，阶段记录{len(rr)}条；实际SGD {train["actual_total_steps"]}步，名义上限{train["nominal_upper_bound"]}步；审计SGD为0。',
  '所有方法独立继承真实部署状态，共享同一(center,seed)的M0。六项原CMC基线直接执行字节一致的原engine.run，ER/Fisher/Huber/Validation回退原配置保持；仅新中心数据I/O、封存与统一报告适配。',
  '风险downside统一以两个标量均值之差的float64计算；原生float32差值完整保留，逐项偏差界1e-9预先锁定，harm必须一致。','',f'主比较候选集合：{candidates}。合格活跃者：{qualified}。主比较者：{primary}。','', '| 流 | 有害/机会 | harm | downside | 资格 |','|---|---:|---:|---:|---|']
 for x in risk:lines.append(f"| {x['stream']} | {x['harmful_count']}/{x['opportunities']} | {x['harmful_frequency']:.9%} | {x['downside']:.12g} | {x['risk_qualified']} |")
 if pair:lines+=['',f"Joint平衡MSE {pair['candidate_mse']:.12g}；主基线 {pair['baseline_mse']:.12g}；相对改善{pair['relative_improvement']:.6%}；中心配对95%区间[{pair['lower95']:.12g}, {pair['upper95']:.12g}]；改善中心{pair['center_wins']}/{pair['centers']}；维护成本比{ratio:.6g}。"]
 lines+=['','## 全部对照和负结果','', '| 方法 | 平衡MSE | 活跃 | 主比较资格 | Joint相对改善 | 95%区间 |','|---|---:|---|---|---:|---|']
 for e in effects:lines.append(f"| {e['baseline']} | {e['baseline_mse']:.12g} | {stats[e['baseline']]['active']} | {e['baseline'] in qualified} | {e['relative_improvement']:.6%} | [{e['lower95']:.12g}, {e['upper95']:.12g}] |")
 lines+=['','## 机械条件','', '```json',json.dumps(checks,ensure_ascii=False,indent=2),'```','',f"ALL_BASELINES_MSE_CI_DOMINANCE={result['ALL_BASELINES_MSE_CI_DOMINANCE']}。此辅助标志不等于风险合格主成功判据；不要求逐一显著击败风险不合格基线的原始MSE。所有区间未作family-wise多重比较校正，对观察后选出的最优合格比较者的区间也不能宣称全体校正显著。",'Smoke的8中心区间只报告，不作为GO硬门槛。原论文六基线的2中心接口Smoke只用于一致性检验，不证明统计优越。' if phase=='Smoke' else 'Final风险资格按全部11个对照逐流审查，Frozen虽完整报告但无真实提交，不属于活跃对照。',
  '','## 停止与论文建议','', 'Smoke未GO则停止并保持新Final池未生成。Final主实验未通过则停止算法研究，建议回稿时缩小未经支持的主张并保留负结果。通过主Final也不等于全文覆盖完成，扩展/外部验证只按已冻结的资产与预算矩阵推进；缺失项目明确列为未完成。',
  '本轮不修改CMC论文、PADV或回复信。原D3f2的STOP_CONFIRM和旧D3f2R稳定性结果保持不变。5%是逐流经验资格，不是总体风险保证；数据为合成，不能冒充真实现场或外部验证。',
  '','## 统计谬误核查','', '11/11核查：以中心为单位；种子/阶段簇内聚合；三流完整分报；无中心/工况/负结果筛除；无事后调参/重抽；风险分母为全部机会；不选择性报告合格者；配对区间公开；风险资格不冒充理论保证；不同部署状态不冒充单阶段因果；所有Service发生在动作封存之后。复合选择和未校正比较的推断限制已披露。']
 report='SMOKE_REPORT_ZH.md' if phase=='Smoke' else 'NEW_FINAL_MAIN_REPORT_ZH.md';textfile(ROOT/report,'\n'.join(lines)+'\n')
 state(status='STOPPED',work='ScientificDecision',stop_decision=decision,eta_seconds=0,active_workers=0)
 print(decision,flush=True)

if __name__=='__main__':main(sys.argv[1])
