"""E0: derive from sealed Main/DevCal only; never invoke a model or optimizer."""
from common import *
import ast,csv,itertools,sys,threading,collections
sys.path.insert(0,str(ROOT/'engineering/throughput'))
from dispatch import execute

def run():
    stage('E0','RUNNING');state(status='RUNNING',phase='E0',work='复用主 Final 表格与因果结果',completed=0,total=82944,unit='条阶段记录')
    tables=ROOT/'E0_MAIN_TABLES';fig=ROOT/'E0_FIGURE_SOURCE_DATA';tables.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
    keys=['RESULTS_BY_ARM.csv','RESULTS_BY_CENTER.csv','RESULTS_BY_STREAM.csv','RESULTS_BY_REGIME.csv','RESOURCE_COSTS.csv','RISK_BY_METHOD_STREAM.csv','PAIRED_MSE_ALL_BASELINES.csv','PAIRED_RISK_ALL_BASELINES.csv','PAIRED_CENTER_RAW.csv']
    copied=[]
    def copy(name):
        src=PRIOR/'results/NewFinal'/name;dst=tables/name
        dst.write_bytes(src.read_bytes());assert sha(src)==sha(dst)
        return dict(source=str(src),target=str(dst),sha256=sha(dst))
    copied=execute(keys,copy,'E0_REUSE_TABLE_SHA',cpu=True)
    agg={};joint=[];count=0
    with (PRIOR/'results/NewFinal/STAGE_RECORDS.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            count+=1;k=(row['arm'],row['stream'],row['regime_audit_only'])
            g=agg.setdefault(k,dict(opportunities=0,harmful=0,downside_sum=0.,commits=0,beneficial_commits=0,beneficial_candidates=0,mse_sum=0.,maintenance_seconds=0.))
            g['opportunities']+=1;g['harmful']+=row['harmful']=='True';g['downside_sum']+=float(row['downside']);g['commits']+=row['commit']=='True'
            g['beneficial_commits']+=row['beneficial_commit']=='True';g['beneficial_candidates']+=row['candidate_beneficial_audit_only']=='True'
            g['mse_sum']+=float(row['deployed_service_mse']);g['maintenance_seconds']+=float(row['maintenance_wall_seconds'])
            if row['arm']=='JOINT__l3__m0':joint.append({k:row[k] for k in ('center','seed','stream','stage','commit','parent_state_hash','candidate_state_hash')})
            if count%1024==0:state(completed=count)
    assert count==82944 and len(joint)==6912
    derived=[]
    for (arm,stream,regime),g in sorted(agg.items()):
        n=g['opportunities'];derived.append(dict(arm=arm,stream=stream,regime=regime,**g,
            mse=g['mse_sum']/n,harmful_frequency=g['harmful']/n,downside=g['downside_sum']/n,commit_rate=g['commits']/n,
            beneficial_commit_recall=g['beneficial_commits']/g['beneficial_candidates'] if g['beneficial_candidates'] else None,
            recall_denominator='all candidate-beneficial opportunities on this method own causal state chain; audit-only labels'))
    save_csv(tables/'BENIGN_REGIME_ACTIVITY_RECALL.csv',derived)
    save_csv(fig/'MAIN_REGIME_ACTIVITY_RECALL.csv',derived)
    gatefile=PRIOR/'f2_gate.py';tree=ast.parse(gatefile.read_text(encoding='utf8'));namespace={}
    import math
    namespace.update(isfinite=math.isfinite)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='joint_gate')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(gatefile),'exec'),namespace)
    selection=read(PRIOR/'DEVCAL_SELECTED_LOCK.json');arm=selection['candidate'];decisions=[];done=0;mutex=threading.RLock()
    def same_state(part):
        nonlocal done
        out=[]
        for row in part:
            path=PRIOR/'science/NewFinal'/row['center']/row['seed']/row['stream']/'JOINT__l3__m0'/f'seal_{int(row["stage"]):03}.json'
            seal=read(path);sc=seal['scores'];assert seal['parent_entry']['state_hash']==row['parent_state_hash'] and seal['candidate_entry']['state_hash']==row['candidate_state_hash']
            d=sc['stage3']['recent_mean'];p=sc['stage3']['mwu_p'];scale=sc['parent_recent_verification_mean_loss']
            actual,score=namespace['joint_gate'](d,scale,p,arm['weight'],arm['threshold']);assert actual==seal['commit']==(row['commit']=='True')
            choices={'JOINT__l3__m0':actual}
            for method,spec in selection['baseline_selection'].items():
                a=spec['arm_spec'];value=d<=-a['threshold']
                if method=='MWU_MEAN':value=value and p<.2
                if method=='DUAL_MEAN':value=value and sc['stage3']['history_mean']<=0
                choices[a['id']]=bool(value)
            out.append(dict(**row,d=d,p=p,parent_verification_loss=scale,joint_score=score,verification_hash=seal['verification_hash'],
                **{k+'_on_joint_same_state':v for k,v in choices.items()},scope='frozen rules on identical Joint parent/candidate/verification; no independent deployment or causal effect claim'))
        with mutex:done+=len(part)
        return out
    state(work='同状态合法门控分歧核验',completed=0,total=len(joint),unit='个封存决策')
    def tick(q):state(completed=done,total=len(joint),active_workers=q['active_workers'],eta_seconds=q['eta_seconds'])
    for part in execute([joint[i:i+32] for i in range(0,len(joint),32)],same_state,'E0_SAME_STATE_GATE_AUDIT',on_tick=tick,cpu=True,progress_snapshot=lambda:(done,done)):decisions.extend(part)
    save_csv(tables/'SAME_STATE_FROZEN_GATE_DECISIONS.csv',decisions)
    pair=[]
    cols=[k for k in decisions[0] if k.endswith('_on_joint_same_state')]
    for a,b in itertools.combinations(cols,2):
        pair.append(dict(gate_a=a,gate_b=b,matched_parent_candidate_verification=len(decisions),different=sum(r[a]!=r[b] for r in decisions),
            a_only_commit=sum(r[a] and not r[b] for r in decisions),b_only_commit=sum(r[b] and not r[a] for r in decisions),scope='decision behavior only; not component effect or significance'))
    save_csv(tables/'SAME_STATE_DISAGREEMENT_SUMMARY.csv',pair);save_csv(fig/'JOINT_DECISION_BEHAVIOR.csv',pair)
    dev=ROOT.parent/'experiments/ravm_d3f2_20261020/D3f2_ISOLATED_WORK/results/DevCal'
    sensitivity=[]
    for name in ('RESULTS_BY_ARM.csv','RESULTS_BY_STREAM.csv'):
        path=dev/name
        if path.exists():
            with path.open(encoding='utf-8-sig',newline='') as f:rr=[r for r in csv.DictReader(f) if r.get('arm','').startswith('JOINT__')]
            assert len({r['arm'] for r in rr})==16
            save_csv(tables/('DEVCAL_JOINT_'+name),rr);sensitivity.append(dict(path=str(path),sha256=sha(path),joint_configurations=16))
    atomic(ROOT/'E0_DEV_SENSITIVITY_SOURCE.json',dict(status='PASS' if len(sensitivity)==2 else 'BLOCKED_ASSET',sources=sensitivity,final_reselection=False))
    figuremap=[dict(original_figure='Fig.3',action='REPLACE_WITH_NEW_RESULT',source='E0_MAIN_TABLES/RESULTS_BY_ARM.csv',reason='Original numerical method is historical RAVM, not frozen Joint'),
        dict(original_figure='Fig.5',action='REPLACE_WITH_NEW_RESULT',source='E0_MAIN_TABLES/BENIGN_REGIME_ACTIVITY_RECALL.csv',reason='Use actual Joint regime outcomes and explicit recall denominators'),
        dict(original_figure='Fig.6',action='NO_LONGER_APPLICABLE',source='E0_MAIN_TABLES/SAME_STATE_DISAGREEMENT_SUMMARY.csv',reason='Historical dual-evidence ablation cannot support new Joint Gate; decision behavior only'),
        dict(original_figure='Fig.7',action='REPLACE_WITH_NEW_RESULT',source='E0_MAIN_TABLES/DEVCAL_JOINT_RESULTS_BY_ARM.csv + future E2',reason='Frozen development sensitivity only; architecture/horizon requires actual new experiments')]
    save_csv(fig/'ORIGINAL_FIGURE_ACTION_MAP.csv',figuremap)
    receipt=dict(status='PASS' if len(sensitivity)==2 else 'PARTIAL_BLOCKED_SENSITIVITY',optimizer_steps=0,stage_rows=count,joint_same_state_decisions=len(decisions),
        same_state_joint_exact=True,development_sensitivity=sensitivity,source_table_identities=copied,original_main_zip_sha256='bb7e55c1c40f0a7093cc97b50611b09cd5cebc05fad93b80b4290ff4fee3b6cc',at_unix=time.time())
    atomic(ROOT/'E0_RECEIPT.json',receipt)
    (ROOT/'E0_READONLY_REPORT_ZH.md').write_text('# E0 主Final只读整理\n\n状态：'+receipt['status']+'。优化器步数0；主Final未重跑。\n\n复用82,944条记录、12方法完整正负比较、中心配对区间，新增四类良性工况活动率与beneficial-commit recall。\n\n在Joint的6,912个实际父模型/候选模型/Verification上复算四种已冻结规则，Joint动作逐项一致。此表仅描述合法同状态决策分歧，不把不同部署状态当作同状态反事实，不宣称组件独立因果增益。\n\n敏感性只取冻结DevCal；不得用Final重选参数。各表与来源见E0_MAIN_TABLES、E0_FIGURE_SOURCE_DATA和E0_RECEIPT.json。\n',encoding='utf8')
    stage('E0',receipt['status'],optimizer_steps=0);state(completed=len(joint),total=len(joint),active_workers=0)
if __name__=='__main__':run()
