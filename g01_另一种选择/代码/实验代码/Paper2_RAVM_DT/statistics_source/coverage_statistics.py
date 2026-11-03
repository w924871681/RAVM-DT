"""Center-cluster statistics from independently validated rows, all signs retained."""
from common import *
import csv,collections,threading,statistics,sys
sys.path.insert(0,str(ROOT/'engineering/throughput'))
from dispatch import execute
from stage_specs import jobs

JOINT='JOINT__l3__m0';METRICS=('mse','harm_frequency','mean_downside','commit_rate','beneficial_recall')
def mean(rows,key):
    values=[x[key]for x in rows if x[key] is not None];return statistics.mean(values) if values else None
def aggregate(rows):
    return {key:mean(rows,key) for key in METRICS}
def group_stats(spec):
    import numpy as np
    from stage_worker import td_path
    group=spec['group'];out=STORAGE/'statistics'/group;out.mkdir(parents=True,exist_ok=True);lock=threading.RLock();stage_keys=['group','center','seed','stream','arm','stage','regime','mse','harmful','downside','commit','beneficial_candidate','beneficial_commit','maintenance_seconds','training_seconds','service_evaluation_seconds','mse_celsius_squared','downside_celsius_squared']
    done=0;planned=jobs(spec);stage_file=out/'STAGE_METRICS.csv'
    state(status='RUNNING',phase=group,work='全部已审计轨迹统计汇总',completed=0,total=len(planned),unit='轨迹')
    with stage_file.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=stage_keys);writer.writeheader()
        def worker(job):
            nonlocal done
            td=td_path(spec,job);audit=td/'AUDIT.json';key=dict(group=group,center=job['center'],seed=job['seed'],stream=job['stream'],arm=job['arm']['id'])
            try:
                assert audit.exists() and read(audit)['status']=='VALIDATED' and read(audit)['execution_lock_sha256']==sha(ROOT/'SCIENCE_EXECUTION.lock.json')
                receipt=read(td/'DONE.json');rows=[];raw=[]
                for t in range(1,spec['stages']+1):
                    p=td/f'result_{t:03}.json';assert sha(p)==receipt['result_sha256'][t-1];r=read(p);scale=spec.get('target_scale')
                    row=dict(**key,stage=t,regime=r['regime_audit_only'],mse=r['deployed_service_mse'],harmful=int(r['harmful']),downside=r['downside'],commit=int(r['commit']),beneficial_candidate=int(r['candidate_beneficial_audit_only']),beneficial_commit=int(r['beneficial_commit']),maintenance_seconds=r['maintenance_wall_seconds'],training_seconds=r['training_seconds'],service_evaluation_seconds=r['post_seal_scientific_evaluation_seconds'],mse_celsius_squared=r['deployed_service_mse']*scale**2 if scale else None,downside_celsius_squared=r['downside']*scale**2 if scale else None)
                    rows.append(row);raw.append(r)
                n=len(rows);benef=sum(x['beneficial_candidate']for x in rows);harm=sum(x['harmful']for x in rows)
                result=dict(status='VALIDATED',**key,regime=rows[0]['regime'],opportunities=n,harmful_count=harm,mse=mean(rows,'mse'),harm_frequency=harm/n,mean_downside=mean(rows,'downside'),commit_rate=mean(rows,'commit'),beneficial_candidates=benef,beneficial_commits=sum(x['beneficial_commit']for x in rows),beneficial_recall=sum(x['beneficial_commit']for x in rows)/benef if benef else None,maintenance_seconds=mean(rows,'maintenance_seconds'),training_seconds=mean(rows,'training_seconds'),service_evaluation_seconds=mean(rows,'service_evaluation_seconds'),mse_celsius_squared=mean(rows,'mse_celsius_squared'),downside_celsius_squared=mean(rows,'downside_celsius_squared'),physical_cluster=spec['physical_cluster'] if spec['domain']!='synthetic' and spec['domain']!='alibaba' else job['center'])
                with lock:writer.writerows(rows);done+=1
                return result
            except Exception as e:
                with lock:done+=1
                return dict(status='UNVERIFIED_NOT_SCIENTIFIC_EVIDENCE',**key,error=str(e))
        results=execute(planned,worker,group+'_STATISTICS',cpu=True,on_tick=lambda z:state(completed=done,active_workers=z['active_workers'],status='RUNNING' if z['active_workers'] else 'WAITING_FOR_RESOURCE'))
    save_csv(out/'TRAJECTORIES.csv',results);valid=[r for r in results if r['status']=='VALIDATED'];center_rows=[];summary=[];pairs=[]
    if spec.get('cost_only'):
        costs=[]
        for job in planned:
            for p in sorted(td_path(spec,job).glob('COST_UNIFIED_*.json')):costs.append(dict(group=group,repetition=spec['repetition'],arm=job['arm']['id'],**read(p)))
        save_csv(out/'UNIFIED_COST.csv',costs)
        atomic(out/'UNIFIED_COST_RECEIPT.json',dict(status='PASS' if len(costs)==len(planned)*spec['stages'] else 'INCOMPLETE',expected=len(planned)*spec['stages'],rows=len(costs)))
    for arm in [a['id']for a in spec['arms']]:
        for stream in spec['stream_ids']:
            selected=[r for r in valid if r['arm']==arm and r['stream']==stream]
            centers=[]
            for center in spec['centers']:
                rr=[r for r in selected if r['center']==center]
                if len(rr)!=len(spec['seeds']):continue
                z=dict(group=group,arm=arm,stream=stream,center=center,regime=rr[0]['regime'],physical_cluster=rr[0]['physical_cluster'],**aggregate(rr));centers.append(z);center_rows.append(z)
            if centers:
                complete_ids={x['center']for x in centers};included=[r for r in selected if r['center']in complete_ids]
                opportunities=sum(r['opportunities']for r in included);harmful=sum(r['harmful_count']for r in included)
                z=dict(group=group,arm=arm,stream=stream,verified_centers=len(centers),expected_centers=len(spec['centers']),seed_count=len(spec['seeds']),opportunities=opportunities,harmful_count=harmful,**aggregate(centers))
                benefit=sum(r['beneficial_candidates']for r in included);commits=sum(r['beneficial_commits']for r in included)
                z.update(beneficial_candidates=benefit,beneficial_commits=commits,beneficial_recall_micro=commits/benefit if benefit else None,beneficial_recall_definition='beneficial_recall: equal-seed/equal-center macro; micro additionally reported with denominator',partial_seed_trajectories_preserved_but_not_in_complete_center_summary=len(selected)-len(included))
                z['source_empirical_risk_pass']=z['harm_frequency']<=.05 and z['mean_downside']<=5.6036475532487204e-5
                z['mse_celsius_squared']=mean(included,'mse_celsius_squared');z['downside_celsius_squared']=mean(included,'downside_celsius_squared');summary.append(z)
    # Equal-seed within center, then equal streams for overall descriptive comparison.
    for arm in [a['id']for a in spec['arms']]:
        for center in spec['centers']:
            rr=[r for r in center_rows if r['arm']==arm and r['center']==center]
            if len(rr)==len(spec['stream_ids']):center_rows.append(dict(group=group,arm=arm,stream='BALANCED_CLEAN_MIXED' if set(spec['stream_ids'])=={'clean','mixed'} else 'EQUAL_CONDITION_MEAN',center=center,regime=rr[0]['regime'],physical_cluster=rr[0]['physical_cluster'],**aggregate(rr)))
    for stream in list(spec['stream_ids'])+['BALANCED_CLEAN_MIXED' if set(spec['stream_ids'])=={'clean','mixed'} else 'EQUAL_CONDITION_MEAN']:
        for baseline in [a['id']for a in spec['arms'] if a['id']!=JOINT]:
            for metric in METRICS:
                left={r['center']:r[metric]for r in center_rows if r['stream']==stream and r['arm']==JOINT and r[metric] is not None};right={r['center']:r[metric]for r in center_rows if r['stream']==stream and r['arm']==baseline and r[metric] is not None}
                ids=sorted(left.keys()&right.keys());delta=np.array([left[k]-right[k]for k in ids]);n=len(ids)
                if not n:continue
                infer=spec['domain'] in ('synthetic','alibaba') and n==len(spec['centers']) and n>2
                if infer:
                    rng=np.random.default_rng(20261022);rep=delta[rng.integers(n,size=(4000,n))].mean(1);lo,hi=np.quantile(rep,[.025,.975])
                else:lo=hi=None
                pairs.append(dict(group=group,stream=stream,baseline=baseline,metric=metric,paired_independent_units=n if spec['domain'] in ('synthetic','alibaba') else 1,delta_joint_minus_baseline=float(delta.mean()),ci_low=float(lo) if lo is not None else None,ci_high=float(hi) if hi is not None else None,bootstrap_replicates=4000 if infer else 0,bootstrap_seed=20261022 if infer else None,uncertainty='unadjusted center/machine paired percentile bootstrap; exploratory across multiple comparisons' if infer else 'descriptive task/house result; no population CI, rooms and seeds not independent buildings'))
    save_csv(out/'CENTERS.csv',center_rows);save_csv(out/'METHOD_STREAM.csv',summary);save_csv(out/'PAIRED_ALL_BASELINES.csv',pairs)
    actual={tuple((r[k]for k in ('center','seed','stream','arm')))for r in valid};expected={(j['center'],j['seed'],j['stream'],j['arm']['id'])for j in planned}
    extras=[str(p)for p in (Path(spec['root'])/'science'/spec['phase']).glob('*/*/*/*') if p.is_dir() and tuple((int(p.parts[-4]),int(p.parts[-3]),p.parts[-2],p.parts[-1])) not in expected]
    cost_ok=not spec.get('cost_only')or read(out/'UNIFIED_COST_RECEIPT.json')['status']=='PASS'
    receipt=dict(status='PASS' if actual==expected and len(valid)==len(expected) and not extras and cost_ok else 'INCOMPLETE',group=group,stage=spec['stage'],domain=spec['domain'],actual_scientific_trajectories=0 if spec.get('cost_only') else len(valid),cost_timing_replicas=len(valid) if spec.get('cost_only') else 0,expected_trajectories=len(expected),validated_stage_rows=len(valid)*spec['stages'],invalid_or_missing=len(expected)-len(valid),unexpected_trajectory_dirs=extras,complete_matrix_identity=actual==expected,unified_cost_complete=cost_ok,statistics_path=str(out),all_baseline_signs_preserved=True,at_unix=time.time())
    atomic(ROOT/'receipts'/f'{group}_STATISTICS.json',receipt);return receipt

def run():
    results=[]
    for p in sorted((ROOT/'registered_groups').glob('*.json')):
        try:results.append(group_stats(read(p)))
        except Exception as e:results.append(dict(group=p.stem,status='BLOCKED_STATISTICS',error=str(e)))
    atomic(ROOT/'STATISTICS_RECEIPT.json',dict(status='PASS' if all(r['status']=='PASS'for r in results)else 'INCOMPLETE',groups=results,at_unix=time.time()))
    return results
if __name__=='__main__':run()
