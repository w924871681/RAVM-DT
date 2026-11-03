"""Locked center-equal summaries, paired bootstrap and deterministic GO/STOP."""
from d3_results import *
from f2_plan import arms,seeds,protocol,PHASES,METHODS
import collections
IDS=read(ROOT/'RUN_MANIFEST.json')['centers']

def metrics_for(phase,rows):
 byarm,bystream=phase_summary(phase,rows);specs={a['id']:a for a in arms(phase)}
 for a in byarm:
  streams=[x for x in bystream if x['arm']==a['arm']];a['active']=a['balanced_commit']>0
  a['dev_qualified']=a['feasible'] and next(s['harmful_frequency'] for s in streams if s['stream']=='coherent_sticky')<=.045
  a['weight']=specs[a['arm']].get('weight',0.);a['threshold']=specs[a['arm']]['threshold']
  a['model_cache_bytes']=sum(p.stat().st_size for p in (ROOT/'science'/phase).glob('*/*/*/'+a['arm']+'/models/*') if p.is_file())
 csvwrite(ROOT/f'results/{phase}/RESULTS_BY_ARM.csv',byarm)
 for group in ('center','regime_audit_only'):
  grouped=collections.defaultdict(list)
  for r in rows:grouped[r['arm'],r[group],r['stream']].append(r)
  result=[dict(phase=phase,arm=a,stream=s,**{group:v},**aggregate(rr)) for (a,v,s),rr in sorted(grouped.items())]
  csvwrite(ROOT/f'results/{phase}/RESULTS_BY_{group.upper()}_AND_STREAM.csv',result)
 return byarm,bystream

def compare(phase,rows,candidate,baseline):
 centers=IDS[phase];cs={c:balanced([r for r in rows if r['center']==c and r['arm']==candidate]) for c in centers};bs={c:balanced([r for r in rows if r['center']==c and r['arm']==baseline]) for c in centers}
 delta=np.array([cs[c]-bs[c] for c in centers]);rng=np.random.default_rng(read(ROOT/'RUN_MANIFEST.json')['bootstrap_seed']);idx=rng.integers(len(centers),size=(4000,len(centers)));b=delta[idx].mean(1)
 return dict(candidate=candidate,baseline=baseline,centers=len(centers),center_wins=int(sum(d<0 for d in delta)),center_ties=int(sum(d==0 for d in delta)),paired_mse_delta=float(delta.mean()),lower95=float(np.quantile(b,.025)),upper95=float(np.quantile(b,.975)),relative_improvement=float(1-np.mean(list(cs.values()))/np.mean(list(bs.values()))),bootstrap='4000 paired center clusters; all seeds/streams/stages remain clustered')

def main(phase):
 protect_originals();verify_science_lock();audit=read(ROOT/f'results/{phase}/INDEPENDENT_AUDIT.json');assert audit['status']=='VALIDATED'
 rows=[read(p) for p in sorted((ROOT/'science'/phase).glob('*/*/*/*/result_*.json'))];p=protocol()['proposed_phases'][phase];assert len(rows)==p['expected_stage_rows']
 aa,ss=metrics_for(phase,rows);specs={a['id']:a for a in arms(phase)};lookup={a['arm']:a for a in aa};dest=ROOT/f'results/{phase}'
 def sortkey(a):return (a['balanced_mse'],a['maintenance_wall_seconds'],a.get('weight',0.),-a['threshold'],a['arm'])
 checks=dict(audit_validated=True,complete_trajectories=audit['trajectories']==p['expected_trajectories'],complete_stage_rows=len(rows)==p['expected_stage_rows']);selection=None;pair=None
 if phase=='P0':
  candidate=next(a['id'] for a in arms(phase) if a['method']=='JOINT');base='RECENT_MEAN__g13';joint=lookup[candidate]
  reference={(r['center'],r['seed'],r['stream'],r['stage']):r for r in rows if r['arm']==base}
  differs=sum(r['commit']!=reference[r['center'],r['seed'],r['stream'],r['stage']]['commit'] for r in rows if r['arm']==candidate)
  checks.update(action_difference_count=differs,at_least_one_action_difference=differs>0,smoke_each_stream_harm=max(s['harmful_frequency'] for s in ss if s['arm']==candidate)<=.15,smoke_balanced_mse=joint['balanced_mse']<=1.20*lookup[base]['balanced_mse'])
  go=all(checks[k] for k in ('audit_validated','complete_trajectories','complete_stage_rows','at_least_one_action_difference','smoke_each_stream_harm','smoke_balanced_mse'));stop='STOP_P0'
 elif phase=='DevCal':
  baseline_selection={}
  for method in METHODS:
   pool=[a for a in aa if a['method']==method];eligible=[a for a in pool if a['dev_qualified'] and a['active']];qualified=[a for a in pool if a['dev_qualified']]
   chosen=min(eligible or qualified or pool,key=sortkey)
   baseline_selection[method]=dict(arm_spec=specs[chosen['arm']],selected_grid_arm=chosen['arm'],threshold=chosen['threshold'],fallback=False,qualified_active=bool(eligible),qualified=chosen['dev_qualified'],scope='Eligible active first, else qualified inactive; if none qualified, min-MSE grid arm is descriptive and never eligible as best comparator')
  strong=[a for a in aa if a['method'] in METHODS and a['dev_qualified'] and a['active']];candidates=[a for a in aa if a['method']=='JOINT' and a['weight']>0 and a['dev_qualified'] and a['active']]
  best=min(strong,key=sortkey) if strong else None;chosen=min(candidates,key=sortkey) if candidates else None
  checks.update(qualified_positive_lambda_candidate=chosen is not None,qualified_active_strong_comparator=best is not None,improvement_at_least_0_003=bool(chosen and best and chosen['balanced_mse']<=.997*best['balanced_mse']))
  go=all(checks.values());stop='STOP_DEV_CAL'
  selection=dict(candidate=specs[chosen['arm']] if chosen else None,best_active_strong=best['arm'] if best else None,baseline_selection=baseline_selection,all_qualified_joint_ids=[a['arm'] for a in candidates],best_candidate_metrics=chosen,best_comparator_metrics=best,rule='Risk plus coherent <=4.5%; min balanced MSE; lower measured maintenance cost, smaller lambda, larger margin, ID as final exact tie break. Active means at least one clean/mixed commit.',authorized_next='Confirm_ONLY_IF_DEVCAL_GO',devcal_audit_sha256=sha(dest/'INDEPENDENT_AUDIT.json'),no_confirm_inputs_generated=True,created_unix=time.time())
  atomic(dest/'SELECTION_EVIDENCE.json',selection,exclusive=True)
  if go:
   assert not any((ROOT/f'data/{q}_IDENTITY.json').exists() for q in ('Confirm','ExpandedDev'))
   h=atomic(ROOT/'DEVCAL_SELECTED_LOCK.json',selection,exclusive=True)
   from d2c_core import AuditLedger
   ledger=AuditLedger(ROOT/'process/DATA_ROLE_EVENTS.jsonl');ev=ledger.append('DEVCAL_SELECTION_DURABLY_FROZEN',('D3F2_ROLE_CONTROL','ALL','ALL'),1,dict(selection_sha256=h))
   atomic(ROOT/'SELECTION_SEAL_RECEIPT.json',dict(selection_sha256=h,role_event_seq=ev['seq'],sealed_unix=time.time(),phase_inputs_existed_before_seal=False),exclusive=True)
  candidate=chosen['arm'] if chosen else None;base=best['arm'] if best else None
 else:
  sel=read(ROOT/'DEVCAL_SELECTED_LOCK.json');candidate=sel['candidate']['id'];base=sel['best_active_strong'];joint=lookup[candidate];pair=compare(phase,rows,candidate,base)
  checks.update(candidate_each_stream_risk=joint['feasible'],improvement_at_least_0_002=joint['balanced_mse']<=.998*lookup[base]['balanced_mse'],at_least_four_center_wins=pair['center_wins']>=4,matched_maintenance_cost_at_most_2_5=joint['maintenance_wall_seconds']<=2.5*lookup[base]['maintenance_wall_seconds'])
  go=all(checks.values()) if phase=='Confirm' else False;stop='STOP_CONFIRM' if phase=='Confirm' else 'STOP_AFTER_EXPANDED'
 effects=[]
 if candidate:
  for base_id in [a['arm'] for a in aa if a['arm']!=candidate]:effects.append(compare(phase,rows,candidate,base_id))
 csvwrite(dest/'PAIRED_CENTER_EFFECTS.csv',effects)
 atomic(dest/'GO_STOP.json',dict(phase=phase,go=go,decision=('GO_'+phase if go else stop),checks=checks,candidate=candidate,best_prelocked_or_devcal_strong=base,paired_primary=pair,next_phase=PHASES[PHASES.index(phase)+1] if go else None,at_unix=time.time(),scientific_scope='Synthetic development; no Final',no_retuning=True),exclusive=True)
 print(phase,'GO' if go else stop,flush=True)
if __name__=='__main__':main(sys.argv[1])
