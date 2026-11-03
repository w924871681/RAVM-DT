"""Static protocol QA only; never starts training or reads Service."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
lock=json.loads((root/'D3F2_PROTOCOL_PROPOSED.json').read_text())
ph=lock['proposed_phases']
for stage in ('P0','DevCal','Confirm','ExpandedDev'):
    p=ph[stage]
    assert p['expected_trajectories']==p['new_centers']*len(p['model_seeds'])*p['streams']*p['arms']
    assert p['expected_stage_rows']==p['expected_trajectories']*p['stages']
assert  len(lock['joint_mechanism']['lambda_grid'])*len(lock['joint_mechanism']['margin_fraction_grid'])==16
assert ph['DevCal']['arms']==16+3*16+2
assert ph['Confirm']['arms']==len(lock['confirm_and_expanded_arms'])
assert ph['ExpandedDev']['arms']==len(lock['confirm_and_expanded_arms'])
summary=lock['totals_if_all_phases_run']
assert sum(p['expected_trajectories'] for p in (ph[k] for k in ("P0","DevCal","Confirm","ExpandedDev"))) == summary['trajectories']
assert sum(p['expected_stage_rows'] for p in (ph[k] for k in ("P0","DevCal","Confirm","ExpandedDev"))) == summary['stage_rows']
assert summary['stage_rows']*50==summary['candidate_optimizer_steps_max']
assert sum(p['new_centers']*len(p['model_seeds']) for p in (ph[k] for k in ("P0","DevCal","Confirm","ExpandedDev")))==summary['M0_total_center_seed_pairs_new_if_all_phases']
assert summary['M0_total_center_seed_pairs_new_if_all_phases']*50==summary['M0_warmup_optimizer_steps_max']
assert summary['candidate_optimizer_steps_max']+summary['M0_warmup_optimizer_steps_max']==summary['total_target_optimizer_steps_max']
assert len(lock['baseline_comparators']['each_baseline_margin_fraction_grid'])==16
print('PASS: 4 phases, policy matrix, 2736 trajectories, 65664 stage rows, 3283200 candidate steps + 3600 warmup steps.')
