"""Pre-data configuration resolution and phase permissions. No numerical outcome imports."""
from d3_common import *
SCALE=.00560364755324872
METHODS=('RECENT_MEAN','MWU_MEAN','DUAL_MEAN')
PHASES=('P0','DevCal','Confirm','ExpandedDev')
def protocol():return read(ROOT/'EXPERIMENT_LOCK.json')
def seeds(phase):return read(ROOT/'RUN_MANIFEST.json')['seeds'][phase]
def arms(phase):
 p=protocol();refs=[dict(id=m,method=m,threshold=0.,generator='U_MSE') for m in ('RAVM_ORIGINAL','RAVM_D2B_V02')]
 if phase=='P0':return [dict(id='JOINT_P0',method='JOINT',weight=.15,threshold=.1*SCALE,generator='U_MSE')]+[dict(id=m+'__g13',method=m,threshold=.1*SCALE,generator='U_MSE') for m in METHODS]+refs
 if phase=='DevCal':
  joint=[dict(id=f'JOINT__l{i}__m{j}',method='JOINT',weight=w,threshold=f*SCALE,generator='U_MSE') for i,w in enumerate(p['joint_mechanism']['lambda_grid']) for j,f in enumerate(p['joint_mechanism']['margin_fraction_grid'])]
  baselines=[dict(id=f'{m}__g{i:02}',method=m,threshold=f*SCALE,generator='U_MSE') for m in METHODS for i,f in enumerate(p['baseline_comparators']['each_baseline_margin_fraction_grid'])]
  return joint+baselines+refs
 if phase in ('Confirm','ExpandedDev'):
  lock=read(ROOT/'DEVCAL_SELECTED_LOCK.json');return [lock['candidate']]+[lock['baseline_selection'][m]['arm_spec'] for m in METHODS]+refs
 raise ValueError('Unknown phase')
def verify_phase_permission(phase):
 verify_science_lock();i=PHASES.index(phase)
 if i==0:
  p=ROOT/'engineering/PREFLIGHT_ACCEPTED.json';assert read(p)['status']=='PASS';return sha(p)
 p=ROOT/f'results/{PHASES[i-1]}/GO_STOP.json';assert read(p)['go'] is True,'Prior phase did not GO'
 if i>=2:assert read(ROOT/'DEVCAL_SELECTED_LOCK.json')['authorized_next']=='Confirm_ONLY_IF_DEVCAL_GO'
 return sha(p)
