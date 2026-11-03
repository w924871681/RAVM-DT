"""Arm identities only; every generator uses the unchanged fixed original/v0.2 gate."""
from d3_common import *
D=2.8018237766243602e-5
def arms(phase):
 if phase=='RootScreen':return [dict(id='RAVM_ORIGINAL',method='RAVM_ORIGINAL',generator='U_MSE',gate_method='RAVM_ORIGINAL',threshold=D,fallback=False)]
 manifest=read(ROOT/'RUN_MANIFEST.json')
 base=[dict(id=m+'+V02',method=m+'+V02',generator=m,gate_method='RAVM_D2B_V02',threshold=D,fallback=False) for m in ('U_MSE','U_LABEL_WEIGHT','U_HUBER')]
 base.append(dict(id='RAVM_ORIGINAL',method='RAVM_ORIGINAL',generator='U_MSE',gate_method='RAVM_ORIGINAL',threshold=D,fallback=False))
 if phase=='DevCal':
  base.append(dict(id='GUARD_ONLY',method='GUARD_ONLY',generator='U_MSE',gate_method='RAVM_DELAYED_GUARD_ONLY',threshold=D,fallback=False))
  return base+[dict(id=f'{m}__g{i:02}',method=m,generator='U_MSE',gate_method=m,threshold=g*manifest['baseline_threshold_scale'],fallback=False) for m in ('DUAL_MEAN','RECENT_MEAN','MWU_MEAN') for i,g in enumerate(manifest['baseline_grid'])]
 choices=manifest['pilot_baselines'] if phase=='Pilot' else read(ROOT/'DEVCAL_SELECTED_LOCK.json')['baseline_selection']
 for m,v in choices.items():base.append(dict(id=m if phase=='Pilot' else m+'_SELECTED',method=m,generator='U_MSE',gate_method=m,threshold=v['threshold'] if not v['fallback'] else 0.,fallback=v['fallback']))
 return base
