from d3_common import *
PHASES=('Smoke','NewFinal')
CORE_IDS=('JOINT__l3__m0','RECENT_MEAN__g12','MWU_MEAN__g04','DUAL_MEAN__g12','RAVM_ORIGINAL','RAVM_D2B_V02')
LEGACY_IDS=('ALWAYS_FT','ER','EWC','HUBER_FT','VALIDATION_COMMIT','FROZEN')
LEGACY_NAMES=dict(zip(LEGACY_IDS,('Always-FT','ER','EWC','Huber-FT','ValidationCommit','Frozen')))

def arms(phase):
 assert phase in PHASES
 s=read(ROOT/'DEVCAL_SELECTED_LOCK.json')
 core=[s['candidate']]+[s['baseline_selection'][m]['arm_spec'] for m in ('RECENT_MEAN','MWU_MEAN','DUAL_MEAN')]+[dict(id=m,method=m,threshold=0.,generator='U_MSE') for m in ('RAVM_ORIGINAL','RAVM_D2B_V02')]
 return core+[dict(id=m,method=m,threshold=0.,legacy_native_method=LEGACY_NAMES[m],generator='NATIVE_LEGACY') for m in LEGACY_IDS]

def jobs(phase):
 m=read(ROOT/'RUN_MANIFEST.json');out=[]
 for c in m['centers'][phase]:
  for seed in m['seeds'][phase]:
   for stream in ('clean','mixed','coherent_sticky'):
    for arm in arms(phase):
     if phase=='Smoke' and arm['id'] in LEGACY_IDS and not(c in (62000,62004) and stream in ('clean','mixed')):continue
     out.append((c,seed,stream,arm))
 return out

def verify_phase_permission(phase):
 assert phase in PHASES;verify_science_lock()
 assert read(ROOT/'GLOBAL_CENTER_PREFLIGHT.json')['status']=='PASS'
 assert read(ROOT/'BASELINE_EXACTNESS_PREFLIGHT.json')['status']=='PASS'
 if phase=='NewFinal':
  assert read(ROOT/'SMOKE_GATE.json')['decision']=='SMOKE_GO'
  assert read(ROOT/'FINAL_ASSET_MATRIX.json')['main_ready'] is True
  return sha(ROOT/'SMOKE_GATE.json')
 return sha(ROOT/'RUN_PROTOCOL_LOCK.json')
