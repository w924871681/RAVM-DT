"""Zero-SGD preflight, signed read-only protocol, then launch one local owner."""
from r_common import *
import ast,subprocess,psutil,shutil,stat,collections
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

def main():
 guard();assert not (ROOT/'RUN_PROTOCOL_LOCK.json').exists();assert not any((ROOT/'data').iterdir())
 for p in ROOT.rglob('*.py'):
  if '__pycache__' not in p.parts:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
 assert read(ROOT/'GLOBAL_CENTER_PREFLIGHT.json')['status']=='PASS'
 from d3_engine import SOURCE,Forecast,DEVICE,torch,state_hash
 import core.engine as native,core.models as models
 from legacy_adapter import NATIVE_RUN,NATIVE_FISHER
 from f2_plan import jobs,arms,CORE_IDS,LEGACY_IDS
 torch.set_num_threads(1)
 def forbidden(*a,**kw):raise PermissionError('No optimizer allowed in preflight')
 torch.optim.SGD.step=forbidden
 identity=read(ROOT/'reference/LEGACY_BASELINE_IMPLEMENTATION_IDENTITY.json')
 assert sha(ROOT/'runtime/core/engine.py')==identity['scientific_engine_sha256']
 assert NATIVE_RUN.__code__.co_filename==str(ROOT/'runtime/core/engine.py')
 assert native.adapt is models.adapt and NATIVE_FISHER is models.fisher_diag
 cfg={**native.DEFAULT,**identity['frozen_config']}
 assert (cfg['ewc_lambda'],cfg['ewc_gamma'],cfg['ewc_fisher_examples'],cfg['huber_delta'],cfg['validation_retain_all'])==(10.,.9,32,.1,True)
 assert len(jobs('Smoke'))==336 and len(jobs('NewFinal'))==3456
 assert [x['id'] for x in arms('NewFinal')]==list(CORE_IDS)+list(LEGACY_IDS)
 sources={}
 for seed,spec in SOURCE.items():
  p=ROOT/spec['path'];assert sha(p)==spec['sha256']
  model=Forecast().to(DEVICE);model.load_state_dict(torch.load(p,map_location=DEVICE,weights_only=True));assert state_hash(model)==spec['state_hash'];sources[seed]=dict(**spec,verified=True)
 x=torch.zeros(1,96,24,device=DEVICE);y=torch.zeros(1,1,device=DEVICE)
 for function in (models.adapt,models.fisher_diag):
  try:function(model,models.Batch(x,y,'service'))
  except AssertionError:pass
  else:raise AssertionError('Original training capability accepted Service role')
 atomic(ROOT/'BASELINE_EXACTNESS_PREFLIGHT.json',dict(status='PASS',all_12_ids=[a['id'] for a in arms('NewFinal')],
  native_legacy_run_byte_identical=True,native_engine_sha256=identity['scientific_engine_sha256'],native_adapt_and_fisher_reused_exactly=True,
  legacy_configuration=cfg,service_training_and_fisher_rejected=True,preflight_optimizer_steps=0,
  source_checkpoints=sources,numerical_compatibility_lifecycles_still_required=48,
  measurement_adapter='Uniform float64 difference of the same scalar Service MSE means; preserve native float32 delta/downside, require harm equality and downside deviation<=1e-9; no gate changes',
  actual_nominal_optimizer_bounds=dict(Smoke=404000,NewFinal=4152000),expected_steps_with_frozen_zero=dict(Smoke=394400,NewFinal=3806400),
  extra_fisher_backwards=dict(Smoke=6400,NewFinal=230400),extra_fisher_optimizer_steps=0),exclusive=True)
 # Exact condition definitions and upper bounds are recorded even when ports/asset identities are not ready.
 coverage=[]
 for c in ('stationary_clean','gradual_clean','abrupt_clean','recurring_clean','mixed_all_regimes'):
  coverage.append(dict(item='Original_five_conditions',condition=c,centers='63000-63031',seeds='7311,7312,7313',methods='all_12',extra_sgd_upper=0,status='DERIVE_FROM_MAIN_IF_MAIN_PASS',blocker=''))
 for kind in ('mixed','persistent_bias','replay'):
  for alpha in (.05,.1,.2):coverage.append(dict(item='Nine_severity_conditions',condition=f'{kind}_alpha{alpha}_'+('both' if kind=='mixed' else 'verification_only'),centers='63000-63031',seeds='7311,7312,7313',methods='all_12',extra_sgd_upper=32*3*24*11*50,status='NOT_EXECUTABLE_YET',blocker='Joint full causal port and independent exactness audit for extended condition interface not sealed; do not improvise after observing Main'))
 for backbone in ('MLP','TCN','GRU'):
  for horizon in (1,4):
   maincell=backbone=='GRU' and horizon==1
   coverage.append(dict(item='Architecture_horizon',condition=f'{backbone}_H{horizon}_clean_mixed',centers='63000-63031',seeds='7311,7312,7313',methods='all_12',extra_sgd_upper=0 if maincell else 32*3*2*24*11*50+32*3*50,status='DERIVE_FROM_MAIN' if maincell else 'NOT_EXECUTABLE_YET',blocker='' if maincell else 'Source checkpoint bytes found but cross-architecture identity and H4 causal port exactness not authenticated; no new source pretraining authorized'))
 archive=(Path(__file__).resolve().parents[2]/'raw/alibaba/machine_usage.tar.gz')
 coverage.append(dict(item='Alibaba_external',condition='40_machines_clean_mixed',centers='Original external identity mapping not yet authenticated',seeds='7311,7312,7313',methods='all_12',extra_sgd_upper='UNRESOLVED_EXACT_WINDOW_AND_SHARED_M0_MAPPING',status='NOT_EXECUTABLE_YET',blocker='Local raw archive exists; authenticated 40-machine preprocessing/role mapping and exact target-side bound are not locked; do not read historical external evaluation results for choices'))
 csvwrite(ROOT/'FINAL_COVERAGE_MATRIX.csv',coverage)
 blockers=sorted({x['blocker'] for x in coverage if x['blocker']})
 atomic(ROOT/'FINAL_ASSET_MATRIX.json',dict(main_ready=True,main_methods=12,main_source_checkpoints=sources,coverage_ready=False,
  coverage_blockers=blockers,local_external_archive_exists=archive.is_file(),local_external_archive_bytes=archive.stat().st_size if archive.exists() else None,
  rule='Per supplied protocol, Main Final may complete while paper coverage stays FINAL_COVERAGE_INCOMPLETE. No unvalidated supplements or external-completion claims.',
  exact_supplementary_condition_matrix_sha256=sha(ROOT/'FINAL_COVERAGE_MATRIX.csv')),exclusive=True)
 atomic(ROOT/'ENVIRONMENT_IDENTITY.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=DEVICE,
    logical_cpu_count=os.cpu_count(),ram_bytes=psutil.virtual_memory().total,preflight_available_ram_bytes=psutil.virtual_memory().available,
    user_ram_admission_min_bytes=2*1024**3,no_network_required=True,progress='Latest-state dashboard + causal/scientific records; no 15-minute logger, no scheduled LLM wakeups'),exclusive=True)
 paths=[]
 for p in ROOT.rglob('*'):
  if p.is_file() and not any(v in p.relative_to(ROOT).parts for v in ('process','progress','science','data','initial','delivery','__pycache__')) and 'PRIVATE' not in p.name.upper():paths.append(p)
 code={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
 protocol=dict(protocol_id='RAVM_D3F2R_SMOKE_TO_NEW_FINAL_20261021_V2_EXECUTION',status='FROZEN_BEFORE_ANY_NEW_SCIENTIFIC_DATA',
   proposed_protocol_sha256=sha(ROOT/'input_package/02_SMOKE_FINAL_PROTOCOL_PROPOSED.json'),effective_protocol_sha256=sha(ROOT/'EXPERIMENT_LOCK.json'),
   author_overrides=read(ROOT/'EXPERIMENT_LOCK.json')['runtime_author_overrides'],run_manifest=read(ROOT/'RUN_MANIFEST.json'),
   core_specs=arms('Smoke')[:6],legacy_specs=arms('NewFinal')[6:],legacy_native_configuration=cfg,
   main_final_primary='Lowest center-equal clean/mixed MSE among all 11 risk-qualified and active baselines; publish every contrast',
   code_and_static_assets_sha256=code,role_definition='Source pretraining reused; shared 50-step M0; Update training; Verification gate; Service only after seal',
   no_old_final_metrics_for_selection=True,local_signature='Self-authenticated local execution signature, not third-party timestamp/notary',frozen_unix=time.time())
 atomic(ROOT/'RUN_PROTOCOL_LOCK.json',protocol,exclusive=True)
 key=Ed25519PrivateKey.generate();atomic(ROOT/'RUN_PROTOCOL_SIGNATURE.json',dict(sha256=sha(ROOT/'RUN_PROTOCOL_LOCK.json'),
   signature=key.sign((ROOT/'RUN_PROTOCOL_LOCK.json').read_bytes()).hex(),public_key=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()),exclusive=True)
 code['RUN_PROTOCOL_LOCK.json']=sha(ROOT/'RUN_PROTOCOL_LOCK.json');code['RUN_PROTOCOL_SIGNATURE.json']=sha(ROOT/'RUN_PROTOCOL_SIGNATURE.json')
 atomic(ROOT/'D3F2_EXECUTION.lock.json',dict(protocol_id=protocol['protocol_id'],status='FROZEN',input_protocol_sha256=sha(ROOT/'EXPERIMENT_LOCK.json'),code_sha256=code),exclusive=True)
 for n in ('RUN_PROTOCOL_LOCK.json','RUN_PROTOCOL_SIGNATURE.json','D3F2_EXECUTION.lock.json','EXPERIMENT_LOCK.json'):(ROOT/n).chmod(stat.S_IREAD)
 from d2c_core import AuditLedger
 ledger=AuditLedger(ROOT/'process/DATA_ROLE_EVENTS.jsonl');ledger.verify();event=ledger.append('SMOKE_FINAL_PROTOCOL_DURABLY_FROZEN',('NEW_SMOKE_FINAL_ROLE_CONTROL','ALL','ALL'),1,dict(protocol_sha256=sha(ROOT/'RUN_PROTOCOL_LOCK.json'),execution_sha256=execution_lock_sha(),no_new_data_existed=True))
 atomic(ROOT/'ROLE_FREEZE_RECEIPT.json',dict(event_seq=event['seq'],protocol_sha256=sha(ROOT/'RUN_PROTOCOL_LOCK.json'),sealed_unix=time.time(),no_new_data_before_freeze=True),exclusive=True)
 verify_science_lock();assert not any((ROOT/'data').iterdir())
 with (ROOT/'process/OWNER.stdout.log').open('ab') as out,(ROOT/'process/OWNER.stderr.log').open('ab') as err:
  proc=subprocess.Popen([sys.executable,'-B','-u','-X','utf8',str(ROOT/'owner.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 atomic(ROOT/'process/OWNER_LAUNCH_RECEIPT.json',dict(pid=proc.pid,create_time=psutil.Process(proc.pid).create_time(),local_runner=True,protocol_sha256=sha(ROOT/'RUN_PROTOCOL_LOCK.json'),at_unix=time.time()))
 print(json.dumps(dict(status='FROZEN_AND_OWNER_LAUNCHED',pid=proc.pid,protocol_sha256=sha(ROOT/'RUN_PROTOCOL_LOCK.json'),smoke_total=336,final_total=3456,source_pretraining_steps=0,preflight_optimizer_steps=0),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
