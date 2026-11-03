"""Freeze the explicit correction before any corrected scientific optimizer execution."""
from r_common import *
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import stat

def main():
 guard();assert read(ROOT/'INPUT_CONTRACT_PREFLIGHT.json')['status']=='PASS'
 assert not (ROOT/'D3F2_EXECUTION.lock.json').exists()
 assert not (ROOT/'process/CORRECTION_TRAINING_STARTED.json').exists()
 assert len(list((ROOT/'science/Smoke').glob('*/*/*/*')))==312
 assert not (ROOT/'data/NewFinal_IDENTITY.json').exists()
 for other in ROOT.parent.iterdir():
  if other.is_dir() and other!=ROOT:
   assert not list((other/'data').glob('center_63*.npz')),'Final center conflict: '+str(other)
 code={}
 excluded={'process','progress','data','initial','science','delivery','__pycache__'}
 for p in ROOT.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(ROOT)
  if any(part in excluded for part in rel.parts) or rel.parts[:2]==('engineering','dashboard_v2'):continue
  if p.name in ('COST_ACCOUNTING.json','RUN_PROTOCOL_LOCK.json','RUN_PROTOCOL_SIGNATURE.json','D3F2_EXECUTION.lock.json','CORRECTION_FREEZE_RECEIPT.json'):continue
  assert 'PRIVATE' not in p.name.upper()
  code[rel.as_posix()]=sha(p)
 old=read(ROOT/'reference/FAILED_RUN_PROTOCOL_LOCK.json')
 protocol={**old,'protocol_id':'RAVM_D3F2R_CLEAN_INPUT_CORRECTION_20261021',
  'status':'CORRECTION_FROZEN_BEFORE_CORRECTED_TRAINING_EXISTING_SMOKE_REUSED',
  'run_manifest':read(ROOT/'RUN_MANIFEST.json'),'code_and_static_assets_sha256':code,
  'original_protocol_sha256':sha(ROOT/'reference/FAILED_RUN_PROTOCOL_LOCK.json'),
  'correction_addendum':read(ROOT/'CORRECTION_PROTOCOL.json'),'input_contract_preflight_sha256':sha(ROOT/'INPUT_CONTRACT_PREFLIGHT.json'),
  'frozen_unix':time.time()}
 atomic(ROOT/'RUN_PROTOCOL_LOCK.json',protocol,exclusive=True)
 key=Ed25519PrivateKey.generate()
 atomic(ROOT/'RUN_PROTOCOL_SIGNATURE.json',dict(sha256=sha(ROOT/'RUN_PROTOCOL_LOCK.json'),signature=key.sign((ROOT/'RUN_PROTOCOL_LOCK.json').read_bytes()).hex(),public_key=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()),exclusive=True)
 for name in ('RUN_PROTOCOL_LOCK.json','RUN_PROTOCOL_SIGNATURE.json'):code[name]=sha(ROOT/name)
 atomic(ROOT/'D3F2_EXECUTION.lock.json',dict(protocol_id=protocol['protocol_id'],status='FROZEN',input_protocol_sha256=sha(ROOT/'EXPERIMENT_LOCK.json'),code_sha256=code),exclusive=True)
 for name in ('EXPERIMENT_LOCK.json','RUN_PROTOCOL_LOCK.json','RUN_PROTOCOL_SIGNATURE.json','D3F2_EXECUTION.lock.json'):(ROOT/name).chmod(stat.S_IREAD)
 from d2c_core import AuditLedger
 ledger=AuditLedger(ROOT/'process/DATA_ROLE_EVENTS.jsonl')
 event=ledger.append('CLEAN_INPUT_CORRECTION_DURABLY_FROZEN',('CLEAN_INPUT_CORRECTION','ALL','ALL'),1,dict(execution_sha256=execution_lock_sha(),correction_sgd_budget=24000,existing_smoke_data_reused=True,no_final_data_existed=True))
 atomic(ROOT/'CORRECTION_FREEZE_RECEIPT.json',dict(event_seq=event['seq'],execution_sha256=execution_lock_sha(),at_unix=time.time()),exclusive=True)
 verify_science_lock()
 state(status='READY_FOR_CORRECTION',work='StartupAdmission',error=None,eta_seconds=None)
 print(json.dumps(dict(status='FROZEN_PASS',frozen_files=len(code),execution_sha256=execution_lock_sha(),root=str(ROOT),correction_sgd_budget=24000)),flush=True)

if __name__=='__main__':main()
