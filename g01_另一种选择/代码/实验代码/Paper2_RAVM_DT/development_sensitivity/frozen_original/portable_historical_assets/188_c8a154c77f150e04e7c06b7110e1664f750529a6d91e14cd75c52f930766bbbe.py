from d3_common import *
import socket,psutil,shutil
def main():
 protect_originals();assert read(ROOT/'PREFLIGHT_ASSETS.json')['status']=='PASS'
 assert not execution_lock_path().exists() and not list((ROOT/'data').glob('center_*.npz')) and not list((ROOT/'science').rglob('result_*.json'))
 with socket.socket() as s:s.bind(('127.0.0.1',8767))
 assert shutil.disk_usage(ROOT).free>8*1024**3
 for v in read(ROOT/'ASSET_IDENTITY.json')['sources'].values():assert sha(ROOT/v['path'])==v['sha256']
 paths=list(ROOT.glob('*.py'))+[p for folder in ('helpers','runtime','protocol','scripts') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[ROOT/n for n in ('EXPERIMENT_LOCK.json','D3D1_PROTOCOL_LOCK.json','RUN_MANIFEST.json','HISTORICAL_ID_REGISTRY.json','PREFLIGHT_ASSETS.json','ASSET_IDENTITY.json')]
 for p in paths:
  if p.suffix=='.py':compile(p.read_text(encoding='utf8'),str(p),'exec')
 atomic(execution_lock_path(),dict(schema='D3D1_PRE_DATA_EXECUTION_LOCK_V1',input_protocol_sha256=sha(ROOT/'EXPERIMENT_LOCK.json'),code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},frozen_unix=time.time(),no_scientific_data_before_freeze=True,engineering_gpu_acceptance_required_before_scientific_inputs=True,max_lifecycle_trajectories=2616,max_lifecycle_stage_rows=62784,root_reference_trajectories=24,root_reference_stages=576,matched_root_candidate_adaptations=576,oracle_max_adaptations=48,upload_zip_max_bytes=512000000),exclusive=True)
 atomic(ROOT/'RUN_AUTHORIZATION_RECEIPT.json',dict(execution_lock_sha256=execution_lock_sha(),authorization='Direct human task: Screen -> Pilot -> DevCal -> Confirm -> conditional ExpandedDev. Dynamic workers including all audit phases; real-time webpage;15min local progress; no ordinary phase approval',new_final=False,source_retraining=False,created_unix=time.time()),exclusive=True)
 verify_science_lock()
 from d3_supervisor import spawn
 p=spawn('d3_supervisor.py');atomic(ROOT/'process/DETACHED_SUPERVISOR_START.json',dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),execution_lock_sha256=execution_lock_sha(),url='http://127.0.0.1:8767/',status='Background engineering acceptance then conditional scientific execution'))
 print(json.dumps(read(ROOT/'process/DETACHED_SUPERVISOR_START.json'),ensure_ascii=False))
if __name__=='__main__':main()
