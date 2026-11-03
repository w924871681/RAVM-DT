"""Local dynamic hash/deflate/CRC/SHA; model bytes remain in explicitly indexed CAS."""
from r_common import *
from r_parallel import execute
from safe_paths import prepare_paths,validate_infos
import zipfile,zlib,threading

def insert(z,info,data):
 info.header_offset=z.fp.tell();z._writecheck(info);z._didModify=True;z.fp.write(info.FileHeader(False));z.fp.write(data);z.filelist.append(info);z.NameToInfo[info.filename]=info;z.start_dir=z.fp.tell()

def main():
 guard();freeze_check();started=time.time();dest=ROOT/'delivery';phase='NewFinal' if (ROOT/'NEW_FINAL_MAIN_GATE.json').exists() else 'Smoke';name='NEW_FINAL_EXPERIMENT_RETURN.zip' if phase=='NewFinal' else 'SMOKE_ONLY_RETURN.zip'
 archive=dest/name;assert not archive.exists();state(status='PACKAGING',work='Packaging',current_role='DYNAMIC_LOCAL_DELIVERY',error=None,eta_seconds=None)
 mutex=threading.RLock();pstate={};done=0
 def update(**kw):
  with mutex:pstate.update(kw,updated_unix=time.time());atomic(ROOT/'progress/PACKAGING_PROGRESS.json',pstate)
 weights=[];files=[]
 for base,dirs,names in os.walk(ROOT):
  if Path(base)==ROOT:dirs[:]=[d for d in dirs if d!='delivery']
  dirs[:]=[d for d in dirs if d!='__pycache__']
  for n in names:
   p=Path(base)/n;rel=p.relative_to(ROOT).as_posix()
   if 'PRIVATE' in n.upper() or n.endswith('.tmp'):continue
   if 'models' in p.relative_to(ROOT).parts:weights.append((rel,p))
   elif rel.startswith(('progress/','process/')) and p.suffix not in ('.json','.jsonl','.log','.md'):continue
   else:files.append((rel,p))
 update(phase='HASHING_EXTERNAL_MODEL_INDEX',completed_files=0,total_files=len(weights),zip_bytes=0,eta_seconds=None)
 def weight(part):
  nonlocal done
  rr=[]
  for rel,p in part:
   h=sha(p);assert p.name==h;rr.append(dict(path=rel,sha256=h,bytes=p.stat().st_size,weight_bytes_in_zip=False))
  with mutex:done+=len(part)
  return rr
 index=[]
 for part in execute([weights[i:i+128] for i in range(0,len(weights),128)],weight,'DELIVERY_CAS_SHA',lambda q:update(completed_files=done,eta_seconds=q['eta_seconds']),cpu=True):index.extend(part)
 cas=dict(scope='All model bytes retained under local scientific root; this ZIP carries exact external CAS references, not these weight bytes',root=str(ROOT),original_model_files=len(index),original_model_bytes=sum(x['bytes'] for x in index),all_indexed_files_sha256_verified=True,rows=index)
 atomic(ROOT/'CAS_EXTERNAL_INDEX.json',cas);files=[(n,p) for n,p in files if n!='CAS_EXTERNAL_INDEX.json']+[('CAS_EXTERNAL_INDEX.json',ROOT/'CAS_EXTERNAL_INDEX.json')]
 snapshots={n:p.read_bytes() for n,p in files if n.startswith(('progress/','process/'))}
 manifest={};compressed=0
 update(phase='LOCAL_PARALLEL_DEFLATE',completed_files=0,total_files=len(files),eta_seconds=None,note='动态本地CPU压缩；大模型字节外置且全部索引。')
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
  def pack(part):
   nonlocal compressed
   for n,p in part:
    b=snapshots[n] if n in snapshots else p.read_bytes();h=hashlib.sha256(b).hexdigest();co=zlib.compressobj(6,zlib.DEFLATED,-15);data=co.compress(b)+co.flush();info=zipfile.ZipInfo(n);info.compress_type=zipfile.ZIP_DEFLATED;info.file_size=len(b);info.compress_size=len(data);info.CRC=zlib.crc32(b)&0xffffffff;info.external_attr=0o600<<16
    with mutex:manifest[n]=dict(bytes=len(b),sha256=h);insert(z,info,data);compressed+=1
   return len(part)
  execute([files[i:i+32] for i in range(0,len(files),32)],pack,'DELIVERY_PARALLEL_DEFLATE',lambda q:update(completed_files=compressed,eta_seconds=q['eta_seconds'],zip_bytes=z.fp.tell()),cpu=True)
  wire=canonical(manifest);z.writestr('RETURN_MANIFEST_SHA256.json',wire)
 assert archive.stat().st_size<512_000_000,'Archive exceeds requested upload bound; preserve closed archive, do not assert delivery complete'
 update(phase='CLEAN_EXTRACTION_CRC_SHA',completed_files=0,total_files=len(manifest)+1,zip_bytes=archive.stat().st_size,eta_seconds=None)
 expected={**manifest,'RETURN_MANIFEST_SHA256.json':dict(bytes=len(wire),sha256=hashlib.sha256(wire).hexdigest())};count=0
 with zipfile.ZipFile(archive) as z:
  names=validate_infos(z.infolist());assert set(names)==set(expected);root,targets=prepare_paths(dest/(phase+'_RETURN_CLEAN_EXTRACTION'),names)
  def extract(part):
   nonlocal count
   for n in part:
    b=z.read(n);r=expected[n];assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],n
    with targets[n].open('xb') as f:f.write(b)
    assert sha(targets[n])==r['sha256'],n
    with mutex:count+=1
   return len(part)
  execute([names[i:i+128] for i in range(0,len(names),128)],extract,'DELIVERY_CRC_EXTRACT_SHA',lambda q:update(completed_files=count,eta_seconds=q['eta_seconds']),cpu=True)
 assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(expected)
 freeze_check();receipt=dict(status='PASS',path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=len(names),crc_pass=True,all_member_sha256_pass=True,exact_member_set_pass=True,clean_extraction_pass=True,
   model_bytes_in_zip=False,external_model_index='CAS_EXTERNAL_INDEX.json',external_model_files=len(index),all_external_model_sha256_verified=True,local_only=True,network_required=False,under_512000000_bytes=True,wall_seconds=time.time()-started,scientific_decision=read(ROOT/'progress/progress_state.json').get('stop_decision'))
 atomic(dest/'PACKAGE_VERIFICATION.json',receipt);archive.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='ascii')
 state(status='STOPPED',work='Complete',current_role='DELIVERY_COMPLETE',active_workers=0,eta_seconds=0);update(phase='COMPLETE',completed_files=len(names),total_files=len(names),zip_bytes=receipt['bytes'],eta_seconds=0)
 print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
