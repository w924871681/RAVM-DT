"""Dynamic resource-admitted evidence hashing and visible serial ZIP/verification progress."""
from d3_common import *
from d3_parallel import execute
import zipfile,shutil,threading
def digest_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def discover():
 for base,dirs,files in os.walk(ROOT):
  if Path(base)==ROOT:dirs[:]=[d for d in dirs if d!='delivery']
  dirs[:]=[d for d in dirs if d!='__pycache__']
  for n in files:
   if not n.endswith('.tmp'):yield Path(base)/n
def main():
 protect_originals();verify_science_lock();assert read(ROOT/'progress/progress_state.json')['status'] in ('STOPPED','FAILED');started=time.time();dest=ROOT/'delivery';archive=dest/'D3f2_EXECUTION_RETURN.zip';assert not archive.exists(),'No overwrite of prior archive'
 progress={};mutex=threading.RLock()
 def update(**kw):
  with mutex:progress.update(kw,updated_unix=time.time());atomic(ROOT/'progress/PACKAGING_PROGRESS.json',progress)
 allfiles=list(discover());cas=[p for p in allfiles if 'models' in p.relative_to(ROOT).parts];external=sum(p.stat().st_size for p in cas)>256*1024**2;index=[];index_started=time.monotonic()
 update(phase='INDEXING',completed_files=0,total_files=len(cas) if external else 0,processed_raw_bytes=0,zip_bytes=0,eta_seconds=None,note='科学计算已停止；完整保留本地CAS，大于256MiB时外置并提供身份索引。')
 if external:
  for i,p in enumerate(cas,1):
   index.append(dict(path=p.relative_to(ROOT).as_posix(),absolute_path=str(p),bytes=p.stat().st_size,sha256=digest_file(p)))
   if i%64==0:update(completed_files=i,eta_seconds=(time.monotonic()-index_started)/i*(len(cas)-i))
 atomic(ROOT/'CAS_EXTERNAL_INDEX.json',dict(external=external,threshold_bytes=256*1024**2,rows=index,local_weights_preserved=True,scope='Only stage model CAS may be outside ZIP; source and M0 checkpoints remain included'))
 files=[p for p in discover() if not (external and 'models' in p.relative_to(ROOT).parts)]
 # The packaging telemetry is active and kept separate from the immutable scientific snapshot.
 files=[p for p in files if p.relative_to(ROOT).as_posix() not in ('progress/PACKAGING_PROGRESS.json','process/DELIVERY.stdout.log','process/DELIVERY.stderr.log','process/supervisor.stdout.log','process/supervisor.stderr.log')]
 allbytes=sum(p.stat().st_size for p in files);chunks=[files[i:i+64] for i in range(0,len(files),64)];manifest={};hashed=0;hashbytes=0
 update(phase='HASHING',completed_files=0,total_files=len(files),processed_raw_bytes=0,eta_seconds=None,note='RAM/VRAM与实测吞吐动态准入哈希worker；科学数据与源码保持只读。')
 def job(part):
  nonlocal hashed,hashbytes
  rr={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=digest_file(p)) for p in part}
  with mutex:hashed+=len(part);hashbytes+=sum(r['bytes'] for r in rr.values())
  return rr
 def tick(z):update(completed_files=hashed,processed_raw_bytes=hashbytes,eta_seconds=z['eta_seconds'])
 for rr in execute(chunks,job,'DELIVERY_HASH',tick):manifest.update(rr)
 for p in (ROOT/'progress').iterdir():
  if p.is_file() and p.name!='PACKAGING_PROGRESS.json' and not p.name.endswith('.tmp'):
   n=p.relative_to(ROOT).as_posix();manifest[n]=dict(bytes=p.stat().st_size,sha256=digest_file(p))
 # Worker and progress records can update during hashing. Preserve exact terminal bytes to ZIP.
 filebytes={}
 for n in list(manifest):
  if n.startswith('progress/') or n.startswith('process/'):
   b=(ROOT/n).read_bytes();filebytes[n]=b;manifest[n]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 wire=canonical(manifest)+b'\n';atomic(ROOT/'RETURN_MANIFEST_SHA256.json',manifest)
 count=raw=0;compress_start=time.monotonic();update(phase='COMPRESSING',completed_files=0,total_files=len(manifest)+1,processed_raw_bytes=0,zip_bytes=0,eta_seconds=None,note='ZIP序列化写入；下方体积为真实已写字节，ETA按已处理原始字节吞吐估计。')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for n,r in sorted(manifest.items()):
   b=filebytes[n] if n in filebytes else (ROOT/n).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],n;z.writestr(n,b);count+=1;raw+=len(b)
   if count%32==0:update(completed_files=count,processed_raw_bytes=raw,zip_bytes=z.fp.tell(),eta_seconds=(time.monotonic()-compress_start)/raw*max(0,allbytes-raw) if raw else None)
  z.writestr('RETURN_MANIFEST_SHA256.json',wire)
 update(phase='CRC',completed_files=0,total_files=len(manifest)+1,processed_raw_bytes=raw,zip_bytes=archive.stat().st_size,eta_seconds=None,note='检查ZIP CRC、唯一成员与完整成员集合。')
 extraction=dest/'clean_extraction'
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;assert len(z.namelist())==len(set(z.namelist()));assert set(z.namelist())==set(manifest)|{'RETURN_MANIFEST_SHA256.json'}
  update(phase='EXTRACTING',completed_files=0,total_files=len(manifest)+1,note='正在全新目录解压验证，不覆盖历史交付。');start=time.monotonic()
  for i,n in enumerate(z.namelist(),1):
   assert (extraction/n).resolve().is_relative_to(extraction.resolve());z.extract(n,extraction)
   if i%32==0:update(completed_files=i,eta_seconds=(time.monotonic()-start)/i*(len(manifest)+1-i))
 update(phase='VERIFYING',completed_files=0,total_files=len(manifest),note='逐文件核对干净解压后的大小和SHA256。');start=time.monotonic()
 for i,(n,r) in enumerate(manifest.items(),1):
  p=extraction/n;assert p.stat().st_size==r['bytes'] and digest_file(p)==r['sha256'],n
  if i%32==0:update(completed_files=i,eta_seconds=(time.monotonic()-start)/i*(len(manifest)-i))
 assert {p.relative_to(extraction).as_posix() for p in extraction.rglob('*') if p.is_file()}==set(manifest)|{'RETURN_MANIFEST_SHA256.json'}
 verify_science_lock()
 for n,r in read(ROOT/'ORIGINAL_INPUT_IDENTITIES.json').items():assert sha(ROOT/n)==r['sha256'],n
 receipt=dict(status='PASS',path=str(archive),bytes=archive.stat().st_size,sha256=digest_file(archive),members=len(manifest)+1,crc_pass=True,member_set_pass=True,member_sha256_pass=True,clean_extraction_pass=True,external_cas_files=len(index),external_cas_index_sha256=sha(ROOT/'CAS_EXTERNAL_INDEX.json'),wall_seconds=time.time()-started,stop_decision=read(ROOT/'progress/progress_state.json').get('stop_decision'),finished_unix=time.time())
 atomic(dest/'PACKAGE_VERIFICATION.json',receipt);archive.with_suffix('.zip.sha256').write_text(receipt['sha256']+'  '+archive.name+'\n',encoding='ascii');shutil.copy2(ROOT/'D3f2_EXECUTION_REPORT_ZH.md',dest/'D3f2_EXECUTION_REPORT_ZH.md');update(phase='COMPLETE',completed_files=len(manifest)+1,total_files=len(manifest)+1,eta_seconds=0,zip_bytes=receipt['bytes'],note='CRC、逐文件SHA256、完整成员集合及干净解压验证全部通过。');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
