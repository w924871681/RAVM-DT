import os,sys,json,hashlib,time,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
sys.dont_write_bytecode=True
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def _replace_retry(tmp,p):
 # Windows readers may temporarily omit FILE_SHARE_DELETE. Retry only the same
 # already-fsynced bytes; never re-execute a model or decision for an IO retry.
 for attempt in range(50):
  try:os.replace(tmp,p);return
  except PermissionError as ex:
   if os.name!='nt' or getattr(ex,'winerror',None) not in (5,32,33) or attempt==49:raise
   log=ROOT/'process/ATOMIC_REPLACE_RETRIES.jsonl';log.parent.mkdir(exist_ok=True)
   with log.open('a',encoding='utf8') as f:f.write(json.dumps(dict(at_unix=time.time(),pid=os.getpid(),path=str(p),attempt=attempt+1,winerror=ex.winerror))+'\n')
   time.sleep(min(.01*(attempt+1),.2))
def atomic_text(p,text):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf8') as f:f.write(text);f.flush();os.fsync(f.fileno())
  _replace_retry(tmp,p)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def atomic(p,x,exclusive=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);wire=canonical(x)+b'\n'
 if exclusive:
  fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
  try:
   with os.fdopen(fd,'wb') as f:f.write(wire);f.flush();os.fsync(f.fileno())
   os.link(tmp,p) # atomic create-if-absent; never exposes a partial sealed JSON
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
 else:
  fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
  try:
   with os.fdopen(fd,'wb') as f:f.write(wire);f.flush();os.fsync(f.fileno())
   _replace_retry(tmp,p)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
 return hashlib.sha256(wire).hexdigest()
def protect_originals():
 roots=(ROOT.parent,)
 def hook(event,args):
  if event=='open':
   p,m,flags=args
   writing=(isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
   if writing and isinstance(p,(str,bytes,os.PathLike)):
    resolved=Path(os.fsdecode(p)).resolve()
    if any(resolved.is_relative_to(r) for r in roots) and not resolved.is_relative_to(ROOT):raise PermissionError('Frozen original write rejected: '+str(resolved))
  if event in ('os.remove','os.rmdir','os.rename'):
   paths=args[:2] if event=='os.rename' else args[:1]
   for p in paths:
    if isinstance(p,(str,bytes,os.PathLike)):
     resolved=Path(os.fsdecode(p)).resolve()
     if any(resolved.is_relative_to(r) for r in roots) and not resolved.is_relative_to(ROOT):raise PermissionError('Frozen original mutation rejected: '+str(resolved))
 sys.addaudithook(hook)
def execution_lock_path():return ROOT/'D3F2_EXECUTION.lock.json'
def execution_lock_sha():return sha(execution_lock_path())
def authorization_receipt_path():return ROOT/'RUN_AUTHORIZATION_RECEIPT.json'
def allowed_execution_shas():return {execution_lock_sha()}
def verify_science_lock():
 lock=read(execution_lock_path());assert sha(ROOT/'EXPERIMENT_LOCK.json')==lock['input_protocol_sha256']
 for rel,h in lock['code_sha256'].items():assert sha(ROOT/rel)==h,rel
 for s,rec in read(ROOT/'ASSET_IDENTITY.json')['sources'].items():assert sha(ROOT/rec['path'])==rec['sha256']
 return lock
