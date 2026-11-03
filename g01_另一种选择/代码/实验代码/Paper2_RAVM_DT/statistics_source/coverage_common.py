"""Durable engineering state; no scientific code imported on dashboard access."""
from pathlib import Path
import hashlib,json,os,tempfile,time,threading
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'experiments/D3f2R_CLEANFIX_20261021'
STORAGE=Path(__file__).resolve().parent
LOCK=threading.RLock()
_LAST_RATE_SAMPLE=(None,0.)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',dir=p.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True);f.flush();os.fsync(f.fileno())
        for attempt in range(50):
            try:os.replace(tmp,p);break
            except PermissionError:
                if attempt==49:raise
                time.sleep(.02)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def state(**kw):
    global _LAST_RATE_SAMPLE
    with LOCK:
        p=ROOT/'progress/progress_state.json';s=read(p) if p.exists() else {};s.update(kw,updated_unix=time.time());atomic(p,s)
        key=(s.get('phase'),s.get('work'),s.get('unit'))
        if 'completed' in s and (key!=_LAST_RATE_SAMPLE[0] or s['updated_unix']-_LAST_RATE_SAMPLE[1]>=1):
            record=dict(at=s['updated_unix'],phase=key[0],work=key[1],unit=key[2],completed=s['completed'],total=s.get('total',0))
            with (ROOT/'progress/RATE_SAMPLES.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            _LAST_RATE_SAMPLE=(key,s['updated_unix'])
    return s
def stage(name,status,**kw):
    with LOCK:
        p=ROOT/'progress/STAGES.json';s=read(p) if p.exists() else {};s[name]=dict(status=status,updated_unix=time.time(),**kw);atomic(p,s)
def save_csv(path,rows):
    import csv
    rows=list(rows);p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
