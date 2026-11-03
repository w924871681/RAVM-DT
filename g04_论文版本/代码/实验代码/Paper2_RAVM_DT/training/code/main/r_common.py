from d3_common import *
import csv,threading

def guard():
 def hook(event,args):
  if event=='open':
   p,m,flags=args
   writing=(isinstance(m,str) and any(c in m for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
   if writing and isinstance(p,(str,bytes,os.PathLike)) and os.fsdecode(p).lower()!=os.devnull.lower() and not Path(os.fsdecode(p)).resolve().is_relative_to(ROOT):raise PermissionError('Write outside new isolated run: '+str(p))
  if event in ('os.remove','os.rmdir','os.rename'):
   for p in args[:2] if event=='os.rename' else args[:1]:
    if isinstance(p,(str,bytes,os.PathLike)) and not Path(os.fsdecode(p)).resolve().is_relative_to(ROOT):raise PermissionError('Mutation outside isolated run')
 sys.addaudithook(hook)

def state(**kw):
 p=ROOT/'progress/progress_state.json';x=read(p);x.update(kw,updated_unix=time.time());atomic(p,x);return x

def freeze_check():return verify_science_lock()

def csvwrite(p,rows):
 from d3_results import csvwrite as write
 Path(p).parent.mkdir(parents=True,exist_ok=True);return write(p,rows)

def textfile(p,s):Path(p).write_text(s,encoding='utf8')
