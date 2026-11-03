"""Metadata/filename-only global center audit; never opens historical numerical results."""
from r_common import *
import subprocess,re,os

def ids(x):
 if isinstance(x,int) and not isinstance(x,bool):return {x}
 if isinstance(x,str):
  found={int(n) for n in re.findall(r'(?<![A-Za-z0-9])(\d{5})(?![A-Za-z0-9])',x)}
  for a,b in re.findall(r'(\d{5})\s*(?:\.\.|[-—–]|to)\s*(\d{5})',x):
   if int(a)<=int(b):found.update(range(int(a),int(b)+1))
  return found
 if isinstance(x,list):return set().union(*(ids(y) for y in x)) if x else set()
 if isinstance(x,dict):return set().union(*(ids(y) for y in x.values())) if x else set()
 return set()

def main():
 guard();fresh=set(range(62000,62008))|set(range(63000,63032));conflicts=[];metadata={};count=0
 for root in (Path(os.environ.get('PAPER2_SCAN_ROOT',str(ROOT.parent))).resolve(),):
  if not root.exists():continue
  command=['rg','--files','--hidden','-g','!**/.git/**','-g','!**/node_modules/**','-g','!**/_parquet_deps/**',str(root)]
  p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  for line in p.stdout:
   q=Path(line.strip())
   if q.is_relative_to(ROOT):continue
   count+=1;found={int(n) for n in q.parts if n.isdigit()}&fresh
   found|={int(n) for n in re.findall(r'(?:center_|M0_[A-Za-z0-9_]*_)(62\d{3}|63\d{3})(?:_|\.|$)',q.name)}&fresh
   if found:conflicts.append(dict(path=str(q),ids=sorted(found),kind='materialized_path'))
   name=q.name.lower()
   if name in ('historical_id_registry.json','run_manifest.json','data_role_boundaries.json') or name.endswith(('identity_registry.json','center_registry.json')):
    x=read(q);decl={k:v for k,v in x.items() if any(w in k.lower() for w in ('center','pool','reserved','known_old_ids'))};found=ids(decl)
    metadata[str(q)]=dict(sha256=sha(q),ids=sorted(found))
    if found&fresh:conflicts.append(dict(path=str(q),ids=sorted(found&fresh),kind='registered_identity'))
   if count%25000==0:state(work='GlobalCenterIdentity',identity_paths_scanned=count)
  assert p.wait()==0,p.stderr.read()
 verdict=dict(status='PASS' if not conflicts else 'COLLISION_STOP',planned_smoke=list(range(62000,62008)),planned_new_final=list(range(63000,63032)),path_count=count,metadata_count=len(metadata),metadata=metadata,conflicts=conflicts,no_new_data_generated=not any((ROOT/'data').glob('center_*.npz')),no_historical_result_metrics_read=True,at_unix=time.time())
 atomic(ROOT/'GLOBAL_CENTER_PREFLIGHT.json',verdict);assert not conflicts,'Center collision; no replacement'
 print(json.dumps({k:v for k,v in verdict.items() if k!='metadata'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
