"""Create separate source (3.11) and lifecycle (3.13) environments.

PyTorch versions/indexes follow https://pytorch.org/get-started/previous-versions/.
No existing user environment is modified. Network access is needed for setup.
"""
import argparse,os,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent
p=argparse.ArgumentParser(__doc__)
p.add_argument('--python311',required=True);p.add_argument('--python313',required=True);p.add_argument('--output',required=True,type=Path)
a=p.parse_args();out=a.output.resolve()
if out.exists():raise RuntimeError('Use a new environment directory')
for name,host,version,torch,index,packages in [
 ('source',a.python311,'3.11.15','2.5.1+cu121','cu121',['numpy==2.4.6','pandas==2.3.3','scipy==1.17.1']),
 ('lifecycle',a.python313,'3.13.5','2.13.0+cu126','cu126',['numpy==2.1.3','scipy==1.15.3','pandas==2.2.3','cryptography==44.0.1','PyYAML==6.0.2','matplotlib==3.10.0','psutil==7.2.2']),
]:
    version_probe='import sys; print(".".join(str(v) for v in sys.version_info[:'+str(len(version.split('.')))+']))'
    actual=subprocess.check_output([host,'-c',version_probe],text=True).strip()
    if actual!=version:raise RuntimeError('Wrong Python version for '+name+': '+actual)
    dest=out/name;cache=out/(name+'_installation_tmp');cache.mkdir(parents=True,exist_ok=True);env=dict(os.environ,TEMP=str(cache),TMP=str(cache))
    subprocess.run([host,'-m','venv','--copies',str(dest)],check=True,env=env)
    python=dest/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    for command in [
      ['install','--no-cache-dir','torch=='+torch,'--index-url','https://download.pytorch.org/whl/'+index],
      ['install','--no-cache-dir',*packages,'--index-url','https://pypi.org/simple'],['check'],
    ]:subprocess.run([str(python),'-m','pip',*command],check=True,env=env)
    print(name,python,flush=True)
