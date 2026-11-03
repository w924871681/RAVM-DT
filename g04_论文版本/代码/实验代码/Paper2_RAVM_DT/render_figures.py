"""Render the version's paper experiment figures using package-local inputs."""
from pathlib import Path
import argparse,runpy,sys
repo=Path(__file__).resolve().parent
version=repo.parents[2]
kind='g01' if version.name.startswith('g01') else 'g04'
script=repo.parents[1]/'实验图代码'/'render_versions.py'
p=argparse.ArgumentParser(__doc__)
p.add_argument('--data',type=Path,default=version/'输出结果'/'绘图输入')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.argv=[str(script),'--version',kind,'--data',str(a.data),'--output',str(a.output)]
runpy.run_path(str(script),run_name='__main__')
