from pathlib import Path
import argparse,ast,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--version',choices=('g04','g01'),required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=Path(__file__).resolve().with_name('redraw_figures.py')
text=source.read_text('utf8')
if a.version=='g01':text=text.replace('MWU_MEAN__g04','MWU_MEAN__g01')
tree=ast.parse(text)
replacements={'D':str(a.data.resolve(strict=True)),'F':str(a.output.resolve()),'REPORT':str(a.output.resolve())}
changed=[]
for node in tree.body:
 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
  name=node.targets[0].id
  node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(replacements[name])],keywords=[])
  changed.append(name)
assert sorted(changed)==sorted(replacements)
if a.output.exists():raise RuntimeError('Fresh output required')
a.output.mkdir(parents=True)
ast.fix_missing_locations(tree)
exec(compile(tree,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
(a.output/'RENDER_PROVENANCE.json').write_text(json.dumps(dict(version=a.version,original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),arm_projection=a.version=='g01',path_overrides=changed)),encoding='utf8')
