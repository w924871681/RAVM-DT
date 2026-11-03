"""Upload-size hard gate. Preserve every included byte, compact without data loss."""
from d3_common import *
import zipfile,gzip,base64
MAX_ZIP_BYTES=512_000_000  # Conservative decimal MB; also below 512 MiB.
def sources(root,dest):
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.is_relative_to(dest) or '__pycache__' in p.parts or 'models' in p.parts or p.name=='PRIVATE_SIMULATION_LABEL_KEY.bin' or '.tmp' in p.suffixes:continue
  yield p
def within_limit(path,limit=MAX_ZIP_BYTES):
 size=Path(path).stat().st_size
 if size>limit:raise ValueError(f'UPLOAD_SIZE_LIMIT: {size} > {limit}; oversized candidate retained locally, never promoted to upload ZIP')
 return size
def build(root,dest,inventory,limit=MAX_ZIP_BYTES):
 root,dest,inventory=Path(root),Path(dest),Path(inventory);archive=dest/'D3d1_EXECUTION_RETURN.zip'
 attempts=[]
 for mode in ('DIRECT_COMPLETE','LOSSLESS_BUNDLED_COMPLETE'):
  candidate=dest/f'D3d1_EXECUTION_RETURN.{mode}.candidate';members=[];bundled=[]
  with zipfile.ZipFile(candidate,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=True) as z:
   bundle=dest/'LOSSLESS_EVIDENCE_BUNDLE.jsonl.gz'
   if mode=='LOSSLESS_BUNDLED_COMPLETE':
    with gzip.open(bundle,'wt',encoding='utf8',compresslevel=9) as f:
     for p in sources(root,dest):
      rel=p.relative_to(root).as_posix()
      if rel.split('/')[0] not in ('science','engineering_test'):continue
      wire=p.read_bytes();rec=dict(path=rel,bytes=len(wire),sha256=hashlib.sha256(wire).hexdigest());bundled.append(rec)
      f.write(json.dumps(dict(**rec,base64=base64.b64encode(wire).decode('ascii')),separators=(',',':'))+'\n')
    z.write(bundle,bundle.name,compress_type=zipfile.ZIP_STORED);members.append(dict(path=bundle.name,bytes=bundle.stat().st_size,sha256=sha(bundle)))
   for p in sources(root,dest):
    rel=p.relative_to(root).as_posix()
    if mode=='LOSSLESS_BUNDLED_COMPLETE' and rel.split('/')[0] in ('science','engineering_test'):continue
    wire=p.read_bytes();z.writestr(rel,wire);members.append(dict(path=rel,bytes=len(wire),sha256=hashlib.sha256(wire).hexdigest()))
   z.write(inventory,inventory.name);members.append(dict(path=inventory.name,bytes=inventory.stat().st_size,sha256=sha(inventory)))
   z.writestr('UPLOAD_SIZE_POLICY.json',canonical(dict(max_zip_bytes=limit,unit='decimal bytes',packaging_mode=mode,no_scientific_result_dropped=True,all_original_evidence_bytes_preserved=True,bundled_file_count=len(bundled),large_CAS_retained_locally=True)))
   z.writestr('RETURN_MEMBER_SHA256.json',canonical(dict(files=members,bundled_files=bundled)))
  attempts.append(dict(mode=mode,bytes=candidate.stat().st_size,within_limit=candidate.stat().st_size<=limit))
  atomic(dest/'ZIP_SIZE_ATTEMPTS.json',dict(max_zip_bytes=limit,attempts=attempts))
  if candidate.stat().st_size>limit:continue
  # Do not expose the designated filename until size, CRC, member set, each hash,
  # and every losslessly bundled original evidence byte have passed.
  with zipfile.ZipFile(candidate) as z:
   assert z.testzip() is None
   assert set(z.namelist())=={m['path'] for m in members}|{'RETURN_MEMBER_SHA256.json','UPLOAD_SIZE_POLICY.json'}
   for m in members:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256'],m['path']
   if bundled:
    expected={v['path']:v for v in bundled};seen=set()
    with z.open('LOSSLESS_EVIDENCE_BUNDLE.jsonl.gz') as compressed,gzip.open(compressed,'rt',encoding='utf8') as f:
     for line in f:
      v=json.loads(line);wire=base64.b64decode(v['base64'],validate=True)
      assert v['path'] not in seen;seen.add(v['path']);assert len(wire)==v['bytes']==expected[v['path']]['bytes']
      assert hashlib.sha256(wire).hexdigest()==v['sha256']==expected[v['path']]['sha256']
    assert seen==set(expected)
  within_limit(candidate,limit);os.replace(candidate,archive);within_limit(archive,limit)
  return dict(archive=archive,zip_bytes=archive.stat().st_size,member_count=len(members)+2,packaging_mode=mode,bundled_file_count=len(bundled),max_zip_bytes=limit,size_limit_pass=True,zip_crc_pass=True,all_member_hashes_pass=True,all_bundled_original_bytes_pass=True)
 atomic(dest/'UPLOAD_BLOCKED_SIZE.json',dict(max_zip_bytes=limit,attempts=attempts,reason='No oversized designated upload ZIP created. All original data and candidate archives retained locally.'))
 raise RuntimeError('UPLOAD_SIZE_LIMIT: both complete lossless candidates exceed cap; see delivery/UPLOAD_BLOCKED_SIZE.json')
