from r_common import *
from r_parallel import execute

def verify_reuse():
    files=list(read(ROOT/'REUSED_FILES_SHA256.json').items())
    def part(batch):
        for rel,h in batch:assert sha(ROOT/rel)==h,'Reused evidence changed: '+rel
        return len(batch)
    result=execute([files[i:i+128] for i in range(0,len(files),128)],part,'REUSE_ASSET_SHA',cpu=True)
    assert sum(result)==len(files)
    prefix=(ROOT/'reference/RETAINED_LEDGER_PREFIX.jsonl').read_bytes()
    with (ROOT/'process/Smoke_SGD_CALL_ACCOUNTING.jsonl').open('rb') as f:assert f.read(len(prefix))==prefix
