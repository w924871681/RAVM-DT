"""Verify clean identity and exact nonclean transforms without any optimizer calls."""
from d3_engine import *
from r_common import guard
import legacy_adapter as legacy
import inspect,collections

def main():
 guard();torch.set_num_threads(1)
 assert read(ROOT/'BUILD_RECEIPT.json')['status']=='PASS'
 assert not (ROOT/'INPUT_CONTRACT_PREFLIGHT.json').exists()
 def forbidden(*a,**k):raise PermissionError('Input contract preflight cannot execute an optimizer')
 torch.optim.SGD.step=forbidden
 counts=collections.Counter();examples=[]
 origin=Path(read(ROOT/'CORRECTION_PROTOCOL.json')['original_root'])
 original=read(origin/'D3F2_EXECUTION.lock.json')
 assert sha(ROOT/'runtime/core/engine.py')==original['code_sha256']['runtime/core/engine.py']
 assert sha(ROOT/'runtime/core/protocol.py')==original['code_sha256']['runtime/core/protocol.py']
 for center in IDS['Smoke']:
  for stream in STREAMS:
   legacy.TLS.ctx={'stream':stream};timeline=Timeline(center)
   for t in range(1,25):
    if t>1:timeline.record_decision(t-1,False)
    for raw in timeline.maintenance(t):
     alpha=0. if stream=='clean' else .1
     result,mask,source_ts,actual=legacy.native_attack(raw,center,t,stream,alpha,'both',9417)
     expected=observed(raw,center,t,stream)
     assert result.hash==expected.hash,(center,t,stream,raw.role)
     counts[stream+'_role_cases']+=1
     if stream=='clean':
      assert result.hash==raw.hash and not mask.any() and (source_ts==-1).all()
      try:legacy.native_attack(raw,center,t,stream,.1,'both',9417)
      except AssertionError:counts['wrong_clean_alpha_rejected']+=1
      else:raise AssertionError('Clean alpha=.1 was accepted')
     if stream=='mixed' and center in (62000,62004):
      seal=read(ROOT/f'science/Smoke/{center}/7311/mixed/ALWAYS_FT/seal_{t:03}.json')
      assert result.hash==seal['update_hash' if raw.role=='update' else 'verification_hash']
      counts['mixed_matches_preserved_seal']+=1
     if len(examples)<3:examples.append(dict(center=center,stream=stream,stage=t,role=raw.role,raw_hash=raw.hash,corrected_hash=result.hash,changed_rows=int(mask.sum())))
 from correction_meter import check_instrumentation,load_accounting
 assert check_instrumentation()
 calls,steps=load_accounting(ROOT/'process/Smoke_SGD_CALL_ACCOUNTING.jsonl')
 assert steps=={'warmup':800,'candidate':369600}
 assert len(read(ROOT/'CORRECTION_TASKS.json'))==24
 assert not (ROOT/'data/NewFinal_IDENTITY.json').exists()
 for name,h in original['code_sha256'].items():assert sha(origin/name)==h,name
 assert all(counts[x+'_role_cases']==384 for x in STREAMS)
 atomic(ROOT/'INPUT_CONTRACT_PREFLIGHT.json',dict(status='PASS',cases=dict(counts),examples=examples,
  clean_untouched=True,wrong_clean_alpha_rejected=True,mixed_unchanged=True,coherent_sticky_unchanged=True,
  native_engine_byte_identical=True,native_protocol_byte_identical=True,accounting_step_AST_identical=True,
  optimizer_steps=0,retained_optimizer_calls=len(calls),retained_optimizer_steps=steps,original_files_unchanged=True,
  final_pool_unopened=True,at_unix=time.time()),exclusive=True)
 print(json.dumps(read(ROOT/'INPUT_CONTRACT_PREFLIGHT.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
