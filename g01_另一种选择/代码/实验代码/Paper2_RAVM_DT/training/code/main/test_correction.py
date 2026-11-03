"""Technical checks using a fake optimizer; zero scientific training."""
from r_common import *
from correction_meter import RecoveryMeter,check_instrumentation,call_key
import types,tempfile,unittest

class CorrectionChecks(unittest.TestCase):
 def test_only_declared_bad_clean_cells_are_replaced(self):
  tasks=read(ROOT/'CORRECTION_TASKS.json')
  self.assertEqual(len(tasks),24)
  self.assertEqual({x['stream'] for x in tasks},{'clean'})
  self.assertEqual({x['center'] for x in tasks},{62000,62004})
  self.assertEqual({x['seed'] for x in tasks},{7311,7312})
  self.assertEqual(sum(24*50 for x in tasks if x['arm']!='FROZEN'),24000)

 def test_tight_correction_cap_rejects_before_optimizer_call(self):
  calls=[]
  api=types.SimpleNamespace(optim=types.SimpleNamespace(SGD=types.SimpleNamespace(step=lambda *a,**k:calls.append(1))))
  meter=RecoveryMeter(api,'Smoke',set(),{'warmup':800,'candidate':393550})
  meter.total_limit=394400;meter.limits['candidate']=393600
  with tempfile.TemporaryDirectory(dir=ROOT/'process/tmp') as temp:
   meter.path=Path(temp)/'ledger.jsonl'
   with meter.scope('candidate',dict(center=62000,seed=7311,stream='clean',arm='ALWAYS_FT')):
    for _ in range(50):api.optim.SGD.step(None)
    with self.assertRaises(AssertionError):api.optim.SGD.step(None)
  self.assertEqual(len(calls),50)
  self.assertEqual(sum(meter.done.values()),394400)

 def test_reused_identity_cannot_be_retrained(self):
  identity=dict(center=62000,seed=7311,stream='mixed',arm='ALWAYS_FT')
  key=call_key('candidate',dict(**identity,phase='Smoke',stage=1));calls=[]
  api=types.SimpleNamespace(optim=types.SimpleNamespace(SGD=types.SimpleNamespace(step=lambda *a,**k:calls.append(1))))
  meter=RecoveryMeter(api,'Smoke',{key},{'warmup':800,'candidate':369600})
  with meter.scope('candidate',identity):
   with self.assertRaises(AssertionError):api.optim.SGD.step(None)
  self.assertEqual(calls,[])

if __name__=='__main__':
 assert check_instrumentation()
 unittest.main(verbosity=2)
