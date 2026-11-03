import unittest,inspect
from f2_gate import joint_gate
from f2_plan import *
from dynamic_worker_resources_v2 import AdaptiveWorkers,GIB
class Acceptance(unittest.TestCase):
 def test_arm_matrix(self):
  self.assertEqual(len(arms('DevCal')),66);self.assertEqual(len([a for a in arms('DevCal') if a['method']=='JOINT']),16)
  for method in METHODS:self.assertEqual(len([a for a in arms('DevCal') if a['method']==method]),16)
 def test_native_single_generator(self):
  self.assertEqual({a['generator'] for a in arms('DevCal')},{'U_MSE'})
 def test_gate_has_no_provider(self):
  self.assertEqual(list(inspect.signature(joint_gate).parameters),['d','parent_mean','p','weight','margin'])
 def test_identity_pool_disjoint(self):
  pools=read(ROOT/'RUN_MANIFEST.json')['centers'];flat=sum(pools.values(),[])
  self.assertEqual(len(flat),len(set(flat)));self.assertFalse(set(flat)&set(range(59000,59032)));self.assertFalse(set(flat)&set(range(47000,47016)))
 def test_phase_permission_denies_future(self):
  for phase in ('DevCal','Confirm','ExpandedDev'):
   with self.assertRaises((FileNotFoundError,AssertionError)):verify_phase_permission(phase)
 def test_worker_memory_blocks(self):
  a=AdaptiveWorkers();cap,s=a.allowed(0,0,dict(available_ram_bytes=0,total_ram_bytes=16*GIB,free_vram_bytes=6*GIB,total_vram_bytes=6*GIB,logical_cpu_count=16));self.assertEqual(cap,0)
 def test_worker_recovers(self):
  a=AdaptiveWorkers();s=dict(available_ram_bytes=0,total_ram_bytes=16*GIB,free_vram_bytes=6*GIB,total_vram_bytes=6*GIB,logical_cpu_count=16);a.allowed(0,0,s);s['available_ram_bytes']=8*GIB;cap,_=a.allowed(0,0,s);self.assertGreater(cap,0)
if __name__=='__main__':unittest.main()
