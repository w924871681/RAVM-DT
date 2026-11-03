from d3_common import *
import unittest,copy,numpy as np
import d3_engine as en
from d3_adapt import *
class Tests(unittest.TestCase):
 def test_reference_math(self):
  import subprocess
  p=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'helpers/test_label_weight_reference.py')],capture_output=True,text=True,encoding='utf8');atomic_text(ROOT/'engineering_test/REFERENCE_TESTS.log',p.stdout+p.stderr);self.assertEqual(p.returncode,0)
 def test_role_and_broadcast_and_gradient(self):
  u=en.Segment(np.zeros((400,24),dtype='float32'),np.full(400,.4,dtype='float32'),np.arange(400),'verification')
  with self.assertRaises(PermissionError):target_weights(u)
  u.role='update';w,raw=target_weights(u);self.assertEqual(len(w),304);self.assertTrue(np.all(w==1))
  pred=torch.tensor([[.1],[.2],[.7]],device=en.DEVICE,requires_grad=True);target=torch.tensor([[.4],[.3],[.8]],device=en.DEVICE);weights=torch.ones(3,device=en.DEVICE)
  a=torch.nn.functional.mse_loss(pred,target);ga=torch.autograd.grad(a,pred,retain_graph=True)[0];bb=(weights[:,None]*(pred-target).square()).sum()/weights.sum();gb=torch.autograd.grad(bb,pred)[0]
  self.assertTrue(torch.equal(ga,gb));self.assertEqual(a.item(),bb.item())
 def test_old_gpu_exact_mse_and_ones(self):
  old=Path(read(ROOT/'PREFLIGHT_ASSETS.json')['prior_root']);td=old/'science/DevCal/56600/7311/clean/RAVM_ORIGINAL';s=read(td/'seal_001.json');cas=en.CAS.__new__(en.CAS);cas.root=td/'models';parent=cas.load(s['parent_entry'])
  with np.load(old/'data/center_56600.npz') as z:a,b=en.spans(1)[0];u=en.Segment(z['x'][a:b].copy(),z['y'][a:b].copy(),z['ts'][a:b].copy(),'update')
  self.assertEqual(u.hash,s['update_hash']);xy=en.tensors(en.windows(u));seed=s['training_seed'];m,meta=make_candidate(parent,u,'U_MSE',seed);ones,_=weighted_adapter(parent,xy,np.ones(len(xy[0]),dtype='float32'),seed)
  archived=cas.load(s['candidate_entry']);archive_parameter_abs=max(float((m.state_dict()[k]-archived.state_dict()[k]).abs().max()) for k in m.state_dict())
  maxabs=max(float((m.state_dict()[k]-ones.state_dict()[k]).abs().max()) for k in m.state_dict());archive_exact=en.state_hash(m)==s['candidate_entry']['state_hash'];score_abs=float(np.max(np.abs(en.errors(m,xy)[0]-en.errors(ones,xy)[0])))
  with np.load(old/'data/center_56600.npz') as z:
   a,b=en.spans(1)[1];v=en.Segment(z['x'][a:b].copy(),z['y'][a:b].copy(),z['ts'][a:b].copy(),'verification');h=en.Segment(z['x'][800:1000].copy(),z['y'][800:1000].copy(),z['ts'][800:1000].copy(),'verification')
  gate=en.OriginalCMCGate(ROOT/'runtime',ROOT/'protocol/frozen_v2_final_commit_config.yaml');d=gate.evaluate(m,parent,en.evidence_blocks(v),en.evidence_blocks(h));do=gate.evaluate(ones,parent,en.evidence_blocks(v),en.evidence_blocks(h));da=gate.evaluate(archived,parent,en.evidence_blocks(v),en.evidence_blocks(h))
  gate_diff=int(d.commit!=do.commit);archive_gate_diff=int(d.commit!=da.commit);self.assertEqual(gate_diff,0)
  self.assertLessEqual(maxabs,1e-5);self.assertLessEqual(score_abs,1e-6)
  self.assertEqual(en.state_hash(parent),s['parent_entry']['state_hash'])
  atomic(ROOT/'engineering_test/GPU_MSE_ONES_REGRESSION.json',dict(passed=True,source='Old observed DevCal clean stage1 regression only; never GO',parent_state_exact=True,mse_candidate_matches_archive_bitwise=archive_exact,weighted_ones_matches_mse_bitwise=en.state_hash(m)==en.state_hash(ones),max_parameter_abs_error=maxabs,max_update_error_abs_diff=score_abs,archive_max_parameter_abs_error=archive_parameter_abs,weighted_vs_mse_gate_action_differences=gate_diff,archive_gate_action_differences=archive_gate_diff,minibatch_digest=meta['sampled_indices_sha256'],frozen_steps=50))
 def test_root_24_stage_seal_and_independent_audit_fixture(self):
  import d3_trajectory as tr,d3_screen as scr,d3_audit_new as au,core.protocol as cp,shutil
  roots=(en.ROOT,tr.ROOT,scr.ROOT,au.ROOT,cp.ROOT);fixture=ROOT/'engineering_test'/('root_fixture_'+str(time.time_ns()));fixture.mkdir()
  for d in ('data','initial','protocol'):(fixture/d).mkdir()
  shutil.copyfile(ROOT/'protocol/frozen_v2_final_commit_config.yaml',fixture/'protocol/frozen_v2_final_commit_config.yaml')
  source=ROOT/en.SOURCE['7311']['path'];(fixture/'checkpoints').mkdir();shutil.copyfile(source,fixture/en.SOURCE['7311']['path'])
  rng=np.random.default_rng(20261020);x=rng.uniform(.1,.9,(17000,24)).astype('float32');x[:,12:]=1.;y=(.4+.02*np.sin(np.arange(17000)/250)).astype('float32');ts=np.arange(17000,dtype='int64');np.savez_compressed(fixture/'data/center_900020.npz',x=x,y=y,ts=ts,meta=json.dumps(dict(pool='ENGINEERING_ONLY')))
  realgate=tr.OriginalCMCGate;oldids=en.IDS['RootScreen'][:];old_sha=(tr.execution_lock_sha,tr.allowed_execution_shas,scr.execution_lock_sha,au.execution_lock_sha)
  en.ROOT=tr.ROOT=scr.ROOT=au.ROOT=cp.ROOT=fixture;en.IDS['RootScreen'].append(900020);tr.execution_lock_sha=scr.execution_lock_sha=au.execution_lock_sha=lambda:'e'*64;tr.allowed_execution_shas=lambda:{'e'*64};tr.OriginalCMCGate=au.OriginalCMCGate=lambda _,cfg:realgate(ROOT/'runtime',cfg)
  try:
   model=en.initial(900020,7311);del model
   from d3_runner import arms
   rows=tr.run_trajectory('RootScreen',900020,7311,'clean',arms('RootScreen')[0],'ENGINEERING_FIXTURE',stop_after=5);td=fixture/'science/RootScreen/900020/7311/clean/RAVM_ORIGINAL';before=[sha(td/f'result_{t:03}.json') for t in range(1,6)]
   rows=tr.run_trajectory('RootScreen',900020,7311,'clean',arms('RootScreen')[0],'ENGINEERING_FIXTURE');self.assertEqual(before,[sha(td/f'result_{t:03}.json') for t in range(1,6)]);self.assertEqual(len(rows),24)
   part=au.audit_td(('RootScreen',td));self.assertEqual(len(part['records']),24);self.assertEqual(len(part['local']),24)
   with self.assertRaises(PermissionError):en.CausalProvider(900020,'clean').service(1,None,en.AuditLedger(td/'journal.jsonl'))
   atomic(ROOT/'engineering_test/ROOT_LIFECYCLE_FIXTURE.json',dict(passed=True,reference_stage_rows=24,matched_candidate_adaptations=24,resume_preserved=True,all_candidates_sealed_before_Service=True,full_original_block_score_and_weight_audit=True,scope='Random engineering data only, not scientific RootScreen'))
  finally:
   en.ROOT,tr.ROOT,scr.ROOT,au.ROOT,cp.ROOT=roots;en.IDS['RootScreen']=oldids;tr.execution_lock_sha,tr.allowed_execution_shas,scr.execution_lock_sha,au.execution_lock_sha=old_sha;tr.OriginalCMCGate=au.OriginalCMCGate=realgate
 def test_formula_is_update_only(self):
  import inspect,ast
  self.assertEqual(list(inspect.signature(target_weights).parameters),['update'])
  tree=ast.parse((ROOT/'d3_adapt.py').read_text(encoding='utf8'));imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)];self.assertNotIn('d3_engine',imports)
 def test_phase_counts_and_original_gates(self):
  from d3_runner import arms
  self.assertEqual(len(arms('RootScreen')),1);self.assertEqual(len(arms('Pilot')),7);self.assertEqual(len(arms('DevCal')),53)
  for a in arms('DevCal'):
   if a['id'].endswith('+V02'):self.assertEqual(a['gate_method'],'RAVM_D2B_V02');self.assertEqual(a['threshold'],2.8018237766243602e-5)
  from d3_pipeline import verify_phase_permission
  # Before code freeze no permission exists for any future scientific input pool.
  self.assertFalse(list((ROOT/'data').glob('center_*.npz')))
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));atomic(ROOT/'engineering_test/ENGINEERING_QA.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,errors=len(r.errors),failures=len(r.failures),scope='Engineering and old regression only, no prospective centers'))
 if not r.wasSuccessful():raise SystemExit(1)
