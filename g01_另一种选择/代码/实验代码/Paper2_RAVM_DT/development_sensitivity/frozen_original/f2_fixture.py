"""Generated engineering arrays only, zero optimizer calls and no scientific data."""
from d3_engine import *
from f2_gate import joint_gate
from joint_gate_reference import joint_decision
from scipy.stats import mannwhitneyu
from f2_plan import arms,PHASES,protocol
def main():
 protect_originals();verify_science_lock();torch.manual_seed(7429);rng=np.random.default_rng(7429)
 assert not list((ROOT/'data').glob('center_*.npz'))
 x=rng.random((1000,24),dtype=np.float32);y=rng.random(1000,dtype=np.float32);ts=np.arange(1000)
 hist=Segment(x[:200],y[:200],ts[:200],'verification');v=Segment(x[200:440],y[200:440],ts[200:440],'verification')
 parent=Forecast().to(DEVICE);candidate=copy.deepcopy(parent)
 with torch.no_grad():candidate.head.bias.add_(.001)
 gate=OriginalCMCGate(ROOT/'runtime',ROOT/'protocol/frozen_v2_final_commit_config.yaml');recent=evidence_blocks(v);history=evidence_blocks(hist);sc=score(candidate,parent,recent,history,v,gate)
 batches=[tensors(windows(b)) for b in history+recent];xy=Batch(torch.cat([b[0] for b in batches]),torch.cat([b[1] for b in batches]),'verification')
 pl=predictions_losses(parent,xy)[1];cl=predictions_losses(candidate,xy)[1];start=sum(len(b[0]) for b in batches[:-6]);d=float((cl-pl)[start:].mean());s=float(pl[start:].mean());p=float(mannwhitneyu(cl[start:],pl[start:],alternative='less',method='asymptotic').pvalue)
 assert d==sc['stage3']['recent_mean'] and s==sc['parent_recent_verification_mean_loss'] and p==sc['stage3']['mwu_p']
 n=0
 for weight in protocol()['joint_mechanism']['lambda_grid']:
  for f in protocol()['joint_mechanism']['margin_fraction_grid']:
   margin=f*protocol()['joint_mechanism']['margin_scale'];a,j=joint_gate(d,s,p,weight,margin);ref=joint_decision(d,s,p,weight=weight,margin=margin);assert a==ref.commit and j==ref.score;n+=1
 # Explicit metamorphic examples cover both novel actions, mean guard and lambda=0.
 for _ in range(200):
  d=float(rng.uniform(-.01,.01));s=float(rng.uniform(0,.02));p=float(rng.uniform());w=float(rng.choice([0,.05,.15,.3]));m=float(rng.uniform(1e-6,.002));a,j=joint_gate(d,s,p,w,m);ref=joint_decision(d,s,p,weight=w,margin=m);assert (a,j)==(ref.commit,ref.score)
  assert joint_gate(d,s,p,0,m)[0]==(d<=-m)
  assert joint_gate(d,s,0,w,m)[1]<=joint_gate(d,s,1,w,m)[1]
  if d>=0:assert not a
 fake=object.__new__(CausalProvider)
 try:fake.service(1,None,None);raise AssertionError('Unsealed Service was allowed')
 except PermissionError:pass
 try:adapt(parent,Batch(torch.zeros((2,96,24),device=DEVICE),torch.zeros((2,1),device=DEVICE),'service'),steps=0,seed=1);raise AssertionError('Service was accepted for training')
 except (AssertionError,PermissionError) as e:
  if str(e)=='Service was accepted for training':raise
 for phase in PHASES[:2]:assert len(arms(phase))==protocol()['proposed_phases'][phase]['arms']
 assert len({a['id'] for a in arms('DevCal')})==66
 for seed,identity in SOURCE.items():
  path=ROOT/identity['path'];assert sha(path)==identity['sha256'];model=Forecast().to(DEVICE);model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True));assert state_hash(model)==identity['state_hash']
 # Replay one authenticated D3f1 Verification state only. No Service/outcome/mask is opened.
 f1=ROOT.parent.parent/'ravm_d3f1_20261020/D3f1_ISOLATED_WORK';seal_path=f1/'opportunities/59000/7311/clean/02/seal.json';seal=read(seal_path)
 def load(entry):
  q=f1/'models'/entry['file_sha256'];assert sha(q)==entry['file_sha256'];m=Forecast().to(DEVICE);m.load_state_dict(torch.load(q,map_location=DEVICE,weights_only=True));assert state_hash(m)==entry['state_hash'];return m
 def segment(q):
  with np.load(q) as z:return Segment(z['x'],z['y'],z['ts'],'verification')
 old=load(seal['parent_entry']);new=load(seal['candidate_entry']);ver=segment(f1/'roles/59000/clean/verification_02.npz');h=segment(f1/'roles/59000/initial_history.npz')
 actual=score(new,old,evidence_blocks(ver),evidence_blocks(h),ver,gate)
 assert actual['stage3']['recent_mean']==seal['features']['recent_mean'] and actual['stage3']['mwu_p']==seal['features']['recent_mwu_p']
 for weight in protocol()['joint_mechanism']['lambda_grid']:
  for f in protocol()['joint_mechanism']['margin_fraction_grid']:
   d=actual['stage3']['recent_mean'];s=actual['parent_recent_verification_mean_loss'];p=actual['stage3']['mwu_p'];margin=f*.00560364755324872
   a,j=joint_gate(d,s,p,weight,margin);ref=joint_decision(d,s,p,weight=weight,margin=margin);assert (a,j)==(ref.commit,ref.score)
 atomic(ROOT/'engineering/FIXTURE_RESULT.json',dict(status='PASS',numerical_stage3_parity=True,parent_scale_same_native_recent_windows=True,reference_gate_grid_cases=n,random_reference_parity_cases=200,service_before_seal_rejected=True,service_training_rejected=True,optimizer_steps=0,source_checkpoints_identity_verified=3,f1_verification_replay_seal_sha256=sha(seal_path),f1_service_or_labels_opened=False,scientific_data_generated=False,device=DEVICE,at_unix=time.time()))
 print('PASS generated numerical/role fixture; zero SGD',flush=True)
if __name__=='__main__':main()
