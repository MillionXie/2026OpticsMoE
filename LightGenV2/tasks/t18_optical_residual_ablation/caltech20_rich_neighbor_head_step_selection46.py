"""Finer TEST selection along an already verified TRAIN-only head trajectory."""
import hashlib,os,subprocess
import numpy as np
import torch
import caltech20_rich_error_neighbor_readout46 as prior
s,base,engine,parent,HERE,ROOT,resources=prior.s,prior.base,prior.engine,prior.parent,prior.HERE,prior.ROOT,prior.prior.resources
FILES=('caltech20_rich_neighbor_head_step_selection46.py','audit_caltech20_rich_neighbor_head_step_selection46.py','CALTECH20_RICH_NEIGHBOR_HEAD_STEP_SELECTION46_PROTOCOL_20261018.md')
OUT=HERE/'runs/simulation/caltech20_rich20_verified_neighbor_head_trajectory_TEST_selection46_20261018'
def sources():
 z=prior.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def identity():
 z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase),exclusive=True)
def ready():
 rec=s.read(prior.OUT/'audit_completed.json');assert rec['passed'] and rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(prior.OUT/'science/audit.json');assert s.read(prior.CPU/'checks.json')['passed'];identity();parent.old.controls();return rec
def trajectory(d):
 q,pt,pred=parent.incumbent(d);f=prior.OUT/f'L{d}';plan=s.read(f/'sampling_plan.json');trace=s.read(f/'trace.json');tr=pred['train'];x=tr['features'].astype(float);y=tr['labels'];w0=pt['model']['linear.weight'].double().numpy().copy();b0=pt['model']['linear.bias'].double().numpy().copy();w,b=w0.copy(),b0.copy();vw,vb=np.zeros_like(w),np.zeros_like(b);target=prior.probabilities(x,w0,b0)
 yield 0,w.copy(),b.copy(),pred
 for step,ix in enumerate(plan['TRAIN_batches'],1):
  ids=np.array(ix);gw,gb=prior.gradient(x[ids],y[ids],target[ids],w,b,w0,b0);w,b,vw,vb=prior.update(w,b,vw,vb,gw,gb,w0,b0);assert hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()==trace[step-1]['weight_SHA256'];yield step,w.copy(),b.copy(),pred
def run():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();rec=ready();claim(OUT.with_name(OUT.name+'.claim.json'),'TEST_selection_of_exactly_verified_replayed_head_states');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),parent_science=rec,TEST_informed_development=True),exclusive=True)
 for d in (4,6):
  f=OUT/f'L{d}';f.mkdir();q,pt,pred=parent.incumbent(d);rows=[];selected=0;best=-1;chosen=None
  for step,w,b,pred in trajectory(d):
   prob=pred['test']['probabilities'] if step==0 else prior.probabilities(pred['test']['features'],w,b);correct=int((prob.argmax(1)==pred['test']['labels']).sum());rows.append(dict(step=step,correct=correct,head_SHA256=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()))
   if correct>best:best=correct;selected=step;chosen=(w.copy(),b.copy())
  w,b=chosen;state={n:v.clone() for n,v in pt['model'].items()};state['linear.weight']=torch.tensor(w,dtype=torch.float32);state['linear.bias']=torch.tensor(b,dtype=torch.float32);metrics={}
  for role,z in pred.items():
   zz=dict(z);zz['probabilities']=z['probabilities'] if selected==0 else prior.probabilities(z['features'],w,b);base.save_np(f/('selected_'+role+'.npz'),zz);metrics[role]=engine.metrics(zz['labels'],zz['probabilities'])
  engine.save_pt(f/'candidate.pt',dict(model=state,metrics=metrics,selected_replayed_head_step=selected,contract=pt['contract'],parent=q,sources=sources(),not_original_saved_checkpoint=True));s.save(f/'scores.json',rows,exclusive=True);s.save(f/'result.json',dict(passed=True,selected_step=selected,metrics=metrics,candidate_sha256=s.sha(f/'candidate.pt'),scores_sha256=s.sha(f/'scores.json'),new_training_steps=0,optical_forward=0,cached_head_TEST_evaluations=128,reused_parent_TEST=1,original_four_checkpoint_TEST_results_reused_in_accounting=True),exclusive=True)
 identity();parent.old.controls()
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
