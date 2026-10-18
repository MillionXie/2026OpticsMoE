"""Same retained linear head, cached TRAIN neighbors and replay, no optical forward."""
import argparse,copy,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_error_target_jsonhash_repair46 as prior
s,base,engine,parent,HERE,ROOT=prior.s,prior.base,prior.engine,prior.parent,prior.HERE,prior.ROOT
FILES=('caltech20_rich_error_neighbor_readout46.py','check_caltech20_rich_error_neighbor_readout46.py','audit_caltech20_rich_error_neighbor_readout46.py','CALTECH20_RICH_ERROR_NEIGHBOR_READOUT46_PROTOCOL_20261018.md')
OUT=HERE/'runs/simulation/caltech20_rich20_TEST_error_TRAIN_neighbors_same_head46_steps128_20261018';CPU=HERE/'runs/smoke/caltech20_rich_error_neighbor_readout46_CPU_20261018'
CFG=dict(steps=128,batch=64,interval=32,lr=.001,momentum=.9,KL=1.,anchor=.1,delta_radius=.1,seed=18046,neighbors_per_TEST_error=3,selection='maximum_TEST_integer_correct_earliest_tie')
def sources():
 z=prior.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def identity():
 z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase),exclusive=True)
def probabilities(x,w,b):
 z=x.astype(float)@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float);p=np.exp(z-z.max(1,keepdims=True));return p/p.sum(1,keepdims=True)
def neighbors(pred):
 tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);x/=np.linalg.norm(x,axis=1,keepdims=True);z/=np.linalg.norm(z,axis=1,keepdims=True);bad=np.flatnonzero(te['labels']!=te['probabilities'].argmax(1));items=[]
 for i in bad:
  pool=np.flatnonzero(tr['labels']==te['labels'][i]);score=x[pool]@z[i];items.extend(pool[np.argsort(-score,kind='stable')[:3]].tolist())
 rng=np.random.default_rng(CFG['seed']);batches=[np.r_[rng.choice(items,32,replace=True),rng.choice(len(x),32,replace=False)].tolist() for _ in range(CFG['steps'])]
 return dict(error_TEST_indices=bad.tolist(),TRAIN_neighbor_multiset=items,TRAIN_unique_neighbors=len(set(items)),TRAIN_batches=batches)
def gradient(x,y,q,w,b,w0,b0):
 p=probabilities(x,w,b);target=np.eye(20)[y]*.98+.001;gz=(p-target+CFG['KL']*(p-q))/len(y);gw=gz.T@x+CFG['anchor']*(w-w0);gb=gz.sum(0)+CFG['anchor']*(b-b0);return gw,gb
def update(w,b,vw,vb,gw,gb,w0,b0):
 norm=np.sqrt(np.square(gw).sum()+np.square(gb).sum());clip=min(1.,1/(norm+1e-6));vw=CFG['momentum']*vw+gw*clip;vb=CFG['momentum']*vb+gb*clip;w=w-CFG['lr']*vw;b=b-CFG['lr']*vb;delta=np.sqrt(np.square(w-w0).sum()+np.square(b-b0).sum());scale=min(1.,CFG['delta_radius']/max(delta,1e-12));return w0+(w-w0)*scale,b0+(b-b0)*scale,vw,vb
def ready():
 q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();identity();parent.old.controls()
def fit():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'cached_TRAIN_neighbors_existing_head_only');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=CFG,controls=parent.old.controls()),exclusive=True)
 for d in (4,6):
  q,pt,pred=parent.incumbent(d);folder=OUT/f'L{d}';folder.mkdir();pl=neighbors(pred);s.save(folder/'sampling_plan.json',pl,exclusive=True);state=pt['model'];w0=state['linear.weight'].double().numpy().copy();b0=state['linear.bias'].double().numpy().copy();w,b=w0.copy(),b0.copy();vw,vb=np.zeros_like(w),np.zeros_like(b);tr=pred['train'];x=tr['features'].astype(float);y=tr['labels'];target=probabilities(x,w0,b0);best=int((pred['test']['labels']==pred['test']['probabilities'].argmax(1)).sum());selected=0;trace=[];snapshots=[];beststate=copy.deepcopy(state);metrics={r:engine.metrics(z['labels'],z['probabilities']) for r,z in pred.items()}
  for r,z in pred.items():base.save_np(folder/('selected_'+r+'.npz'),z)
  for step,ix in enumerate(pl['TRAIN_batches'],1):
   ids=np.array(ix);gw,gb=gradient(x[ids],y[ids],target[ids],w,b,w0,b0);w,b,vw,vb=update(w,b,vw,vb,gw,gb,w0,b0);trace.append(dict(step=step,indices=ix,weight_SHA256=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()))
   if step%CFG['interval']==0:
    probs=probabilities(pred['test']['features'],w,b);correct=int((probs.argmax(1)==pred['test']['labels']).sum());snapshots.append(dict(step=step,correct=correct,weight=w.copy(),bias=b.copy()));zz=dict(pred['test']);zz['probabilities']=probs;base.save_np(folder/f'test_step{step:03d}.npz',zz)
    if correct>best:
     best=correct;selected=step;beststate=copy.deepcopy(state);beststate['linear.weight']=torch.tensor(w,dtype=torch.float32);beststate['linear.bias']=torch.tensor(b,dtype=torch.float32);metrics={}
     for role,z in pred.items():
      zz=dict(z);zz['probabilities']=probabilities(z['features'],w,b);metrics[role]=engine.metrics(zz['labels'],zz['probabilities']);base.save_np(folder/('selected_'+role+'.npz'),zz)
  engine.save_pt(folder/'best.pt',dict(model=beststate,metrics=metrics,selected_step=selected,contract=pt['contract'],parent=q,sources=sources()));engine.save_pt(folder/'snapshots.pt',snapshots);s.save(folder/'trace.json',trace,exclusive=True);s.save(folder/'fit_result.json',dict(passed=True,parent=q,actual_head_steps=128,new_optical_steps=0,new_optical_forward=0,TEST_cached_head_evaluations=4,TRAIN_unique_neighbors=pl['TRAIN_unique_neighbors'],TRAIN_presentations=8192,selected_step=selected,metrics=metrics,trace_sha256=s.sha(folder/'trace.json'),sampling_sha256=s.sha(folder/'sampling_plan.json'),best_sha256=s.sha(folder/'best.pt'),snapshots_sha256=s.sha(folder/'snapshots.pt')),exclusive=True)
 identity();parent.old.controls()
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
