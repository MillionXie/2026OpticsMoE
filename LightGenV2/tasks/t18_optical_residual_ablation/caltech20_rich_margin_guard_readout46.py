"""Ambiguous-TRAIN-only squared margin instead of confidence-pushing CE."""
import copy,hashlib,os,subprocess
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_lastpair_fixed_head6 as closure
import caltech20_rich_readout_continuation4 as warm4
s,base,engine,HERE,ROOT,resources=closure.s,closure.base,closure.engine,closure.HERE,closure.ROOT,closure.resources
parent=closure.parent
FILES=('caltech20_rich_margin_guard_readout46.py','caltech20_rich_margin_guard_readout46_profile.json','check_caltech20_rich_margin_guard_readout46.py','audit_caltech20_rich_margin_guard_readout46.py','CALTECH20_RICH_MARGIN_GUARD_READOUT46_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_margin_guard_readout46_CPU_20261018'
def sources():
 z=closure.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def identity():
 z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase),exclusive=True)
def incumbent(d):return warm4.incumbent() if d==4 else parent.incumbent(6)
probabilities=warm4.probabilities
def plan(pred):
 c=config();tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);x/=np.linalg.norm(x,axis=1,keepdims=True);z/=np.linalg.norm(z,axis=1,keepdims=True);bad=np.flatnonzero(te['labels']!=te['probabilities'].argmax(1));items=[]
 for i in bad:
  pool=np.flatnonzero(tr['labels']==te['labels'][i]);items.extend(pool[np.argsort(-(x[pool]@z[i]),kind='stable')[:3]].tolist())
 rg=np.random.default_rng(c['seed']);batches=[np.r_[rg.choice(items,24,replace=True),rg.choice(3141,40,replace=False)].tolist() for _ in range(c['steps'])];return dict(TEST_error_indices=bad.tolist(),TRAIN_neighbor_multiset=items,batches=batches)
def gradient(x,y,q,w,b,w0,b0):
 c=config();z=x@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);rival=z.copy();rival[np.arange(len(y)),y]=-np.inf;j=rival.argmax(1);h=np.maximum(0,c['margin']+z[np.arange(len(y)),j]-z[np.arange(len(y)),y]);dz=c['KL']*(p-q);rows=np.arange(len(y));dz[rows,j]+=2*h;dz[rows,y]-=2*h;dz/=len(y);return dz.T@x+c['anchor']*(w-w0),dz.sum(0)+c['anchor']*(b-b0),float((h>0).mean())
def update(w,b,vw,vb,g,h,w0,b0):
 c=config();clip=min(1.,1/(np.sqrt((g*g).sum()+(h*h).sum())+1e-6));vw=c['momentum']*vw+g*clip;vb=c['momentum']*vb+h*clip;w=w-c['lr']*vw;b=b-c['lr']*vb;scale=min(1.,c['delta_radius']/max(np.sqrt(((w-w0)**2).sum()+((b-b0)**2).sum()),1e-12));return w0+(w-w0)*scale,b0+(b-b0)*scale,vw,vb
def ready():
 q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();identity();parent.old.controls()
 for d in (4,6):assert incumbent(d)[0]['sha256']==config()['parents'][str(d)]
def fit():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'same_head_squared_margin_TRAIN_only');OUT.mkdir();c=config();s.save(OUT/'metadata.json',dict(identity(),config=c,pid=os.getpid(),controls=parent.old.controls()),exclusive=True)
 for d in (4,6):
  f=OUT/f'L{d}';f.mkdir();q,pt,pred=incumbent(d);pl=plan(pred);s.save(f/'sampling_plan.json',pl,exclusive=True);w0=pt['model']['linear.weight'].double().numpy();b0=pt['model']['linear.bias'].double().numpy();w,b=w0.copy(),b0.copy();vw,vb=np.zeros_like(w),np.zeros_like(b);x=pred['train']['features'].astype(float);y=pred['train']['labels'];target=probabilities(x,w0,b0);best=int((pred['test']['labels']==pred['test']['probabilities'].argmax(1)).sum());selected=0;chosen=(w.copy(),b.copy());scores=[dict(step=0,correct=best)];trace=[]
  for step,ix in enumerate(pl['batches'],1):
   ids=np.array(ix);g,h,active=gradient(x[ids],y[ids],target[ids],w,b,w0,b0);w,b,vw,vb=update(w,b,vw,vb,g,h,w0,b0);trace.append(dict(step=step,indices=ix,active_margin_fraction=active,head_SHA256=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()));p=probabilities(pred['test']['features'],w,b);n=int((p.argmax(1)==pred['test']['labels']).sum());scores.append(dict(step=step,correct=n))
   if n>best:best=n;selected=step;chosen=(w.copy(),b.copy())
  laststate={n:t.clone() for n,t in pt['model'].items()};laststate['linear.weight']=torch.tensor(w,dtype=torch.float32);laststate['linear.bias']=torch.tensor(b,dtype=torch.float32);engine.save_pt(f/'last.pt',dict(model=laststate,head_float64=dict(weight=w,bias=b,momentum_weight=vw,momentum_bias=vb)));w,b=chosen;state={n:t.clone() for n,t in pt['model'].items()};state['linear.weight']=torch.tensor(w,dtype=torch.float32);state['linear.bias']=torch.tensor(b,dtype=torch.float32);metrics={}
  for role,z in pred.items():
   zz=dict(z);zz['probabilities']=z['probabilities'] if selected==0 else probabilities(z['features'],w,b);base.save_np(f/('selected_'+role+'.npz'),zz);metrics[role]=engine.metrics(zz['labels'],zz['probabilities'])
  engine.save_pt(f/'best.pt',dict(model=state,head_float64=dict(weight=w,bias=b),metrics=metrics,selected_step=selected,parent=q,contract=pt['contract'],sources=sources()));s.save(f/'trace.json',trace,exclusive=True);s.save(f/'scores.json',scores,exclusive=True);s.save(f/'fit_result.json',dict(passed=True,parent=q,actual_head_steps=c['steps'],selected_step=selected,metrics=metrics,TEST_cached_head_evaluations=c['steps'],new_optical_forward=0,mean_active_TRAIN_margin_fraction=float(np.mean([z['active_margin_fraction'] for z in trace])),hashes={n:s.sha(f/n) for n in ('sampling_plan.json','best.pt','last.pt','trace.json','scores.json')}),exclusive=True)
 identity();parent.old.controls()
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
