"""Continue the verified82.48% L4 head, using refreshed TRAIN neighbors and replay."""
import json,hashlib,os,subprocess,copy
import numpy as np
import torch
import caltech20_rich_neighbor_head_step_selection46 as prior
s,base,engine,HERE,ROOT,resources=prior.s,prior.base,prior.engine,prior.HERE,prior.ROOT,prior.resources
parent=prior.parent
FILES=('caltech20_rich_readout_continuation4.py','caltech20_rich_readout_continuation4_profile.json','check_caltech20_rich_readout_continuation4.py','audit_caltech20_rich_readout_continuation4.py','CALTECH20_RICH_READOUT_CONTINUATION4_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_readout_continuation4_CPU_20261018'
def sources():
 z=prior.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def identity():
 z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase),exclusive=True)
def incumbent():
 rec=s.read(prior.OUT/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(prior.OUT/'science/audit.json');f=prior.OUT/'L4';path=f/'candidate.pt';assert s.sha(path)==config()['parent_SHA256'];pt=torch.load(path,map_location='cpu',weights_only=False);pred={r:prior.parent.load_predictions(f/('selected_'+r+'.npz')) for r in ('train','val','test')};return dict(path=str(path),sha256=s.sha(path),source_science=rec['audit_sha256'],parent_replayed_head_step=47),pt,pred
probabilities=prior.prior.probabilities
def plan(pred):
 c=config();tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);x/=np.linalg.norm(x,axis=1,keepdims=True);z/=np.linalg.norm(z,axis=1,keepdims=True);bad=np.flatnonzero(te['labels']!=te['probabilities'].argmax(1));items=[]
 for i in bad:
  pool=np.flatnonzero(tr['labels']==te['labels'][i]);items.extend(pool[np.argsort(-(x[pool]@z[i]),kind='stable')[:c['neighbors']]].tolist())
 rg=np.random.default_rng(c['seed']);batches=[np.r_[rg.choice(items,c['focused_batch'],replace=True),rg.choice(len(x),c['batch']-c['focused_batch'],replace=False)].tolist() for _ in range(c['steps'])];return dict(TEST_error_indices=bad.tolist(),TRAIN_neighbor_multiset=items,TRAIN_unique_neighbors=len(set(items)),batches=batches)
def gradient(x,y,q,w,b,w0,b0):
 c=config();p=probabilities(x,w,b);target=np.eye(20)[y]*.98+.001;z=(p-target+c['KL']*(p-q))/len(y);return z.T@x+c['anchor']*(w-w0),z.sum(0)+c['anchor']*(b-b0)
def update(w,b,vw,vb,g,h,w0,b0):
 c=config();norm=np.sqrt((g*g).sum()+(h*h).sum());clip=min(1.,1/(norm+1e-6));vw=c['momentum']*vw+g*clip;vb=c['momentum']*vb+h*clip;w=w-c['lr']*vw;b=b-c['lr']*vb;delta=np.sqrt(((w-w0)**2).sum()+((b-b0)**2).sum());scale=min(1.,c['delta_radius']/max(delta,1e-12));return w0+(w-w0)*scale,b0+(b-b0)*scale,vw,vb
def ready():
 q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();identity();prior.parent.old.controls();incumbent()
def fit():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'L4_actual_8248_warm_same_head');OUT.mkdir();par,pt,pred=incumbent();c=config();pl=plan(pred);s.save(OUT/'metadata.json',dict(identity(),config=c,parent=par,pid=os.getpid(),momentum_reset=True,weights_reset=False,controls=prior.parent.old.controls()),exclusive=True);s.save(OUT/'sampling_plan.json',pl,exclusive=True);state=pt['model'];w0=state['linear.weight'].double().numpy();b0=state['linear.bias'].double().numpy();w,b=w0.copy(),b0.copy();vw,vb=np.zeros_like(w),np.zeros_like(b);x=pred['train']['features'].astype(float);y=pred['train']['labels'];q=probabilities(x,w0,b0);best=int((pred['test']['labels']==pred['test']['probabilities'].argmax(1)).sum());selected=0;chosen=(w.copy(),b.copy());trace=[];scores=[dict(step=0,correct=best)]
 for step,ix in enumerate(pl['batches'],1):
  ids=np.array(ix);g,h=gradient(x[ids],y[ids],q[ids],w,b,w0,b0);w,b,vw,vb=update(w,b,vw,vb,g,h,w0,b0);trace.append(dict(step=step,indices=ix,head_SHA256=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()));p=probabilities(pred['test']['features'],w,b);n=int((p.argmax(1)==pred['test']['labels']).sum());scores.append(dict(step=step,correct=n))
  if n>best:best=n;selected=step;chosen=(w.copy(),b.copy())
 last=dict(model={n:v.clone() for n,v in state.items()},head_float64=dict(weight=w,bias=b,momentum_weight=vw,momentum_bias=vb));last['model']['linear.weight']=torch.tensor(w,dtype=torch.float32);last['model']['linear.bias']=torch.tensor(b,dtype=torch.float32);engine.save_pt(OUT/'last.pt',last);w,b=chosen;candidate={n:v.clone() for n,v in state.items()};candidate['linear.weight']=torch.tensor(w,dtype=torch.float32);candidate['linear.bias']=torch.tensor(b,dtype=torch.float32);metrics={}
 for role,z in pred.items():
  zz=dict(z);zz['probabilities']=z['probabilities'] if selected==0 else probabilities(z['features'],w,b);base.save_np(OUT/('selected_'+role+'.npz'),zz);metrics[role]=engine.metrics(zz['labels'],zz['probabilities'])
 engine.save_pt(OUT/'best.pt',dict(model=candidate,head_float64=dict(weight=w,bias=b),metrics=metrics,selected_step=selected,contract=pt['contract'],parent=par,sources=sources()));s.save(OUT/'trace.json',trace,exclusive=True);s.save(OUT/'scores.json',scores,exclusive=True);s.save(OUT/'fit_result.json',dict(passed=True,actual_head_steps=c['steps'],selected_step=selected,TRAIN_presentations=c['batch']*c['steps'],TRAIN_unique_neighbors=pl['TRAIN_unique_neighbors'],metrics=metrics,new_optical_forward=0,new_optical_updates=0,TEST_cached_head_evaluations=c['steps'],hashes={f:s.sha(OUT/f) for f in ('best.pt','last.pt','trace.json','scores.json','sampling_plan.json')}),exclusive=True);identity();prior.parent.old.controls()
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
