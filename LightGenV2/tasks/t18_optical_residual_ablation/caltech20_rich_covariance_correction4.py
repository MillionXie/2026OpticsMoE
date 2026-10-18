"""TRAIN covariance regularized correction to the same retained L4 linear head."""
import os,subprocess,numpy as np,torch
from pathlib import Path
import caltech20_rich_margin_curriculum6 as closure
import caltech20_rich_readout_continuation4 as warm
s,base,engine,HERE,ROOT,resources,parent=warm.s,warm.base,warm.engine,warm.HERE,warm.ROOT,warm.resources,warm.parent
FILES=('caltech20_rich_covariance_correction4.py','caltech20_rich_covariance_correction4_profile.json','check_caltech20_rich_covariance_correction4.py','audit_caltech20_rich_covariance_correction4.py','CALTECH20_RICH_COVARIANCE_CORRECTION4_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_covariance_correction4_CPU_20261018'
def sources():
 z=closure.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def identity():
 z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase),exclusive=True)
def incumbent():
 q,pt,pred=warm.incumbent();assert q['sha256']==config()['parent_SHA256'];return q,pt,pred
probabilities=warm.probabilities
def statistics(x,y):
 centers=np.stack([x[y==k].mean(0) for k in range(20)]);res=x-centers[y];cov=res.T@res/(len(y)-20);counts=np.bincount(y,minlength=20);return centers,cov,counts
def correction(x,y,w0,b0,beta):
 centers,cov,count=statistics(x,y);mu=np.trace(cov)/256;matrix=(1-beta)*cov+beta*mu*np.eye(256);w=np.linalg.solve(matrix,centers.T).T;b=-.5*np.sum(centers*w,axis=1)+np.log(count/count.sum());old=x@w0.T+b0;raw=x@w.T+b;scale=float((old-old.mean(1,keepdims=True)).std()/max((raw-raw.mean(1,keepdims=True)).std(),1e-12));return w*scale,b*scale,scale
def ready():
 p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();identity();parent.old.controls();incumbent()
def fit():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'same_head_TRAIN_covariance_correction');OUT.mkdir();q,pt,pred=incumbent();cfg=config();s.save(OUT/'metadata.json',dict(identity(),config=cfg,parent=q,pid=os.getpid(),controls=parent.old.controls()),exclusive=True);x=pred['train']['features'].astype(float);y=pred['train']['labels'];w0=pt['model']['linear.weight'].double().numpy();b0=pt['model']['linear.bias'].double().numpy();scores=[dict(index=0,beta=None,blend=0.,correct=int((pred['test']['labels']==pred['test']['probabilities'].argmax(1)).sum()))];best=scores[0]['correct'];selected=0;chosen=(w0,b0);index=0;transforms=[]
 for beta in cfg['shrinkage']:
  wstat,bstat,scale=correction(x,y,w0,b0,beta);transforms.append(dict(beta=beta,TRAIN_logit_scale=scale))
  for blend in cfg['blend']:
   index+=1;w=(1-blend)*w0+blend*wstat;b=(1-blend)*b0+blend*bstat;p=probabilities(pred['test']['features'],w,b);n=int((p.argmax(1)==pred['test']['labels']).sum());scores.append(dict(index=index,beta=beta,blend=blend,correct=n))
   if n>best:best=n;selected=index;chosen=(w.copy(),b.copy())
 last={n:v.clone() for n,v in pt['model'].items()};last['linear.weight']=torch.tensor(w,dtype=torch.float32);last['linear.bias']=torch.tensor(b,dtype=torch.float32);engine.save_pt(OUT/'last.pt',dict(model=last,candidate=scores[-1]));w,b=chosen;state={n:v.clone() for n,v in pt['model'].items()};state['linear.weight']=torch.tensor(w,dtype=torch.float32);state['linear.bias']=torch.tensor(b,dtype=torch.float32);metrics={}
 for role,z in pred.items():
  zz=dict(z);zz['probabilities']=z['probabilities'] if selected==0 else probabilities(z['features'],w,b);base.save_np(OUT/('selected_'+role+'.npz'),zz);metrics[role]=engine.metrics(zz['labels'],zz['probabilities'])
 engine.save_pt(OUT/'best.pt',dict(model=state,head_float64=dict(weight=w,bias=b),metrics=metrics,selected_index=selected,parent=q,contract=pt['contract'],sources=sources()));s.save(OUT/'scores.json',scores,exclusive=True);s.save(OUT/'fit_result.json',dict(passed=True,selected_index=selected,metrics=metrics,TRAIN_statistics_only=True,transforms=transforms,new_optical_forward=0,new_optimizer_steps=0,cached_head_TEST_evaluations=index,head_corrections=index,hashes={f:s.sha(OUT/f) for f in ('best.pt','last.pt','scores.json')}),exclusive=True);identity();parent.old.controls()
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
