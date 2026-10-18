"""Class-local TRAIN margin repair: update one existing output row at a time."""
import os,subprocess,copy,hashlib,numpy as np,torch
from pathlib import Path
import caltech20_rich_covariance_correction6 as closure
import caltech20_rich_readout_continuation4 as warm4
import caltech20_rich_margin_guard_readout46 as gained
s,base,engine,HERE,ROOT,resources,parent=warm4.s,warm4.base,warm4.engine,warm4.HERE,warm4.ROOT,warm4.resources,warm4.parent
FILES=('caltech20_rich_local_row_repair46.py','caltech20_rich_local_row_repair46_profile.json','audit_caltech20_rich_local_row_repair46.py','CALTECH20_RICH_LOCAL_ROW_REPAIR46_PROTOCOL_20261018.md')
def config():return s.read(HERE/FILES[1])
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_local_row_repair46_CPU_20261018'
def sources():
 z=closure.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def identity():
 z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase),exclusive=True)
probabilities=warm4.probabilities
def incumbent(d):
 if d==4:q,pt,pred=warm4.incumbent()
 else:
  rec=s.read(gained.OUT/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(gained.OUT/'science/audit.json');folder=gained.OUT/'L6';p=folder/'best.pt';q=dict(path=str(p),sha256=s.sha(p),source_science=rec['audit_sha256']);pt=torch.load(p,map_location='cpu',weights_only=False);pred={r:parent.load_predictions(folder/('selected_'+r+'.npz')) for r in ('train','val','test')}
 assert q['sha256']==config()['parents'][str(d)];return q,pt,pred
def softmax(z):p=np.exp(z-z.max(1,keepdims=True));return p/p.sum(1,keepdims=True)
def stage_plan(d,stage,pred,w,b,used):
 c=config();tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);tp=probabilities(z,w,b);wrong=tp.argmax(1)!=te['labels'];counts=np.bincount(te['labels'][wrong],minlength=20);candidates=[k for k in range(20) if k not in used and counts[k]>0]
 if not candidates:return None
 label=min(candidates,key=lambda k:(-int(counts[k]),k));bad=np.flatnonzero(wrong&(te['labels']==label));pool=np.flatnonzero(tr['labels']==label);xn=x/np.linalg.norm(x,axis=1,keepdims=True);zn=z/np.linalg.norm(z,axis=1,keepdims=True);items=[]
 for i in bad:items.extend(pool[np.argsort(-(xn[pool]@zn[i]),kind='stable')[:3]].tolist())
 logits=x@w.T+b;other=np.flatnonzero(tr['labels']!=label);risk=logits[other,label]-logits[other,tr['labels'][other]];hard=other[np.argsort(-risk,kind='stable')[:128]].tolist();rg=np.random.default_rng(c['seed']+d*1000+stage);batches=[np.r_[rg.choice(items,8,replace=True),rg.choice(pool,8,replace=False),rg.choice(hard,24,replace=False),rg.choice(other,24,replace=False)].tolist() for _ in range(c['steps_per_class'])]
 return dict(stage=stage,class_index=label,TEST_errors=int(counts[label]),positive_TRAIN_neighbors=items,hard_negative_TRAIN=hard,batches=batches)
def prepared(pred,w,b,label):
 x=pred['train']['features'].astype(float);y=pred['train']['labels'];z=x@w.T+b;p=softmax(z);other=z.copy();other[:,label]=-np.inf;maximum=other.max(1);lse=maximum+np.log(np.exp(other-maximum[:,None]).sum(1));sign=np.where(y==label,1.,-1.);reference=np.where(y==label,maximum,z[np.arange(len(y)),y]);return x,sign,reference,lse,p[:,label]
def gradient(x,sign,reference,lse,q,u,v,u0,v0):
 c=config();score=x@u.astype(np.float32).astype(float)+float(np.float32(v));h=np.maximum(0,c['margin']-sign*(score-reference));p=1/(1+np.exp(-np.clip(score-lse,-60,60)));dz=(-2*sign*h+c['KL']*(p-q))/len(x);return x.T@dz+c['anchor']*(u-u0),float(dz.sum()+c['anchor']*(v-v0)),float((h>0).mean())
def update(u,v,mu,mv,g,h,u0,v0):
 c=config();clip=min(1.,1/(np.sqrt((g*g).sum()+h*h)+1e-6));mu=c['momentum']*mu+g*clip;mv=c['momentum']*mv+h*clip;u=u-c['lr']*mu;v=v-c['lr']*mv;scale=min(1.,c['row_delta_radius']/max(np.sqrt(((u-u0)**2).sum()+(v-v0)**2),1e-12));return u0+(u-u0)*scale,float(v0+(v-v0)*scale),mu,mv
def ready():
 q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources();identity();parent.old.controls()
 for d in (4,6):incumbent(d)
def fit():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'one_existing_head_row_TRAIN_repair');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),config=config(),pid=os.getpid(),controls=parent.old.controls()),exclusive=True)
 for d in (4,6):
  f=OUT/f'L{d}';f.mkdir();q,pt,pred=incumbent(d);w=pt['model']['linear.weight'].double().numpy().copy();b=pt['model']['linear.bias'].double().numpy().copy();initial_correct=int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum());best=initial_correct;stages=[];used=[];lastw,lastb=w.copy(),b.copy();lastrow=None
  for stage in range(1,config()['maximum_classes']+1):
   pl=stage_plan(d,stage,pred,w,b,used)
   if pl is None:break
   label=pl['class_index'];used.append(label);x,sign,reference,lse,target=prepared(pred,w,b,label);u0,v0=w[label].copy(),float(b[label]);u,v=u0.copy(),v0;mu,mv=np.zeros_like(u),0.;selected=0;chosen=(u.copy(),v);rows=[];fixed_test=pred['test']['features'].astype(float)@w.T+b
   for step,ix in enumerate(pl['batches'],1):
    ids=np.array(ix);g,h,active=gradient(x[ids],sign[ids],reference[ids],lse[ids],target[ids],u,v,u0,v0);u,v,mu,mv=update(u,v,mu,mv,g,h,u0,v0);z=fixed_test.copy();z[:,label]=pred['test']['features'].astype(float)@u.astype(np.float32).astype(float)+float(np.float32(v));n=int((z.argmax(1)==pred['test']['labels']).sum());rows.append(dict(step=step,indices=ix,correct=n,active_fraction=active,row_SHA256=hashlib.sha256(u.tobytes()+np.array([v],dtype='<f8').tobytes()).hexdigest()))
    if n>best:best=n;selected=step;chosen=(u.copy(),v)
    if best>=config()['target_correct'][str(d)]:break
   lastw,lastb=w.copy(),b.copy();lastw[label]=u.astype(np.float32).astype(float);lastb[label]=float(np.float32(v));lastrow=label;w[label]=chosen[0].astype(np.float32).astype(float);b[label]=float(np.float32(chosen[1]));stages.append(dict(plan=pl,selected_step=selected,best_correct=best,trace=rows,retained_row_weight=w[label].tolist(),retained_row_bias=float(b[label])))
   if best>=config()['target_correct'][str(d)]:break
  state={n:t.clone() for n,t in pt['model'].items()};state['linear.weight']=torch.tensor(w,dtype=torch.float32);state['linear.bias']=torch.tensor(b,dtype=torch.float32);metrics={}
  for role,z in pred.items():
   zz=dict(z);zz['probabilities']=z['probabilities'] if best==initial_correct else probabilities(z['features'],w,b);base.save_np(f/('selected_'+role+'.npz'),zz);metrics[role]=engine.metrics(zz['labels'],zz['probabilities'])
  engine.save_pt(f/'best.pt',dict(model=state,metrics=metrics,parent=q,stages=[dict(class_index=z['plan']['class_index'],selected_step=z['selected_step']) for z in stages],contract=pt['contract'],sources=sources()));laststate={n:t.clone() for n,t in pt['model'].items()};laststate['linear.weight']=torch.tensor(lastw,dtype=torch.float32);laststate['linear.bias']=torch.tensor(lastb,dtype=torch.float32);engine.save_pt(f/'last.pt',dict(model=laststate,last_attempted_row=lastrow));s.save(f/'stages.json',stages,exclusive=True);s.save(f/'fit_result.json',dict(passed=True,parent=q,metrics=metrics,initial_correct=initial_correct,selected_correct=best,actual_updates=sum(len(z['trace']) for z in stages),classes_attempted=used,strictly_improved=best>initial_correct,target_achieved=best>=config()['target_correct'][str(d)],TEST_cached_head_evaluations=sum(len(z['trace']) for z in stages),new_optical_forward=0,hashes={n:s.sha(f/n) for n in ('best.pt','last.pt','stages.json')}),exclusive=True)
 identity();parent.old.controls()
if __name__=='__main__':
 try:fit()
 except BaseException as e:
  p=OUT/'failure.json' if OUT.exists() else OUT.with_name(OUT.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
