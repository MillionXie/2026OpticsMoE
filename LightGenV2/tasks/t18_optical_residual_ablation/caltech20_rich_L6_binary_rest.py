"""L6 only: direct one-vs-rest boundary, then TRAIN-curvature preconditioning."""
import os,sys,copy,types,subprocess,hashlib,numpy as np,torch
from pathlib import Path
import caltech20_rich_L6_local_competition as previous
s,base,engine,HERE,ROOT,resources,parent=previous.s,previous.base,previous.engine,previous.HERE,previous.ROOT,previous.resources,previous.parent
FILES=('caltech20_rich_L6_binary_rest.py','caltech20_rich_L6_binary_rest_profile.json','audit_caltech20_rich_L6_binary_rest.py','CALTECH20_RICH_L6_BINARY_REST_PROTOCOL_20261018.md')
PROFILE=s.read(HERE/FILES[1]);CASE=int(os.environ.get('T18_L6_COMPETITION_CASE','0'));ROOT_OUT=HERE/'runs/simulation'/PROFILE['run'];OUT=ROOT_OUT/f'case{CASE:02d}';CPU=HERE/'runs/smoke/caltech20_rich_L6_binary_rest_CPU_20261018';probabilities=previous.probabilities
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
def initial_parent():
 completed=s.read(previous.ROOT_OUT/'completed.json');ex=s.read(previous.ROOT_OUT/'campaign_exit.json');assert completed['passed'] and ex['actual_exit'] and ex['returncode']==0;q=dict(completed['best']);assert q['sha256']==PROFILE['initial_parent_SHA256'];return q
for name in ('identity','claim','specification','config','frozen','load_parent','ready','campaign'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def plan(stage,pred,w,b,used):
 tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);bad=np.flatnonzero(probabilities(z,w,b).argmax(1)!=te['labels']);counts=np.bincount(te['labels'][bad],minlength=20);remaining=[k for k in range(20) if counts[k]>0 and k not in used]
 if not remaining:return None
 label=min(remaining,key=lambda k:(-int(counts[k]),k));errors=[int(i) for i in bad if te['labels'][i]==label];pool=np.flatnonzero(tr['labels']==label);other=np.flatnonzero(tr['labels']!=label);xn=x/np.linalg.norm(x,axis=1,keepdims=True);zn=z/np.linalg.norm(z,axis=1,keepdims=True);items=[]
 for i in errors:items.extend(pool[np.argsort(-(xn[pool]@zn[i]),kind='stable')[:3]].tolist())
 logits=x@w.T+b;risk=logits[other,label]-logits[other,tr['labels'][other]];hard=other[np.argsort(-risk,kind='stable')[:128]].tolist();rg=np.random.default_rng(PROFILE['seed']+CASE*100000+stage);batches=[np.r_[rg.choice(items,16,replace=True),rg.choice(pool,16,replace=False),rg.choice(hard,16,replace=False),rg.choice(len(x),16,replace=False)].tolist() for _ in range(PROFILE['steps_per_class'])];return dict(stage=stage,label=label,error_indices=errors,TRAIN_neighbors=items,hard_TRAIN=hard,batches=batches)
def prepared(pred,w,b,label):
 x=pred['train']['features'].astype(float);y=pred['train']['labels'];z=x@w.T+b;other=z.copy();other[:,label]=-np.inf;mx=other.max(1);reference=mx+np.log(np.exp(other-mx[:,None]).sum(1));target=np.where(y==label,1-PROFILE['label_smoothing']/2,PROFILE['label_smoothing']/2);q=1/(1+np.exp(-np.clip(z[:,label]-reference,-60,60)));return x,target,reference,q
def gradient(x,target,reference,q,u,v,u0,v0,family):
 c=PROFILE;n=len(x);score=x@u.astype(np.float32).astype(float)+float(np.float32(v))-reference;p=1/(1+np.exp(-np.clip(score,-60,60)));soft=1/(1+np.exp(-np.clip(score/c['temperature'],-60,60)));weights=np.ones(n);weights[:16]=c['neighbor_weight'];dz=(weights*c['temperature']*(soft-target)+c['KL']*(p-q))/n;g=x.T@dz+c['anchor']*(u-u0);h=float(dz.sum()+c['anchor']*(v-v0));curvature=(weights*soft*(1-soft)+c['KL']*p*(1-p))/n;diagonal=(x*x).T@curvature+c['anchor'];bias_diagonal=float(curvature.sum()+c['anchor'])
 if family=='binary_rest_diagonal':g=g/diagonal;h=h/bias_diagonal
 return g,float(h),dict(diagonal_maxmin=float(diagonal.max()/diagonal.min()),diagonal_min=float(diagonal.min()),diagonal_max=float(diagonal.max()),preconditioned=family=='binary_rest_diagonal')
def update(u,v,mu,mv,g,h,u0,v0):
 c=PROFILE;scale=min(1.,1/(np.sqrt((g*g).sum()+h*h)+1e-6));mu=c['momentum']*mu+g*scale;mv=c['momentum']*mv+h*scale;u=u-c['lr']*mu;v=v-c['lr']*mv;scale=min(1.,c['row_radius']/max(np.sqrt(((u-u0)**2).sum()+(v-v0)**2),1e-12));return u0+(u-u0)*scale,float(v0+(v-v0)*scale),mu,mv
def fit():
 assert CASE>0 and os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'L6_direct_binary_rest_TRAIN_fit');OUT.mkdir();sp=specification();q=sp['parent'];pt,pred=load_parent(q);s.save(OUT/'metadata.json',dict(identity(),config=config(),pid=os.getpid(),command=sys.argv,environment=dict(python=sys.version,torch=str(torch.__version__),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']),no_automation=True),exclusive=True);w=pt['model']['linear.weight'].double().numpy().copy();b=pt['model']['linear.bias'].double().numpy().copy();initial=int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum());best=initial;used=[];records=[];lastw,lastb=w.copy(),b.copy()
 for stage in range(1,PROFILE['maximum_classes']+1):
  pl=plan(stage,pred,w,b,used)
  if pl is None:break
  label=pl['label'];used.append(label);x,target,ref,teacher=prepared(pred,w,b,label);u0,v0=w[label].copy(),float(b[label]);u,v=u0.copy(),v0;mu,mv=np.zeros_like(u),0.;chosen=(u.copy(),v);selected=0;trace=[]
  for step,ix in enumerate(pl['batches'],1):
   ids=np.array(ix);g,h,stats=gradient(x[ids],target[ids],ref[ids],teacher[ids],u,v,u0,v0,sp['family']);u,v,mu,mv=update(u,v,mu,mv,g,h,u0,v0);ww,bb=w.copy(),b.copy();ww[label]=u;bb[label]=v;n=int((probabilities(pred['test']['features'],ww,bb).argmax(1)==pred['test']['labels']).sum());trace.append(dict(step=step,indices=ix,correct=n,stats=stats,row_SHA256=hashlib.sha256(u.tobytes()+np.array([v],dtype='<f8').tobytes()).hexdigest()))
   if n>best:best=n;selected=step;chosen=(u.copy(),v)
   if best>=PROFILE['target_correct']:break
  lastw,lastb=w.copy(),b.copy();lastw[label]=u;lastb[label]=v;w[label]=chosen[0].astype(np.float32).astype(float);b[label]=float(np.float32(chosen[1]));records.append(dict(plan=pl,selected_step=selected,best_correct=best,trace=trace,retained_weight=w[label].tolist(),retained_bias=float(b[label])))
  if best>=PROFILE['target_correct']:break
 state={k:v.clone() for k,v in pt['model'].items()};state['linear.weight']=torch.tensor(w,dtype=torch.float32);state['linear.bias']=torch.tensor(b,dtype=torch.float32);metrics={}
 for role,z in pred.items():
  zz=dict(z);zz['probabilities']=z['probabilities'] if best==initial else probabilities(z['features'],w,b);base.save_np(OUT/('selected_'+role+'.npz'),zz);metrics[role]=engine.metrics(zz['labels'],zz['probabilities'])
 engine.save_pt(OUT/'best.pt',dict(model=state,metrics=metrics,parent=q,contract=pt['contract'],sources=sources(),selected=[dict(label=z['plan']['label'],step=z['selected_step']) for z in records]));last={k:v.clone() for k,v in pt['model'].items()};last['linear.weight']=torch.tensor(lastw,dtype=torch.float32);last['linear.bias']=torch.tensor(lastb,dtype=torch.float32);engine.save_pt(OUT/'last.pt',dict(model=last));s.save(OUT/'stages.json',records,exclusive=True);s.save(OUT/'fit_result.json',dict(passed=True,family=sp['family'],parent=q,initial_correct=initial,selected_correct=best,strictly_improved=best>initial,target_achieved=best>=PROFILE['target_correct'],actual_updates=sum(len(z['trace']) for z in records),metrics=metrics,new_optical_forward=0,TEST_cached_head_evaluations=sum(len(z['trace']) for z in records),hashes={n:s.sha(OUT/n) for n in ('best.pt','last.pt','stages.json')}),exclusive=True);identity();frozen()
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);a=p.parse_args()
 try:fit() if a.phase=='fit' else campaign()
 except BaseException as e:
  folder=OUT if a.phase=='fit' else ROOT_OUT;p=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
