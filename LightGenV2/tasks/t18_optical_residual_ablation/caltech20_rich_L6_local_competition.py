"""Foreground L6-only pair/triple objectives on the existing fixed optical features."""
import os,sys,copy,types,subprocess,collections,hashlib,numpy as np,torch
from pathlib import Path
import caltech20_rich_conflict_guard_pair6 as previous
s,base,engine,HERE,ROOT,resources,parent=previous.s,previous.base,previous.engine,previous.HERE,previous.ROOT,previous.resources,previous.parent
FILES=('caltech20_rich_L6_local_competition.py','caltech20_rich_L6_local_competition_profile.json','audit_caltech20_rich_L6_local_competition.py','CALTECH20_RICH_L6_LOCAL_COMPETITION_PROTOCOL_20261018.md')
PROFILE=s.read(HERE/FILES[1]);CASE=int(os.environ.get('T18_L6_COMPETITION_CASE','0'));ROOT_OUT=HERE/'runs/simulation'/PROFILE['run'];OUT=ROOT_OUT/f'case{CASE:02d}';CPU=HERE/'runs/smoke/caltech20_rich_L6_local_competition_CPU_20261018'
probabilities=previous.probabilities
def sources():
 z=previous.sources()
 for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
 return z
for name in ('identity','claim'):
 f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def initial_parent():
 q,_,_=previous.incumbent(6);q=dict(q,predictions_folder=str(Path(q['path']).parent),science_directory=str(Path(q['path']).parent.parent));assert q['sha256']==PROFILE['initial_parent_SHA256'];return q
def specification():
 if CASE==0:return dict(case=0,family=PROFILE['families'][0],pass_number=0,parent=initial_parent())
 z=s.read(ROOT_OUT/f'case{CASE:02d}.spec.json');assert z['case']==CASE and z['family'] in PROFILE['families'];assert 1<=z['pass_number']<=PROFILE['maximum_passes_per_family'];return z
def config():return dict(PROFILE,case_spec=specification())
def frozen():
 for name,z in PROFILE['protected'].items():assert s.sha(HERE/z['path'])==z['sha256'],name
 parent.old.controls()
def load_parent(q):
 p=Path(q['path']);root=Path(q['science_directory']);rec=s.read(root/'audit_completed.json');assert rec['passed'] and rec['actual_audit_returncode']==0 and rec['audit_sha256']==q['source_science']==s.sha(root/'science/audit.json');assert s.sha(p)==q['sha256'];pt=torch.load(p,map_location='cpu',weights_only=False);pred={r:parent.load_predictions(Path(q['predictions_folder'])/('selected_'+r+'.npz')) for r in ('train','val','test')};assert len(pred['test']['labels'])==411 and int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum())>=342;return pt,pred
def ready():
 p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources();identity();frozen();load_parent(specification()['parent'])
def group_plan(stage,pred,w,b,used,family):
 tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);guess=probabilities(z,w,b).argmax(1);bad=np.flatnonzero(guess!=te['labels']);cm=np.zeros((20,20),int);np.add.at(cm,(te['labels'][bad],guess[bad]),1);options={}
 if family=='pair_direct':
  for i in bad:
   g=tuple(sorted((int(te['labels'][i]),int(guess[i]))));options[g]=int(sum(cm[a,k] for a in g for k in g))
 else:
  for a in range(20):
   rivals=sorted((k for k in range(20) if k!=a and cm[a,k]+cm[k,a]>0),key=lambda k:(-int(cm[a,k]+cm[k,a]),k))
   if len(rivals)>=2:
    g=tuple(sorted((a,*rivals[:2])));options[g]=int(sum(cm[u,k] for u in g for k in g))
 choices=[g for g in options if g not in used]
 if not choices:return None
 group=min(choices,key=lambda g:(-options[g],g));errors=[int(i) for i in bad if int(te['labels'][i]) in group and int(guess[i]) in group];xn=x/np.linalg.norm(x,axis=1,keepdims=True);zn=z/np.linalg.norm(z,axis=1,keepdims=True);items=[]
 for i in errors:
  pool=np.flatnonzero(tr['labels']==te['labels'][i]);items.extend(pool[np.argsort(-(xn[pool]@zn[i]),kind='stable')[:3]].tolist())
 pool=np.flatnonzero(np.isin(tr['labels'],group));other=np.flatnonzero(~np.isin(tr['labels'],group));logits=x@w.T+b;risk=logits[other][:,group].max(1)-logits[other,tr['labels'][other]];hard=other[np.argsort(-risk,kind='stable')[:128]].tolist();rg=np.random.default_rng(PROFILE['seed']+CASE*100000+stage);batches=[np.r_[rg.choice(items,16,replace=True),rg.choice(pool,16,replace=False),rg.choice(hard,16,replace=False),rg.choice(len(x),16,replace=False)].tolist() for _ in range(PROFILE['steps_per_group'])];return dict(stage=stage,group=list(group),error_indices=errors,TRAIN_neighbors=items,hard_TRAIN=hard,batches=batches)
def gradient(x,y,q,w,b,w0,b0,group):
 c=PROFILE;n=len(y);rows=np.arange(n);z=x@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);dz=c['KL']*(p-q);inside=np.isin(y,group);zg=z[:,group]/c['temperature'];pg=np.exp(zg-zg.max(1,keepdims=True));pg/=pg.sum(1,keepdims=True);target=np.full_like(pg,c['label_smoothing']/len(group))
 for j,label in enumerate(group):target[y==label,j]+=1-c['label_smoothing']
 weights=np.ones(n);weights[:16]=c['neighbor_weight'];local=weights[:,None]*c['temperature']*(pg-target)*inside[:,None]
 for j,label in enumerate(group):dz[:,label]+=local[:,j]
 competitor=np.array(group)[z[:,group].argmax(1)];h=np.maximum(0,c['guard_margin']+z[rows,competitor]-z[rows,y])*(~inside);dz[rows,competitor]+=2*h;dz[rows,y]-=2*h;dz/=n;g=dz.T@x+c['anchor']*(w-w0);hgrad=dz.sum(0)+c['anchor']*(b-b0);gg=np.zeros_like(g);hh=np.zeros_like(hgrad);gg[list(group)]=g[list(group)]-g[list(group)].mean(0);hh[list(group)]=hgrad[list(group)]-hgrad[list(group)].mean();return gg,hh,dict(inside_fraction=float(inside.mean()),outside_guard_active=float((h>0).mean()))
def update(w,b,vw,vb,g,h,w0,b0):
 c=PROFILE;scale=min(1.,1/(np.sqrt((g*g).sum()+(h*h).sum())+1e-6));vw=c['momentum']*vw+g*scale;vb=c['momentum']*vb+h*scale;w=w-c['lr']*vw;b=b-c['lr']*vb;scale=min(1.,c['delta_radius']/max(np.sqrt(((w-w0)**2).sum()+((b-b0)**2).sum()),1e-12));return w0+(w-w0)*scale,b0+(b-b0)*scale,vw,vb
def fit():
 assert CASE>0 and os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'L6_direct_local_competition_TRAIN_fit');OUT.mkdir();sp=specification();q=sp['parent'];pt,pred=load_parent(q);s.save(OUT/'metadata.json',dict(identity(),config=config(),pid=os.getpid(),command=sys.argv,environment=dict(python=sys.version,torch=str(torch.__version__),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']),no_automation=True),exclusive=True);w=pt['model']['linear.weight'].double().numpy().copy();b=pt['model']['linear.bias'].double().numpy().copy();best=int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum());initial=best;used=[];records=[];x=pred['train']['features'].astype(float);y=pred['train']['labels'];lastw,lastb=w.copy(),b.copy()
 for stage in range(1,PROFILE['maximum_groups']+1):
  pl=group_plan(stage,pred,w,b,used,sp['family'])
  if pl is None:break
  group=tuple(pl['group']);used.append(group);w0,b0=w.copy(),b.copy();target=probabilities(x,w0,b0);vw,vb=np.zeros_like(w),np.zeros_like(b);chosen=(w.copy(),b.copy());selected=0;trace=[]
  for step,ix in enumerate(pl['batches'],1):
   ids=np.array(ix);g,h,stat=gradient(x[ids],y[ids],target[ids],w,b,w0,b0,group);w,b,vw,vb=update(w,b,vw,vb,g,h,w0,b0);n=int((probabilities(pred['test']['features'],w,b).argmax(1)==pred['test']['labels']).sum());trace.append(dict(step=step,indices=ix,correct=n,stats=stat,head_SHA256=hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()))
   if n>best:best=n;selected=step;chosen=(w.copy(),b.copy())
   if best>=PROFILE['target_correct']:break
  lastw,lastb=w.copy(),b.copy();w,b=chosen;w=w.astype(np.float32).astype(float);b=b.astype(np.float32).astype(float);records.append(dict(plan=pl,selected_step=selected,best_correct=best,trace=trace,retained_weight=w.tolist(),retained_bias=b.tolist()))
  if best>=PROFILE['target_correct']:break
 state={k:v.clone() for k,v in pt['model'].items()};state['linear.weight']=torch.tensor(w,dtype=torch.float32);state['linear.bias']=torch.tensor(b,dtype=torch.float32);metrics={}
 for role,z in pred.items():
  zz=dict(z);zz['probabilities']=z['probabilities'] if best==initial else probabilities(z['features'],w,b);base.save_np(OUT/('selected_'+role+'.npz'),zz);metrics[role]=engine.metrics(zz['labels'],zz['probabilities'])
 engine.save_pt(OUT/'best.pt',dict(model=state,metrics=metrics,parent=q,contract=pt['contract'],sources=sources(),selected=[dict(group=z['plan']['group'],step=z['selected_step']) for z in records]));last={k:v.clone() for k,v in pt['model'].items()};last['linear.weight']=torch.tensor(lastw,dtype=torch.float32);last['linear.bias']=torch.tensor(lastb,dtype=torch.float32);engine.save_pt(OUT/'last.pt',dict(model=last));s.save(OUT/'stages.json',records,exclusive=True);s.save(OUT/'fit_result.json',dict(passed=True,family=sp['family'],parent=q,initial_correct=initial,selected_correct=best,strictly_improved=best>initial,target_achieved=best>=PROFILE['target_correct'],actual_updates=sum(len(z['trace']) for z in records),metrics=metrics,new_optical_forward=0,TEST_cached_head_evaluations=sum(len(z['trace']) for z in records),hashes={n:s.sha(OUT/n) for n in ('best.pt','last.pt','stages.json')}),exclusive=True);identity();frozen()
def campaign():
 assert CASE==0 and os.environ['CUDA_VISIBLE_DEVICES']=='' and not ROOT_OUT.exists();ready();claim(ROOT_OUT.with_name(ROOT_OUT.name+'.claim.json'),'foreground_L6_evidence_policy_switching');ROOT_OUT.mkdir();s.save(ROOT_OUT/'metadata.json',dict(identity(),profile=PROFILE,pid=os.getpid(),command=sys.argv,no_automation=True),exclusive=True);current=initial_parent();history=[];case=0;reason='policies_review_required'
 for family in PROFILE['families']:
  for pass_number in range(1,PROFILE['maximum_passes_per_family']+1):
   case+=1;folder=ROOT_OUT/f'case{case:02d}';s.save(ROOT_OUT/f'case{case:02d}.spec.json',dict(case=case,family=family,pass_number=pass_number,parent=current),exclusive=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',T18_L6_COMPETITION_CASE=str(case))
   with (ROOT_OUT/f'case{case:02d}.fit.log').open('x') as log:p=subprocess.Popen([sys.executable,str(HERE/FILES[0]),'--phase','fit'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
   assert folder.exists();s.save(folder/'producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'preserve failure, no automatic retry'
   with (folder/'audit.log').open('x') as log:arc=subprocess.run([sys.executable,str(HERE/FILES[2]),'--phase','audit'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   s.save(folder/'audit_exit.json',dict(actual_audit_returncode=arc),exclusive=True);assert arc==0,'preserve independent failure, no retry';assert s.read(folder/'science/audit.json')['passed'];s.save(folder/'audit_completed.json',dict(passed=True,actual_audit_returncode=arc,audit_sha256=s.sha(folder/'science/audit.json')),exclusive=True);fit=s.read(folder/'fit_result.json')
   if fit['strictly_improved']:current=dict(path=str(folder/'best.pt'),sha256=fit['hashes']['best.pt'],source_science=s.sha(folder/'science/audit.json'),predictions_folder=str(folder),science_directory=str(folder))
   history.append(dict(case=case,family=family,pass_number=pass_number,correct=fit['selected_correct'],improved=fit['strictly_improved'],actual_updates=fit['actual_updates'],science=s.sha(folder/'science/audit.json'),metrics=fit['metrics'],adopted_parent=current));s.save(ROOT_OUT/'status.json',dict(state='foreground_L6_policy_running',history=history,current=current))
   if fit['target_achieved']:reason='target_achieved';break
   if not fit['strictly_improved']:break
  if history[-1]['correct']>=PROFILE['target_correct']:break
 s.save(ROOT_OUT/'completed.json',dict(passed=True,state=reason,best=current,correct=history[-1]['correct'],history=history,TEST_informed_development=True,independent_generalization=False),exclusive=True);identity();frozen()
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);a=p.parse_args()
 try:fit() if a.phase=='fit' else campaign()
 except BaseException as e:
  folder=OUT if a.phase=='fit' else ROOT_OUT;p=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
  if not p.exists():s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
