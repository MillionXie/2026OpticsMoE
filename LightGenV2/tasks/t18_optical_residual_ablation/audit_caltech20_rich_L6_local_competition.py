"""Independent sampler, F64 derivative, group projection and full cached-head trajectory."""
import argparse,os,hashlib,numpy as np,torch
from pathlib import Path
import caltech20_rich_L6_local_competition as m
import audit_caltech20_rich_pair50_alpha as saved
def independent_plan(stage,pred,w,b,used,family):
 tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);yh=m.probabilities(z,w,b).argmax(1);bad=[i for i in range(len(yh)) if yh[i]!=te['labels'][i]];cm=np.zeros((20,20),int)
 for i in bad:cm[te['labels'][i],yh[i]]+=1
 candidates={}
 if family=='pair_direct':
  for i in bad:
   g=tuple(sorted((int(te['labels'][i]),int(yh[i]))));candidates[g]=int(cm[g[0],g[1]]+cm[g[1],g[0]])
 else:
  for a in range(20):
   ranked=sorted([k for k in range(20) if k!=a and cm[a,k]+cm[k,a]>0],key=lambda k:(-int(cm[a,k]+cm[k,a]),k))
   if len(ranked)>=2:
    g=tuple(sorted([a,ranked[0],ranked[1]]));candidates[g]=sum(int(cm[j,k]) for j in g for k in g)
 choices=sorted((g for g in candidates if g not in used),key=lambda g:(-candidates[g],g))
 if not choices:return None
 g=choices[0];errors=[i for i in bad if int(te['labels'][i]) in g and int(yh[i]) in g];xn=x/np.linalg.norm(x,axis=1,keepdims=True);zn=z/np.linalg.norm(z,axis=1,keepdims=True);items=[]
 for i in errors:
  possible=np.flatnonzero(tr['labels']==te['labels'][i]).tolist();items.extend(sorted(possible,key=lambda j:(-float(xn[j]@zn[i]),j))[:3])
 pool=np.flatnonzero(np.isin(tr['labels'],g));other=np.flatnonzero(~np.isin(tr['labels'],g));z=x@w.T+b;hard=sorted(other.tolist(),key=lambda i:(-float(max(z[i,list(g)])-z[i,tr['labels'][i]]),i))[:128];rg=np.random.default_rng(m.PROFILE['seed']+m.CASE*100000+stage);batches=[list(map(int,np.r_[rg.choice(items,16,replace=True),rg.choice(pool,16,replace=False),rg.choice(hard,16,replace=False),rg.choice(len(x),16,replace=False)])) for _ in range(m.PROFILE['steps_per_group'])];return dict(stage=stage,group=list(g),error_indices=errors,TRAIN_neighbors=items,hard_TRAIN=hard,batches=batches)
def independent_gradient(x,y,q,w,b,w0,b0,group):
 c=m.PROFILE;n=len(y);z=x@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);dz=c['KL']*(p-q);small=z[:,group]/c['temperature'];prob=np.exp(small-small.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);inside=np.isin(y,group);guard=[]
 for i in range(n):
  if inside[i]:
   target=np.full(len(group),c['label_smoothing']/len(group));target[group.index(int(y[i]))]+=1-c['label_smoothing'];weight=c['neighbor_weight'] if i<16 else 1.;dz[i,list(group)]+=weight*c['temperature']*(prob[i]-target);guard.append(False)
  else:
   rival=int(group[int(z[i,list(group)].argmax())]);hinge=max(0.,c['guard_margin']+z[i,rival]-z[i,y[i]]);dz[i,rival]+=2*hinge;dz[i,y[i]]-=2*hinge;guard.append(hinge>0)
 dz/=n;g=dz.T@x+c['anchor']*(w-w0);h=dz.sum(0)+c['anchor']*(b-b0);gg=np.zeros_like(g);hh=np.zeros_like(h);gg[list(group)]=g[list(group)]-g[list(group)].mean(0);hh[list(group)]=h[list(group)]-h[list(group)].mean();return gg,hh,dict(inside_fraction=float(inside.mean()),outside_guard_active=float(np.mean(guard)))
def cpu():
 assert m.CASE==0 and not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_pair_triple_local_objective');m.frozen();q=m.initial_parent();pt,pred=m.load_parent(q);w=pt['model']['linear.weight'].double().numpy();b=pt['model']['linear.bias'].double().numpy();checks={};c=m.PROFILE
 for family in c['families']:
  pl=m.group_plan(1,pred,w,b,[],family);assert pl==independent_plan(1,pred,w,b,[],family);group=pl['group'];ix=np.array(pl['batches'][0]);x=pred['train']['features'][ix].astype(float);y=pred['train']['labels'][ix];qprob=m.probabilities(x,w,b);ww=torch.tensor(w,requires_grad=True);bb=torch.tensor(b,dtype=torch.float64,requires_grad=True);z=torch.tensor(x)@ww.T+bb;inside=np.isin(y,group);t=np.full((64,len(group)),c['label_smoothing']/len(group))
  for j,k in enumerate(group):t[y==k,j]+=1-c['label_smoothing']
  weight=np.ones(64);weight[:16]=c['neighbor_weight'];ce=-(torch.tensor(t)*torch.log_softmax(z[:,group]/c['temperature'],1)).sum(1);loss=(ce*torch.tensor(weight*inside)).mean()*c['temperature']**2+c['KL']*(-torch.tensor(qprob)*torch.log_softmax(z,1)).sum(1).mean();guard=torch.relu(c['guard_margin']+z[:,group].max(1).values-z[torch.arange(64),torch.tensor(y)]);loss=loss+(guard.square()*torch.tensor(~inside)).mean();gw,gb=torch.autograd.grad(loss,(ww,bb));g=np.zeros_like(w);h=np.zeros_like(b);g[group]=gw.numpy()[group]-gw.numpy()[group].mean(0);h[group]=gb.numpy()[group]-gb.numpy()[group].mean();pg,ph,stats=m.gradient(x,y,qprob,w,b,w,b,group);ig,ih,istat=independent_gradient(x,y,qprob,w,b,w,b,group);err=max(float(np.max(abs(pg-g))),float(np.max(abs(ph-h))),float(np.max(abs(ig-g))),float(np.max(abs(ih-h))));assert err<1e-10 and stats==istat and np.linalg.norm(pg)>0;assert np.max(abs(pg.sum(0)))<1e-12 and abs(ph.sum())<1e-12;others=[k for k in range(20) if k not in group];assert np.count_nonzero(pg[others])==0 and np.count_nonzero(ph[others])==0;assert not set(pred['train']['ids'])&set(pred['test']['ids']);checks[family]=dict(group=group,gradient_maxabs=err,only_selected_rows=True,zero_mean_projection=True)
 m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,parent=q,checks=checks,real_optical_models=0,optical_forward=0,TEST_forward=0,optimizer_steps=0),exclusive=True);m.frozen()
def audit():
 assert m.CASE>0 and os.environ['CUDA_VISIBLE_DEVICES']=='' and not (m.OUT/'science').exists();m.claim(m.OUT/'audit_claim.json','independent_complete_L6_local_competition');meta=m.s.read(m.OUT/'metadata.json');ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['sources']==m.sources() and meta['config']==m.config();m.engine.verify_sources(meta['sources'],meta['git_commit']);fit=m.s.read(m.OUT/'fit_result.json');records=m.s.read(m.OUT/'stages.json');bestpt=torch.load(m.OUT/'best.pt',map_location='cpu',weights_only=False);lastpt=torch.load(m.OUT/'last.pt',map_location='cpu',weights_only=False);spec=m.specification();q=spec['parent'];assert fit['parent']==q;pt,pred=m.load_parent(q);w=pt['model']['linear.weight'].double().numpy().copy();b=pt['model']['linear.bias'].double().numpy().copy();initial=int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum());best=initial;used=[];updates=0;x=pred['train']['features'].astype(float);y=pred['train']['labels'];lastw,lastb=w.copy(),b.copy();c=m.PROFILE
 for stage,record in enumerate(records,1):
  plan=independent_plan(stage,pred,w,b,used,spec['family']);assert plan==record['plan'];group=tuple(plan['group']);used.append(group);w0,b0=w.copy(),b.copy();z=x@w0.T+b0;target=np.exp(z-z.max(1,keepdims=True));target/=target.sum(1,keepdims=True);vw,vb=np.zeros_like(w),np.zeros_like(b);chosen=(w.copy(),b.copy());selected=0
  for step,row in enumerate(record['trace'],1):
   ix=np.array(plan['batches'][step-1]);assert row['indices']==ix.tolist() and row['step']==step;g,h,stats=independent_gradient(x[ix],y[ix],target[ix],w,b,w0,b0,list(group));scale=min(1.,1/(np.sqrt((g*g).sum()+(h*h).sum())+1e-6));vw=c['momentum']*vw+g*scale;vb=c['momentum']*vb+h*scale;w-=c['lr']*vw;b-=c['lr']*vb;scale=min(1.,c['delta_radius']/max(np.sqrt(((w-w0)**2).sum()+((b-b0)**2).sum()),1e-12));w=w0+(w-w0)*scale;b=b0+(b-b0)*scale;n=int((m.probabilities(pred['test']['features'],w,b).argmax(1)==pred['test']['labels']).sum());assert n==row['correct'] and stats==row['stats'];assert hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()==row['head_SHA256'];others=[k for k in range(20) if k not in group];assert np.array_equal(w[others],w0[others]) and np.array_equal(b[others],b0[others]);assert np.max(abs((w-w0)[list(group)].sum(0)))<1e-12 and abs((b-b0)[list(group)].sum())<1e-12
   if n>best:best=n;selected=step;chosen=(w.copy(),b.copy())
   updates+=1
  lastw,lastb=w.copy(),b.copy();w,b=chosen;w=w.astype(np.float32).astype(float);b=b.astype(np.float32).astype(float);assert record['selected_step']==selected and record['best_correct']==best;assert np.array_equal(w,np.array(record['retained_weight'])) and np.array_equal(b,np.array(record['retained_bias']));assert len(record['trace'])<=c['steps_per_group'] and (len(record['trace'])==c['steps_per_group'] or best>=c['target_correct'])
 assert len(records)<=c['maximum_groups'] and fit['initial_correct']==initial and fit['selected_correct']==best and fit['actual_updates']==updates;assert fit['strictly_improved']==(best>initial) and fit['target_achieved']==(best>=c['target_correct']);assert np.array_equal(bestpt['model']['linear.weight'].numpy(),w) and np.array_equal(bestpt['model']['linear.bias'].numpy(),b);assert np.array_equal(lastpt['model']['linear.weight'].numpy(),lastw.astype(np.float32)) and np.array_equal(lastpt['model']['linear.bias'].numpy(),lastb.astype(np.float32));assert all(torch.equal(v,bestpt['model'][k]) for k,v in pt['model'].items() if not k.startswith('linear.'))
 for role in ('train','val','test'):metrics,_=saved.verify_predictions(m.OUT/('selected_'+role+'.npz'),bestpt['model'],role,m.base.manifest());saved.agree(metrics,fit['metrics'][role]);saved.agree(metrics,bestpt['metrics'][role])
 for file,sha in fit['hashes'].items():assert m.s.sha(m.OUT/file)==sha
 m.frozen();(m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],case=m.CASE,family=spec['family'],correct=best,actual_updates=updates,sampler_and_full_gradient_trajectory_replayed=True,nonhead_bytes_unchanged=True,only_group_rows_updated=True,L2_L4_not_loaded_or_trained=True,optical_forward=0,TEST_informed_development=True,independent_generalization=False),exclusive=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
