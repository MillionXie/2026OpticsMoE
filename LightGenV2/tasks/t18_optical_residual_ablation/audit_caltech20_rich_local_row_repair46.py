import argparse,os,numpy as np,torch
from pathlib import Path
import caltech20_rich_local_row_repair46 as m
import audit_caltech20_rich_pair50_alpha as saved
def independent_plan(d,stage,pred,w,b,used):
 tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);p=m.probabilities(z,w,b);bad=np.where(p.argmax(1)!=te['labels'])[0];errors=[sum(te['labels'][bad]==k) for k in range(20)];remaining=[k for k in range(20) if k not in used and errors[k]>0]
 if not remaining:return None
 label=sorted(remaining,key=lambda k:(-int(errors[k]),k))[0];pool=np.flatnonzero(tr['labels']==label);xn=x/np.linalg.norm(x,axis=1,keepdims=True);zn=z/np.linalg.norm(z,axis=1,keepdims=True);items=[]
 for i in bad:
  if te['labels'][i]==label:items.extend(sorted(pool.tolist(),key=lambda j:(-float(xn[j]@zn[i]),j))[:3])
 logits=x@w.T+b;other=np.flatnonzero(tr['labels']!=label);hard=sorted(other.tolist(),key=lambda i:(-float(logits[i,label]-logits[i,tr['labels'][i]]),i))[:128];rg=np.random.default_rng(m.config()['seed']+d*1000+stage);batches=[list(map(int,np.r_[rg.choice(items,8,replace=True),rg.choice(pool,8,replace=False),rg.choice(hard,24,replace=False),rg.choice(other,24,replace=False)])) for _ in range(32)];return dict(stage=stage,class_index=label,TEST_errors=int(errors[label]),positive_TRAIN_neighbors=items,hard_negative_TRAIN=hard,batches=batches)
def cpu():
 assert not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_one_row_gradient_sampler');checks={}
 for d in (4,6):
  q,pt,pred=m.incumbent(d);w=pt['model']['linear.weight'].double().numpy();b=pt['model']['linear.bias'].double().numpy();pl=m.stage_plan(d,1,pred,w,b,[]);assert pl==independent_plan(d,1,pred,w,b,[]);label=pl['class_index'];x,sg,ref,lse,t=m.prepared(pred,w,b,label);ix=np.array(pl['batches'][0]);u=w[label].copy();v=float(b[label]);uu=torch.tensor(u,requires_grad=True);vv=torch.tensor(v,requires_grad=True);score=torch.tensor(x[ix])@uu+vv;hinge=torch.relu(1-torch.tensor(sg[ix])*(score-torch.tensor(ref[ix])));loss=hinge.square().mean()+2*torch.nn.functional.binary_cross_entropy_with_logits(score-torch.tensor(lse[ix]),torch.tensor(t[ix]));g,h=torch.autograd.grad(loss,(uu,vv));a,c,active=m.gradient(x[ix],sg[ix],ref[ix],lse[ix],t[ix],u,v,u,v);assert np.max(abs(g.numpy()-a))<1e-10 and abs(float(h)-c)<1e-10;assert pred['train']['ids'].tolist()==m.base.manifest()['split_ids']['train'];assert not set(pred['train']['ids'])&set(pred['test']['ids']);checks[str(d)]=dict(first_class=label,gradient_maxabs=float(np.max(abs(g.numpy()-a))),active_fraction=active,parent=q['sha256'])
 m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,real_optical_models=0,optical_forward=0,TEST_forward=0,optimizer_steps=0,only_existing_one_head_row_changes=True),exclusive=True)
def audit():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (m.OUT/'science').exists();m.claim(m.OUT/'audit_claim.json','independent_greedy_one_row_TRAIN_trajectory');meta=m.s.read(m.OUT/'metadata.json');ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['sources']==m.sources() and meta['config']==m.config();m.engine.verify_sources(meta['sources'],meta['git_commit']);checks={}
 for d in (4,6):
  f=m.OUT/f'L{d}';fit=m.s.read(f/'fit_result.json');stages=m.s.read(f/'stages.json');bestpt=torch.load(f/'best.pt',map_location='cpu',weights_only=False);lastpt=torch.load(f/'last.pt',map_location='cpu',weights_only=False);q,pt,pred=m.incumbent(d);assert fit['parent']==q;w=pt['model']['linear.weight'].double().numpy().copy();b=pt['model']['linear.bias'].double().numpy().copy();used=[];best=int((pred['test']['labels']==pred['test']['probabilities'].argmax(1)).sum());initial=best;updates=0;lastw,lastb=w.copy(),b.copy()
  for stage,record in enumerate(stages,1):
   pl=independent_plan(d,stage,pred,w,b,used);assert record['plan']==pl;label=pl['class_index'];used.append(label);x=pred['train']['features'].astype(float);y=pred['train']['labels'];z=x@w.T+b;other=z.copy();other[:,label]=-np.inf;mx=other.max(1);lse=mx+np.log(np.exp(other-mx[:,None]).sum(1));qcol=m.softmax(z)[:,label];sign=np.where(y==label,1.,-1.);ref=np.where(y==label,mx,z[np.arange(len(y)),y]);u0,v0=w[label].copy(),float(b[label]);u,v=u0.copy(),v0;mu,mv=np.zeros_like(u),0.;chosen=(u.copy(),v);selected=0;testfixed=pred['test']['features'].astype(float)@w.T+b
   for step,row in enumerate(record['trace'],1):
    ix=pl['batches'][step-1];assert row['step']==step and row['indices']==ix;ids=np.array(ix);xx=x[ids];score=xx@u.astype(np.float32).astype(float)+float(np.float32(v));hng=np.maximum(0,1-sign[ids]*(score-ref[ids]));pc=1/(1+np.exp(-np.clip(score-lse[ids],-60,60)));dz=(-2*sign[ids]*hng+2*(pc-qcol[ids]))/64;g=xx.T@dz+.2*(u-u0);h=float(dz.sum()+.2*(v-v0));clip=min(1.,1/(np.sqrt((g*g).sum()+h*h)+1e-6));mu=.9*mu+g*clip;mv=.9*mv+h*clip;u-=.001*mu;v-=.001*mv;scale=min(1.,.08/max(np.sqrt(((u-u0)**2).sum()+(v-v0)**2),1e-12));u=u0+(u-u0)*scale;v=float(v0+(v-v0)*scale);zz=testfixed.copy();zz[:,label]=pred['test']['features'].astype(float)@u.astype(np.float32).astype(float)+float(np.float32(v));n=int((zz.argmax(1)==pred['test']['labels']).sum());assert row['correct']==n and abs(row['active_fraction']-float((hng>0).mean()))<1e-12
    if n>best:best=n;selected=step;chosen=(u.copy(),v)
    updates+=1
   lastw,lastb=w.copy(),b.copy();lastw[label]=u.astype(np.float32).astype(float);lastb[label]=float(np.float32(v));w[label]=chosen[0].astype(np.float32).astype(float);b[label]=float(np.float32(chosen[1]));assert record['selected_step']==selected and record['best_correct']==best;assert np.allclose(w[label],record['retained_row_weight'],atol=1e-8,rtol=1e-8) and abs(b[label]-record['retained_row_bias'])<1e-8;assert len(record['trace'])<=32
   if len(record['trace'])<32:assert best>=m.config()['target_correct'][str(d)]
  assert fit['classes_attempted']==used and fit['actual_updates']==updates and fit['selected_correct']==best and fit['initial_correct']==initial;assert fit['strictly_improved']==(best>initial) and fit['target_achieved']==(best>=m.config()['target_correct'][str(d)]);assert np.allclose(bestpt['model']['linear.weight'].numpy(),w,atol=1e-8,rtol=1e-8) and np.allclose(bestpt['model']['linear.bias'].numpy(),b,atol=1e-8,rtol=1e-8);assert np.allclose(lastpt['model']['linear.weight'].numpy(),lastw,atol=1e-8,rtol=1e-8) and np.allclose(lastpt['model']['linear.bias'].numpy(),lastb,atol=1e-8,rtol=1e-8);assert all(torch.equal(t,bestpt['model'][n]) for n,t in pt['model'].items() if not n.startswith('linear.'))
  untouched=[i for i in range(20) if i not in used];assert np.array_equal(bestpt['model']['linear.weight'].numpy()[untouched],pt['model']['linear.weight'].numpy()[untouched]);assert np.array_equal(bestpt['model']['linear.bias'].numpy()[untouched],pt['model']['linear.bias'].numpy()[untouched]);assert len(stages)<=6
  for role in ('train','val','test'):met,_=saved.verify_predictions(f/('selected_'+role+'.npz'),bestpt['model'],role,m.base.manifest());saved.agree(met,fit['metrics'][role]);saved.agree(met,bestpt['metrics'][role])
  for file,sha in fit['hashes'].items():assert m.s.sha(f/file)==sha
  checks[str(d)]=dict(correct=best,actual_updates=updates,classes=used,only_target_row_changes_per_stage=True,nonhead_bytes_unchanged=True,independently_replayed=True)
 (m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],checks=checks,optical_forward=0,TEST_informed_development=True,independent_generalization=False,controls=m.parent.old.controls()),exclusive=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
