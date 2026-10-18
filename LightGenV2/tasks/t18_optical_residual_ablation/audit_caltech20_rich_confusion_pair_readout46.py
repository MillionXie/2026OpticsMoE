"""Independent pair sampler, projected gradients and complete saved-state trajectory."""
import argparse,os,collections,numpy as np,torch,hashlib
from pathlib import Path
import caltech20_rich_confusion_pair_readout46 as m
import audit_caltech20_rich_pair50_alpha as saved
def independent_plan(d,stage,pred,w,b,used):
 tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);guess=m.probabilities(z,w,b).argmax(1);cnt={};errors=[]
 for i,(y,h) in enumerate(zip(te['labels'],guess)):
  if y!=h:
   p=tuple(sorted((int(y),int(h))));cnt[p]=cnt.get(p,0)+1;errors.append(i)
 options=sorted((p for p in cnt if p not in used),key=lambda p:(-cnt[p],p))
 if not options:return None
 pair=options[0];bad=[i for i in errors if tuple(sorted((int(te['labels'][i]),int(guess[i]))))==pair];xn=x/np.linalg.norm(x,axis=1,keepdims=True);zn=z/np.linalg.norm(z,axis=1,keepdims=True);items=[]
 for i in bad:
  pool=np.flatnonzero(tr['labels']==te['labels'][i]);items.extend(sorted(pool.tolist(),key=lambda j:(-float(xn[j]@zn[i]),j))[:3])
 pool=np.flatnonzero(np.isin(tr['labels'],pair));other=np.flatnonzero(~np.isin(tr['labels'],pair));logits=x@w.T+b;hard=sorted(other.tolist(),key=lambda j:(-float(max(logits[j,list(pair)])-logits[j,tr['labels'][j]]),j))[:128];rg=np.random.default_rng(m.config()['seed']+1000*d+stage);batches=[list(map(int,np.r_[rg.choice(items,16,replace=True),rg.choice(pool,16,replace=False),rg.choice(hard,16,replace=False),rg.choice(len(x),16,replace=False)])) for _ in range(m.config()['steps_per_pair'])];return dict(stage=stage,pair=list(pair),TEST_error_indices=bad,TRAIN_neighbor_multiset=items,hard_negative_TRAIN=hard,batches=batches)
def cpu():
 assert not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_pair_sampler_and_projected_gradient');checks={}
 for d in (4,6):
  q,pt,pred=m.incumbent(d);w=pt['model']['linear.weight'].double().numpy();b=pt['model']['linear.bias'].double().numpy();pl=m.plan(d,1,pred,w,b,[]);assert pl==independent_plan(d,1,pred,w,b,[]);ids=np.array(pl['batches'][0]);x=pred['train']['features'][ids].astype(float);y=pred['train']['labels'][ids];target=m.probabilities(x,w,b);ww=torch.tensor(w,requires_grad=True);bb=torch.tensor(b,dtype=torch.float64,requires_grad=True);z=torch.tensor(x)@ww.T+bb;other=z.clone();other[torch.arange(64),torch.tensor(y)]=-torch.inf;loss=torch.relu(1+other.max(1).values-z[torch.arange(64),torch.tensor(y)]).square().mean()+2*(-torch.tensor(target)*torch.log_softmax(z,1)).sum(1).mean();gg,hh=torch.autograd.grad(loss,(ww,bb));g,h,active=m.gradient(x,y,target,w,b,w,b);assert np.max(abs(g-gg.numpy()))<1e-10 and np.max(abs(h-hh.numpy()))<1e-10;pg,ph=m.projected(g,h,pl['pair']);a,c=pl['pair'];expected=(gg.numpy()[a]-gg.numpy()[c])/2;assert np.max(abs(pg[a]-expected))<1e-10 and np.array_equal(pg[a],-pg[c]);others=[j for j in range(20) if j not in pl['pair']];assert np.count_nonzero(pg[others])==0 and np.count_nonzero(ph[others])==0;assert not set(pred['train']['ids'])&set(pred['test']['ids']);checks[str(d)]=dict(pair=pl['pair'],gradient_maxabs=float(np.max(abs(g-gg.numpy()))),only_two_rows_projected=True,parent=q['sha256'])
 m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,real_optical_models=0,optical_forward=0,TEST_forward=0,optimizer_steps=0),exclusive=True)
def audit():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (m.OUT/'science').exists();m.claim(m.OUT/'audit_claim.json','independent_complete_symmetric_pair_trajectory');meta=m.s.read(m.OUT/'metadata.json');ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['sources']==m.sources() and meta['config']==m.config();m.engine.verify_sources(meta['sources'],meta['git_commit']);checks={};c=m.config()
 for d in (4,6):
  folder=m.OUT/f'L{d}';fit=m.s.read(folder/'fit_result.json');stages=m.s.read(folder/'stages.json');bestpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);lastpt=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);q,pt,pred=m.incumbent(d);assert q==fit['parent'];w=pt['model']['linear.weight'].double().numpy().copy();b=pt['model']['linear.bias'].double().numpy().copy();best=int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum());initial=best;used=[];updates=0;x=pred['train']['features'].astype(float);y=pred['train']['labels'];lastw,lastb=w.copy(),b.copy()
  for stage,record in enumerate(stages,1):
   pl=independent_plan(d,stage,pred,w,b,used);assert pl==record['plan'];a,k=pl['pair'];used.append((a,k));w0,b0=w.copy(),b.copy();vw,vb=np.zeros_like(w),np.zeros_like(b);z=x@w0.T+b0;t=np.exp(z-z.max(1,keepdims=True));t/=t.sum(1,keepdims=True);selected=0;chosen=(w.copy(),b.copy())
   for step,row in enumerate(record['trace'],1):
    ids=np.array(pl['batches'][step-1]);assert row['indices']==ids.tolist() and row['step']==step;xx=x[ids];yy=y[ids];z=xx@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);rival=z.copy();rival[np.arange(64),yy]=-np.inf;j=rival.argmax(1);hinge=np.maximum(0,1+z[np.arange(64),j]-z[np.arange(64),yy]);dz=2*(p-t[ids]);dz[np.arange(64),j]+=2*hinge;dz[np.arange(64),yy]-=2*hinge;dz/=64;gg=dz.T@xx+.2*(w-w0);hh=dz.sum(0)+.2*(b-b0);g=np.zeros_like(gg);h=np.zeros_like(hh);g[a]=(gg[a]-gg[k])/2;g[k]=-g[a];h[a]=(hh[a]-hh[k])/2;h[k]=-h[a];clip=min(1.,1/(np.sqrt((g*g).sum()+(h*h).sum())+1e-6));vw=.9*vw+g*clip;vb=.9*vb+h*clip;w-=.001*vw;b-=.001*vb;scale=min(1.,.1/max(np.sqrt(((w-w0)**2).sum()+((b-b0)**2).sum()),1e-12));w=w0+(w-w0)*scale;b=b0+(b-b0)*scale;n=int((m.probabilities(pred['test']['features'],w,b).argmax(1)==pred['test']['labels']).sum());assert n==row['correct'] and abs(float((hinge>0).mean())-row['active_fraction'])<1e-12;assert hashlib.sha256(w.tobytes()+b.tobytes()).hexdigest()==row['head_SHA256'];others=[i for i in range(20) if i not in (a,k)];assert np.array_equal(w[others],w0[others]) and np.array_equal(b[others],b0[others]);assert np.max(abs(w[a]+w[k]-w0[a]-w0[k]))<1e-12 and abs(b[a]+b[k]-b0[a]-b0[k])<1e-12
    if n>best:best=n;selected=step;chosen=(w.copy(),b.copy())
    updates+=1
   lastw,lastb=w.copy(),b.copy();w,b=chosen;w=w.astype(np.float32).astype(float);b=b.astype(np.float32).astype(float);assert record['selected_step']==selected and record['best_correct']==best;assert np.array_equal(w,np.array(record['retained_weight'])) and np.array_equal(b,np.array(record['retained_bias']));assert len(record['trace'])<=c['steps_per_pair'];assert len(record['trace'])==c['steps_per_pair'] or best>=c['target_correct'][str(d)]
  assert len(stages)<=c['maximum_pairs'] and fit['actual_updates']==updates and fit['pairs_attempted']==[list(z) for z in used] and fit['selected_correct']==best and fit['initial_correct']==initial;assert fit['strictly_improved']==(best>initial) and fit['target_achieved']==(best>=c['target_correct'][str(d)]);assert np.array_equal(bestpt['model']['linear.weight'].numpy(),w) and np.array_equal(bestpt['model']['linear.bias'].numpy(),b);assert np.array_equal(lastpt['model']['linear.weight'].numpy(),lastw.astype(np.float32)) and np.array_equal(lastpt['model']['linear.bias'].numpy(),lastb.astype(np.float32));assert all(torch.equal(v,bestpt['model'][n]) for n,v in pt['model'].items() if not n.startswith('linear.'))
  for role in ('train','val','test'):met,_=saved.verify_predictions(folder/('selected_'+role+'.npz'),bestpt['model'],role,m.base.manifest());saved.agree(met,fit['metrics'][role]);saved.agree(met,bestpt['metrics'][role])
  for name,sha in fit['hashes'].items():assert m.s.sha(folder/name)==sha
  checks[str(d)]=dict(correct=best,actual_updates=updates,pairs=used,nonhead_bytes_unchanged=True,only_two_rows_change_each_stage=True,independently_replayed=True)
 (m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],checks=checks,TEST_informed_development=True,independent_generalization=False,optical_forward=0,controls=m.parent.old.controls()),exclusive=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','audit'],required=True);a=p.parse_args()
 try:cpu() if a.phase=='cpu' else audit()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json') if a.phase=='cpu' else m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
