"""Independently replay every cached-head step and verify unchanged non-head weights."""
import hashlib,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_error_neighbor_readout46 as m
import audit_caltech20_rich_pair50_alpha as saved
def run():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (m.OUT/'science').exists();m.claim(m.OUT/'audit_claim.json','independent_complete_head_trajectory');meta=m.s.read(m.OUT/'metadata.json');ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['sources']==m.sources() and meta['config']==m.CFG;m.engine.verify_sources(meta['sources'],meta['git_commit']);records={}
 for d in (4,6):
  folder=m.OUT/f'L{d}';fit=m.s.read(folder/'fit_result.json');q,pt,pred=m.parent.incumbent(d);plan=m.s.read(folder/'sampling_plan.json');tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);xn=x/np.linalg.norm(x,axis=1,keepdims=True);z=te['features'].astype(float);z/=np.linalg.norm(z,axis=1,keepdims=True);bad=np.where(te['labels']!=te['probabilities'].argmax(1))[0].tolist();items=[]
  for i in bad:
   group=np.flatnonzero(tr['labels']==te['labels'][i]);rank=sorted(group.tolist(),key=lambda j:(-float(xn[j]@z[i]),j));items.extend(rank[:3])
  assert bad==plan['error_TEST_indices'] and items==plan['TRAIN_neighbor_multiset'];rg=np.random.default_rng(m.CFG['seed']);batches=[list(map(int,np.r_[rg.choice(items,32,replace=True),rg.choice(3141,32,replace=False)])) for _ in range(128)];assert batches==plan['TRAIN_batches'];w0=pt['model']['linear.weight'].double().numpy();b0=pt['model']['linear.bias'].double().numpy();w,b=w0.copy(),b0.copy();vw,vb=np.zeros_like(w),np.zeros_like(b);a=x@w0.T+b0;t=np.exp(a-a.max(1,keepdims=True));t/=t.sum(1,keepdims=True);trace=m.s.read(folder/'trace.json');snaps=torch.load(folder/'snapshots.pt',map_location='cpu',weights_only=False);choices=[(0,int((te['labels']==te['probabilities'].argmax(1)).sum()))];states={0:pt['model']}
  for step,ix in enumerate(batches,1):
   xx=x[ix];label=np.eye(20)[tr['labels'][ix]]*.98+.001;ww=w.astype(np.float32).astype(float);bb=b.astype(np.float32).astype(float);a=xx@ww.T+bb;p=np.exp(a-a.max(1,keepdims=True));p/=p.sum(1,keepdims=True);delta=(2*p-label-t[ix])/64;gw=delta.T@xx+.1*(w-w0);gb=delta.sum(0)+.1*(b-b0);norm=np.sqrt((gw*gw).sum()+(gb*gb).sum());coef=min(1.,1/(norm+1e-6));vw=.9*vw+gw*coef;vb=.9*vb+gb*coef;w-=.001*vw;b-=.001*vb;radius=np.sqrt(((w-w0)**2).sum()+((b-b0)**2).sum());shrink=min(1.,.1/max(radius,1e-12));w=w0+(w-w0)*shrink;b=b0+(b-b0)*shrink;assert trace[step-1]['indices']==ix
   # Algebraic reassociation may change last float64 bits; compare saved checkpoints numerically.
   if step%32==0:
    snap=snaps[step//32-1];assert snap['step']==step and np.allclose(snap['weight'],w,atol=1e-10,rtol=1e-10) and np.allclose(snap['bias'],b,atol=1e-10,rtol=1e-10);zz=m.parent.load_predictions(folder/f'test_step{step:03d}.npz');state=dict(pt['model']);state['linear.weight']=torch.tensor(snap['weight'],dtype=torch.float32);state['linear.bias']=torch.tensor(snap['bias'],dtype=torch.float32);met,_=saved.verify_predictions(folder/f'test_step{step:03d}.npz',state,'test',m.base.manifest());n=int((zz['labels']==zz['probabilities'].argmax(1)).sum());assert n==snap['correct'];choices.append((step,n));states[step]=state
  best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);chosen=max(choices,key=lambda z:z[1])[0];assert chosen==best['selected_step']==fit['selected_step'];assert m.s.digest(states[chosen])==m.s.digest(best['model']);assert all(torch.equal(v,best['model'][n]) for n,v in pt['model'].items() if not n.startswith('linear.'));assert fit['actual_head_steps']==128 and fit['new_optical_forward']==fit['new_optical_steps']==0
  for role in ('train','val','test'):met,_=saved.verify_predictions(folder/('selected_'+role+'.npz'),best['model'],role,m.base.manifest());saved.agree(met,fit['metrics'][role]);saved.agree(met,best['metrics'][role])
  for key,name in (('trace_sha256','trace.json'),('sampling_sha256','sampling_plan.json'),('best_sha256','best.pt'),('snapshots_sha256','snapshots.pt')):assert fit[key]==m.s.sha(folder/name)
  records[str(d)]=dict(selected_step=chosen,correct=next(n for st,n in choices if st==chosen),all_128_updates_replayed=True,nonhead_bytes_unchanged=True,TEST_cached_evaluations=4)
 (m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],checks=records,optical_forward=0,TEST_informed_development=True,independent_generalization=False,controls=m.parent.old.controls()),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
