import os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_margin_guard_readout46 as m
from check_caltech20_rich_margin_guard_readout46 import independent_plan
import audit_caltech20_rich_pair50_alpha as saved
def run():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (m.OUT/'science').exists();m.claim(m.OUT/'audit_claim.json','independent_complete_margin_trajectory');meta=m.s.read(m.OUT/'metadata.json');ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['sources']==m.sources() and meta['config']==m.config();m.engine.verify_sources(meta['sources'],meta['git_commit']);records={}
 for d in (4,6):
  folder=m.OUT/f'L{d}';fit=m.s.read(folder/'fit_result.json');best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);trace=m.s.read(folder/'trace.json');scores=m.s.read(folder/'scores.json');q,pt,pred=m.incumbent(d);assert fit['parent']==q;bad,items,batches=independent_plan(pred);pl=m.s.read(folder/'sampling_plan.json');assert (bad,items,batches)==(pl['TEST_error_indices'],pl['TRAIN_neighbor_multiset'],pl['batches']);assert len(trace)==fit['actual_head_steps']==128 and len(scores)==129
  x=pred['train']['features'].astype(float);labels=pred['train']['labels'];w0=pt['model']['linear.weight'].double().numpy();b0=pt['model']['linear.bias'].double().numpy();z=x@w0.T+b0;t=np.exp(z-z.max(1,keepdims=True));t/=t.sum(1,keepdims=True);w,b=w0.copy(),b0.copy();vw,vb=np.zeros_like(w),np.zeros_like(b);correct=[];active=[]
  for step in range(129):
   if step:
    ix=batches[step-1];xx=x[ix];y=labels[ix];z=xx@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);competitor=z.copy();competitor[np.arange(64),y]=-np.inf;j=competitor.argmax(1);hng=np.maximum(0,1+z[np.arange(64),j]-z[np.arange(64),y]);dz=2*(p-t[ix]);
    for i in range(64):dz[i,j[i]]+=2*hng[i];dz[i,y[i]]-=2*hng[i]
    dz/=64;g=dz.T@xx+.2*(w-w0);h=dz.sum(0)+.2*(b-b0);clip=min(1.,1/(np.sqrt((g*g).sum()+(h*h).sum())+1e-6));vw=.9*vw+g*clip;vb=.9*vb+h*clip;w-=.001*vw;b-=.001*vb;scale=min(1.,.1/max(np.sqrt(((w-w0)**2).sum()+((b-b0)**2).sum()),1e-12));w=w0+(w-w0)*scale;b=b0+(b-b0)*scale;assert trace[step-1]['indices']==ix and abs(trace[step-1]['active_margin_fraction']-float((hng>0).mean()))<1e-12;active.append(float((hng>0).mean()))
   p=pred['test']['probabilities'] if step==0 else m.probabilities(pred['test']['features'],w,b);n=int((p.argmax(1)==pred['test']['labels']).sum());correct.append(n);assert scores[step]==dict(step=step,correct=n)
   if step==best['selected_step']:assert np.allclose(best['head_float64']['weight'],w,atol=1e-9,rtol=1e-9) and np.allclose(best['head_float64']['bias'],b,atol=1e-9,rtol=1e-9)
  assert best['selected_step']==fit['selected_step']==int(np.argmax(correct));assert np.allclose(last['head_float64']['weight'],w,atol=1e-9,rtol=1e-9);assert abs(np.mean(active)-fit['mean_active_TRAIN_margin_fraction'])<1e-12;assert all(torch.equal(v,best['model'][n]) for n,v in pt['model'].items() if not n.startswith('linear.'))
  for role in ('train','val','test'):met,_=saved.verify_predictions(folder/('selected_'+role+'.npz'),best['model'],role,m.base.manifest());saved.agree(met,fit['metrics'][role]);saved.agree(met,best['metrics'][role])
  for file,sha in fit['hashes'].items():assert m.s.sha(folder/file)==sha
  records[str(d)]=dict(selected_step=best['selected_step'],correct=max(correct),all_128_steps_replayed=True,nonhead_bytes_unchanged=True,mean_active_fraction=float(np.mean(active)))
 (m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],checks=records,optical_forward=0,TEST_informed_development=True,independent_generalization=False,controls=m.parent.old.controls()),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
