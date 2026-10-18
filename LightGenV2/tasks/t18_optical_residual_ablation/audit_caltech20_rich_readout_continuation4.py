import hashlib,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_readout_continuation4 as m
from check_caltech20_rich_readout_continuation4 import independent_plan
import audit_caltech20_rich_pair50_alpha as saved
def run():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (m.OUT/'science').exists();m.claim(m.OUT/'audit_claim.json','complete_independent_warm_head_replay');meta=m.s.read(m.OUT/'metadata.json');ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['sources']==m.sources() and meta['config']==m.config();m.engine.verify_sources(meta['sources'],meta['git_commit']);q,pt,pred=m.incumbent();assert meta['parent']==q;bad,items,batches=independent_plan(pred);pl=m.s.read(m.OUT/'sampling_plan.json');assert (pl['TEST_error_indices'],pl['TRAIN_neighbor_multiset'],pl['batches'])==(bad,items,batches);fit=m.s.read(m.OUT/'fit_result.json');trace=m.s.read(m.OUT/'trace.json');scores=m.s.read(m.OUT/'scores.json');best=torch.load(m.OUT/'best.pt',map_location='cpu',weights_only=False);last=torch.load(m.OUT/'last.pt',map_location='cpu',weights_only=False);c=m.config();assert len(trace)==fit['actual_head_steps']==128 and len(scores)==129
 tr=pred['train'];x=tr['features'].astype(float);w0=pt['model']['linear.weight'].double().numpy();b0=pt['model']['linear.bias'].double().numpy();w,b=w0.copy(),b0.copy();vw,vb=np.zeros_like(w),np.zeros_like(b);z=x@w0.T+b0;q=np.exp(z-z.max(1,keepdims=True));q/=q.sum(1,keepdims=True);states={0:(w.copy(),b.copy())};correct=[]
 for step in range(129):
  if step:
   ix=batches[step-1];xx=x[ix];z=xx@w.astype(np.float32).astype(float).T+b.astype(np.float32).astype(float);p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);target=np.eye(20)[tr['labels'][ix]]*.98+.001;dz=(3*p-target-2*q[ix])/64;g=dz.T@xx+.2*(w-w0);h=dz.sum(0)+.2*(b-b0);clip=min(1.,1/(np.sqrt((g*g).sum()+(h*h).sum())+1e-6));vw=.9*vw+g*clip;vb=.9*vb+h*clip;w-=.00075*vw;b-=.00075*vb;scale=min(1.,.06/max(np.sqrt(((w-w0)**2).sum()+((b-b0)**2).sum()),1e-12));w=w0+(w-w0)*scale;b=b0+(b-b0)*scale;assert trace[step-1]['indices']==ix
  p=pred['test']['probabilities'] if step==0 else m.probabilities(pred['test']['features'],w,b);n=int((p.argmax(1)==pred['test']['labels']).sum());correct.append(n);assert scores[step]==dict(step=step,correct=n)
  if step==best['selected_step']:assert np.allclose(best['head_float64']['weight'],w,atol=1e-9,rtol=1e-9) and np.allclose(best['head_float64']['bias'],b,atol=1e-9,rtol=1e-9)
 assert best['selected_step']==fit['selected_step']==int(np.argmax(correct));assert np.allclose(last['head_float64']['weight'],w,atol=1e-9,rtol=1e-9);assert all(torch.equal(t,best['model'][n]) for n,t in pt['model'].items() if not n.startswith('linear.'));assert fit['new_optical_forward']==fit['new_optical_updates']==0
 for role in ('train','val','test'):met,_=saved.verify_predictions(m.OUT/('selected_'+role+'.npz'),best['model'],role,m.base.manifest());saved.agree(met,fit['metrics'][role]);saved.agree(met,best['metrics'][role])
 for file,sha in fit['hashes'].items():assert m.s.sha(m.OUT/file)==sha
 (m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],selected_step=best['selected_step'],correct=max(correct),full_128_updates_independently_replayed=True,optical_forward=0,TEST_informed_development=True,independent_generalization=False,controls=m.prior.parent.old.controls()),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
