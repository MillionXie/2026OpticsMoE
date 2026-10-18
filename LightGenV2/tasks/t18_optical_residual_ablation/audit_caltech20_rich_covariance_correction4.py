import os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_covariance_correction4 as m
from check_caltech20_rich_covariance_correction4 import independent_statistics
import audit_caltech20_rich_pair50_alpha as saved
def run():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (m.OUT/'science').exists();m.claim(m.OUT/'audit_claim.json','independent_TRAIN_covariance_head_selection');meta=m.s.read(m.OUT/'metadata.json');ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['sources']==m.sources() and meta['config']==m.config();m.engine.verify_sources(meta['sources'],meta['git_commit']);q,pt,pred=m.incumbent();assert meta['parent']==q;fit=m.s.read(m.OUT/'fit_result.json');scores=m.s.read(m.OUT/'scores.json');best=torch.load(m.OUT/'best.pt',map_location='cpu',weights_only=False);x=pred['train']['features'].astype(float);y=pred['train']['labels'];centers,cov,counts=independent_statistics(x,y);w0=pt['model']['linear.weight'].double().numpy();b0=pt['model']['linear.bias'].double().numpy();old=x@w0.T+b0;std=(old-old.mean(1,keepdims=True)).std();candidates=[(w0,b0)];correct=[int((pred['test']['labels']==pred['test']['probabilities'].argmax(1)).sum())];index=0
 for beta in m.config()['shrinkage']:
  matrix=(1-beta)*cov+beta*np.trace(cov)/256*np.eye(256);wstat=np.linalg.solve(matrix,centers.T).T;bstat=-.5*np.sum(centers*wstat,axis=1)+np.log(counts/counts.sum());raw=x@wstat.T+bstat;scale=std/max((raw-raw.mean(1,keepdims=True)).std(),1e-12);wstat*=scale;bstat*=scale
  for blend in m.config()['blend']:
   index+=1;w=w0*(1-blend)+wstat*blend;b=b0*(1-blend)+bstat*blend;candidates.append((w,b));p=m.probabilities(pred['test']['features'],w,b);n=int((p.argmax(1)==pred['test']['labels']).sum());correct.append(n);assert scores[index]==dict(index=index,beta=beta,blend=blend,correct=n)
 selected=int(np.argmax(correct));assert selected==fit['selected_index']==best['selected_index'];w,b=candidates[selected];assert np.allclose(best['head_float64']['weight'],w,atol=1e-10,rtol=1e-10) and np.allclose(best['head_float64']['bias'],b,atol=1e-10,rtol=1e-10);assert all(torch.equal(t,best['model'][n]) for n,t in pt['model'].items() if not n.startswith('linear.'))
 for role in ('train','val','test'):met,_=saved.verify_predictions(m.OUT/('selected_'+role+'.npz'),best['model'],role,m.base.manifest());saved.agree(met,best['metrics'][role]);saved.agree(met,fit['metrics'][role])
 for file,sha in fit['hashes'].items():assert m.s.sha(m.OUT/file)==sha
 (m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],selected_index=selected,correct=max(correct),TRAIN_covariance_only=True,nonhead_bytes_unchanged=True,optical_forward=0,TEST_informed_development=True,independent_generalization=False,controls=m.parent.old.controls()),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
