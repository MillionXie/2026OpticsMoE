import math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_tail_separation6 as m
from check_caltech20_rich_tail_separation6 import independent_plan
import audit_caltech20_rich_pair50_alpha as saved
def run():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not (m.OUT/'science').exists();m.claim(m.OUT/'audit_claim.json','independent_last_global_scope_steps_Adam_EMA');meta=m.s.read(m.OUT/'metadata.json');ex=m.s.read(m.OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert meta['config']==m.config() and meta['sources']==m.sources();m.engine.verify_sources(meta['sources'],meta['git_commit']);q,pt,pred=m.parent_data();assert meta['parent']==q and meta['init_digest']==m.s.digest(pt['model']);fit=m.s.read(m.OUT/'fit_result.json');pl=m.s.read(m.OUT/'sampling_plan.json');bad,items,replay,batches,negative,centers=independent_plan(pred);assert (bad,items,replay,batches)==(pl['TEST_error_indices'],pl['TRAIN_neighbors'],pl['replay'],pl['batches']) and negative.tolist()==pl['negative_indices'];assert np.allclose(centers,np.array(pl['centers']),atol=1e-7);scale=m.s.read(m.OUT/'triplet_scale.json');assert scale==m.triplet_scale(pl,pred,pt);trace=m.s.read(m.OUT/'trace.json');history=m.s.read(m.OUT/'history.json');best=torch.load(m.OUT/'best.pt',map_location='cpu',weights_only=False);last=torch.load(m.OUT/'last.pt',map_location='cpu',weights_only=False);boundary=torch.load(m.OUT/'optimizer_boundary.pt',map_location='cpu',weights_only=False);assert len(trace)==fit['actual_tail_steps']==64 and len(history)==fit['TEST_model_evaluations']==8
 for file,sha in fit['hashes'].items():assert m.s.sha(m.OUT/file)==sha
 gate=m.s.read(m.OUT/'numeric_gate.json');assert gate['passed'] and gate['trainable']==list(m.TRAINABLE) and gate['parent_probability_maxabs']<=3e-6
 for row in gate['checks']:assert set(row['gradient_norms'])==set(m.TRAINABLE) and row['gradient_max_relative']<=.02 and row['gradient_min_cosine']>=.999 and row['probability_maxabs']<=3e-6 and all(np.isfinite(v) and v>0 for v in row['gradient_norms'].values())
 for step,row in enumerate(trace,1):
  assert row['step']==step and row['indices']==batches[step-1];rg=np.random.default_rng(np.random.SeedSequence([m.config()['seed'],step]));theta=np.zeros((16,2,3),np.float32);theta[:,0,0]=theta[:,1,1]=1;theta[:,:,2]=rg.uniform(-.04,.04,(16,2));assert m.s.digest(theta)==row['theta_SHA256'];parts=row['loss_parts'];assert abs(row['loss']-(parts[0]+parts[1]+scale['beta']*parts[2]))<2e-6
 choices=[(0,int((pred['test']['probabilities'].argmax(1)==pred['test']['labels']).sum()))]
 for step,row in zip(range(8,65,8),history):
  assert row['step']==step;z=m.parent.load_predictions(m.OUT/f'test_step{step:03d}.npz');assert z['ids'].tolist()==m.base.manifest()['split_ids']['test'] and np.array_equal(z['labels'],pred['test']['labels']);actual=saved.metrics(z['labels'],z['probabilities']);saved.agree(actual,row['test']);n=int((z['probabilities'].argmax(1)==z['labels']).sum());assert n==row['correct'];choices.append((step,n))
 selected=max(choices,key=lambda a:a[1])[0];assert selected==fit['selected_step']==best['selected_step'];assert m.s.digest(best['model'])==(m.s.digest(pt['model']) if selected==0 else next(r['ema_digest'] for r in history if r['step']==selected));assert m.s.digest(last['ema'])==history[-1]['ema_digest'];assert m.frozen_digest(best['model'])==m.frozen_digest(last['model'])==m.frozen_digest(last['ema'])==m.frozen_digest(pt['model'])==meta['frozen_prefix_digest']
 norm=torch.stack([g.double().square().sum() for g in boundary['gradients'].values()]).sum().sqrt().item();assert abs(norm-boundary['clip_norm'])<max(1e-6,1e-5*norm);clip=min(1.,1/(norm+1e-6));maximum=0
 from caltech20_rich_projected_alpha_contained import ALPHA_MIN
 for group,names in zip(boundary['optimizer_before']['param_groups'],boundary['groups']):
  b1,b2=group['betas']
  for pid,n in zip(group['params'],names):
   old=boundary['optimizer_before']['state'][pid];new=last['optimizer']['state'][pid];assert float(old['step'])==63 and float(new['step'])==64;g=boundary['gradients'][n].double()*clip;exp=old['exp_avg'].double()*b1+(1-b1)*g;var=old['exp_avg_sq'].double()*b2+(1-b2)*g.square();raw=boundary['raw_before'][n].double()-group['lr']*(exp/(1-b1**64))/(var.sqrt()/math.sqrt(1-b2**64)+group['eps'])
   if n=='fusion_logits.5':raw=raw.clamp(ALPHA_MIN,.95)
   maximum=max(maximum,float((raw-last['model'][n].double()).abs().max()));assert torch.allclose(raw,last['model'][n].double(),atol=2e-6,rtol=2e-5);assert torch.allclose(exp,new['exp_avg'].double(),atol=1e-7,rtol=2e-5) and torch.allclose(var,new['exp_avg_sq'].double(),atol=1e-7,rtol=2e-5);expected=.9*boundary['ema_before'][n].double()+.1*last['model'][n].double();assert torch.allclose(expected,last['ema'][n].double(),atol=2e-7,rtol=1e-6)
 for role in ('train','val','test'):met,_=saved.verify_predictions(m.OUT/('selected_'+role+'.npz'),best['model'],role,m.base.manifest());saved.agree(met,fit['metrics'][role]);saved.agree(met,best['metrics'][role])
 assert set(best['model'])==set(pt['model']) and all(.4001<float(v)<=.9500001 for n,v in best['model'].items() if n.startswith('fusion_logits.'))
 (m.OUT/'science').mkdir();m.s.save(m.OUT/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],selected_step=selected,TEST_correct=max(n for st,n in choices),frozen_prefix5_bytes_unchanged=True,trainable=list(m.TRAINABLE),Adam_maxabs=maximum,new_model_forward=0,TEST_informed_development=True,independent_generalization=False,controls=m.parent.old.controls()),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.OUT/'audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
