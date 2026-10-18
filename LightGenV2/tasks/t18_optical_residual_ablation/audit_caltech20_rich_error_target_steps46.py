"""Independent saved TRAIN sampler, Adam/EMA boundary, head arithmetic and TEST rank."""
import argparse,math,os
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_error_target_steps46 as m
from check_caltech20_rich_error_target_steps46 import independent_plan
import audit_caltech20_rich_pair50_alpha as saved

def run(d):
 assert d in (4,6) and os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(4);out=m.OUT/f'L{d}';assert not (out/'science').exists();m.claim(out/'audit_claim.json','independent_short_step_saved_science');ex=m.s.read(m.OUT/f'L{d}.producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();meta=m.s.read(out/'metadata.json');fit=m.s.read(out/'fit_result.json');trace=m.s.read(out/'method_trace.json');history=m.s.read(out/'history.json');plan=m.s.read(out/'sampling_plan.json');q,pt,pred=m.parent.incumbent(d);assert meta['parent']==q and meta['sources']==m.sources() and meta['config']==m.CFG and meta['controls']==m.parent.old.controls();m.engine.verify_sources(meta['sources'],meta['git_commit']);assert meta['init_digest']==m.s.digest(pt['model'])
 focus,hard,replay,batches=independent_plan(pred);assert (plan['focus_classes'],plan['hard'],plan['replay'],plan['batches'])==(focus,hard,replay,batches);assert len(trace)==fit['actual_steps']==64 and fit['TRAIN_presentations']==1024 and fit['unique_selected_TRAIN']==176 and fit['full_TRAIN_training_epochs']==0;assert len(history)==fit['TEST_evaluations']==4
 for fn,sha in fit['hashes'].items():assert m.s.sha(out/fn)==sha
 gate=m.s.read(out/'numeric_gate.json');assert gate['passed'] and gate['optimizer_steps']==gate['TEST_forwards']==0 and gate['parent_probability_maxabs']<=3e-6
 for row in gate['checks']:
  assert row['gradient_max_relative']<=.02 and row['gradient_min_cosine']>=.999 and row['probability_maxabs']<=3e-6 and len(row['gradient_norms'])=={4:21,6:30}[d] and all(np.isfinite(v) and v>0 for v in row['gradient_norms'].values())
 for step,row in enumerate(trace,1):
  assert row['step']==row['adam_updates']==step and row['indices']==batches[step-1];rg=np.random.default_rng(np.random.SeedSequence([m.CFG['seed'],step]));theta=np.zeros((16,2,3),np.float32);theta[:,0,0]=theta[:,1,1]=1;theta[:,:,2]=rg.uniform(-.04,.04,(16,2));assert m.s.digest(theta)==row['theta_sha256'];assert np.isfinite([row['loss'],row['clip_norm']]).all()
 best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);last=torch.load(out/'last.pt',map_location='cpu',weights_only=False);boundary=torch.load(out/'optimizer_boundary.pt',map_location='cpu',weights_only=False);manifest=m.base.manifest();basecorrect=int((pred['test']['labels']==pred['test']['probabilities'].argmax(1)).sum());choices=[(0,basecorrect)]
 for step,row in zip((16,32,48,64),history):
  assert row['step']==step;z=m.parent.load_predictions(out/f'test_step{step:03d}.npz');assert z['ids'].tolist()==manifest['split_ids']['test'] and z['labels'].tolist()==pred['test']['labels'].tolist();actual=saved.metrics(z['labels'],z['probabilities']);saved.agree(actual,row['test']);n=int((z['labels']==z['probabilities'].argmax(1)).sum());assert n==row['test_correct'];choices.append((step,n))
 chosen=max(choices,key=lambda t:t[1])[0];assert chosen==fit['selected_step']==best['selected_step'];assert m.s.digest(best['model'])==(m.s.digest(pt['model']) if chosen==0 else next(r['ema_digest'] for r in history if r['step']==chosen));assert m.s.digest(last['ema'])==history[-1]['ema_digest'];assert last['steps']==64
 for role in ('train','val','test'):actual,_=saved.verify_predictions(out/('selected_'+role+'.npz'),best['model'],role,manifest);saved.agree(actual,best['metrics'][role]);saved.agree(actual,fit['metrics'][role])
 norm=torch.stack([a.double().square().sum() for a in boundary['gradients'].values()]).sum().sqrt().item();assert abs(norm-boundary['clip_norm'])<max(1e-6,1e-5*norm);clip=min(1.,1/(norm+1e-6));maximum=0
 from caltech20_rich_projected_alpha_contained import ALPHA_MIN
 for group,names in zip(boundary['optimizer_before']['param_groups'],boundary['groups']):
  b1,b2=group['betas']
  for pid,n in zip(group['params'],names):
   old=boundary['optimizer_before']['state'][pid];new=last['optimizer']['state'][pid];assert float(old['step'])==63 and float(new['step'])==64;g=boundary['gradients'][n].double()*clip;exp=old['exp_avg'].double()*b1+(1-b1)*g;var=old['exp_avg_sq'].double()*b2+(1-b2)*g.square();raw=boundary['raw_before'][n].double()-group['lr']*(exp/(1-b1**64))/(var.sqrt()/math.sqrt(1-b2**64)+group['eps'])
   if n.startswith('fusion_logits.'):raw=raw.clamp(ALPHA_MIN,.95)
   maximum=max(maximum,float((raw-last['model'][n].double()).abs().max()));assert torch.allclose(raw,last['model'][n].double(),atol=2e-6,rtol=2e-5);assert torch.allclose(exp,new['exp_avg'].double(),atol=1e-7,rtol=2e-5) and torch.allclose(var,new['exp_avg_sq'].double(),atol=1e-7,rtol=2e-5);expected=boundary['ema_before'][n].double()*.9+last['model'][n].double()*.1;assert torch.allclose(expected,last['ema'][n].double(),atol=2e-7,rtol=1e-6)
 assert all(.4001<float(v)<=.9500001 for n,v in best['model'].items() if n.startswith('fusion_logits.'));assert set(best['model'])==set(pt['model'])
 (out/'science').mkdir();m.s.save(out/'science/audit.json',dict(passed=True,sources=meta['sources'],training_commit=meta['git_commit'],selected_step=chosen,TEST_correct=next(n for st,n in choices if st==chosen),independent_TRAIN_sampler=True,Adam_maxabs=maximum,new_forward=0,TEST_informed_development=True,independent_generalization=False,no_L2_model=True,controls=m.parent.old.controls()),exclusive=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--depth',type=int,required=True);z=a.parse_args()
 try:run(z.depth)
 except BaseException as e:
  p=m.OUT/f'L{z.depth}/audit_failure.json'
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
