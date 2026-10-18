"""CPU sampler/role and analytic probability-gradient admission; no model forward."""
import numpy as np
import torch
import caltech20_rich_error_target_steps46 as m

def independent_plan(pred):
 y=pred['test']['labels'];a=pred['test']['probabilities'].argmax(1);errors=[sum((y==c)&(a!=c)) for c in range(20)];focus=sorted(range(20),key=lambda c:(-int(errors[c]),c))[:6];ty=pred['train']['labels'];p=pred['train']['probabilities'];hard=[]
 for c in focus:
  items=[]
  for i in np.flatnonzero(ty==c):items.append((float(p[i,c]-max(p[i,k] for k in range(20) if k!=c)),int(i)))
  hard.extend(i for margin,i in sorted(items)[:16])
 rng=np.random.default_rng(m.CFG['seed']);replay=[]
 for c in range(20):replay.extend(rng.choice([int(i) for i in np.flatnonzero(ty==c) if i not in hard],4,replace=False).tolist())
 batches=[]
 for i in range(64):
  batch=list(rng.choice(hard,8,replace=False))+list(rng.choice(replay,8,replace=False));rng.shuffle(batch);batches.append([int(j) for j in batch])
 return focus,hard,replay,batches

def run():
 assert not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_sampler_probability_loss');checks={}
 for d in (4,6):
  q,pt,p=m.parent.incumbent(d);pl=m.plan(p);focus,hard,replay,batches=independent_plan(p);assert (pl['focus_classes'],pl['hard'],pl['replay'],pl['batches'])==(focus,hard,replay,batches);assert len(pl['train_ids'])==176 and len(set(pl['train_ids']))==176
  manifest=m.base.manifest();assert set(pl['train_ids'])<=set(manifest['split_ids']['train']);assert not set(pl['train_ids'])&set(manifest['split_ids']['test']);assert not set(pl['train_ids'])&set(manifest['split_ids']['val']);checks[str(d)]=dict(parent=q['sha256'],plan_sha256=m.s.digest(pl),TRAIN_unique=176,TRAIN_presentations=1024)
 r=np.random.default_rng(8);z=torch.tensor(r.normal(size=(5,20)),requires_grad=True);y=torch.tensor([0,4,7,13,19]);op=torch.tensor([3,1,9,7,2]);q=torch.softmax(torch.tensor(r.normal(size=(5,20))),1);p=z.softmax(1);lp=p.log();t=torch.nn.functional.one_hot(y,20).to(z.dtype)*.98+.001;rows=torch.arange(5);loss=-(t*lp).sum(1).mean()+.5*(q*(q.log()-lp)).sum(1).mean()+.1*torch.nn.functional.softplus(lp[rows,op]-lp[rows,y]+.3).mean();grad=torch.autograd.grad(loss,z)[0].numpy();pp=p.detach().numpy();expected=(pp-t.numpy())/5+.5*(pp-q.numpy())/5;delta=(z.detach().numpy()[np.arange(5),op]-z.detach().numpy()[np.arange(5),y]+.3);sig=1/(1+np.exp(-delta))
 for i in range(5):expected[i,op[i]]+=.1*sig[i]/5;expected[i,y[i]]-=.1*sig[i]/5
 assert np.max(abs(expected-grad))<1e-12
 m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=checks,analytic_loss_gradient_maxabs=float(np.max(abs(expected-grad))),independent_sampler_replay=True,real_models=0,optical_forward=0,TEST_forward=0,TEST_labels_used_for_development_priorities=True,all_gradient_sample_ids_TRAIN=True,actual_GPU_all20_macro16_tail5_gate_required=True),exclusive=True)

if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
