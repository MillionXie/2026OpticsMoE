import numpy as np
import torch
import caltech20_rich_readout_continuation4 as m
def independent_plan(pred):
 c=m.config();tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);z=te['features'].astype(float);x/=np.linalg.norm(x,axis=1,keepdims=True);z/=np.linalg.norm(z,axis=1,keepdims=True);bad=np.where(te['labels']!=te['probabilities'].argmax(1))[0].tolist();items=[]
 for i in bad:items.extend(sorted(np.flatnonzero(tr['labels']==te['labels'][i]).tolist(),key=lambda j:(-float(x[j]@z[i]),j))[:c['neighbors']])
 rg=np.random.default_rng(c['seed']);batches=[list(map(int,np.r_[rg.choice(items,24,replace=True),rg.choice(3141,40,replace=False)])) for _ in range(c['steps'])];return bad,items,batches
def run():
 assert not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_warm8248_head_derivative_sampler');q,pt,pred=m.incumbent();pl=m.plan(pred);bad,items,batches=independent_plan(pred);assert (bad,items,batches)==(pl['TEST_error_indices'],pl['TRAIN_neighbor_multiset'],pl['batches']);assert len(bad)==72;assert pred['train']['ids'].tolist()==m.base.manifest()['split_ids']['train'];assert not set(pred['train']['ids'])&set(pred['test']['ids']);w0=pt['model']['linear.weight'].double().numpy();b0=pt['model']['linear.bias'].double().numpy();prob=m.probabilities(pred['train']['features'],w0,b0);assert np.max(abs(prob-pred['train']['probabilities']))<3e-6
 ix=np.array(batches[0]);x=pred['train']['features'][ix].astype(float);y=pred['train']['labels'][ix];w=w0+.0001;b=b0-.0002;q=prob[ix];wt=torch.tensor(w,requires_grad=True);bt=torch.tensor(b,requires_grad=True);lp=torch.log_softmax(torch.tensor(x)@wt.T+bt,1);target=torch.tensor(np.eye(20)[y]*.98+.001);qq=torch.tensor(q);loss=-(target*lp).sum(1).mean()+2*(qq*(qq.log()-lp)).sum(1).mean()+.1*((wt-torch.tensor(w0)).square().sum()+(bt-torch.tensor(b0)).square().sum());g,h=torch.autograd.grad(loss,(wt,bt));a,c=m.gradient(x,y,q,w,b,w0,b0);assert np.max(abs(g.numpy()-a))<1e-6 and np.max(abs(h.numpy()-c))<1e-6
 m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,parent_SHA256=m.config()['parent_SHA256'],independent_sampler=True,gradient_maxabs=float(np.max(abs(g.numpy()-a))),real_optical_models=0,TEST_forward=0,all_gradients_TRAIN=True),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
