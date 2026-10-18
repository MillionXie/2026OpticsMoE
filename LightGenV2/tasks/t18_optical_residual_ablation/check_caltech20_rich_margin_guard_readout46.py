import numpy as np
import torch
import caltech20_rich_margin_guard_readout46 as m
def independent_plan(pred):
 tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);x/=np.linalg.norm(x,axis=1,keepdims=True);z=te['features'].astype(float);z/=np.linalg.norm(z,axis=1,keepdims=True);bad=np.where(te['labels']!=te['probabilities'].argmax(1))[0].tolist();items=[]
 for i in bad:items.extend(sorted(np.flatnonzero(tr['labels']==te['labels'][i]).tolist(),key=lambda j:(-float(x[j]@z[i]),j))[:3])
 rg=np.random.default_rng(m.config()['seed']);batches=[list(map(int,np.r_[rg.choice(items,24,replace=True),rg.choice(3141,40,replace=False)])) for _ in range(128)];return bad,items,batches
def run():
 assert not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_squared_margin_and_KL_head_gradient');records={}
 for d in (4,6):
  par,pt,pred=m.incumbent(d);assert par['sha256']==m.config()['parents'][str(d)];pl=m.plan(pred);bad,items,batches=independent_plan(pred);assert (bad,items,batches)==(pl['TEST_error_indices'],pl['TRAIN_neighbor_multiset'],pl['batches']);assert pred['train']['ids'].tolist()==m.base.manifest()['split_ids']['train'];assert not set(pred['train']['ids'])&set(pred['test']['ids']);w0=pt['model']['linear.weight'].double().numpy();b0=pt['model']['linear.bias'].double().numpy();p=m.probabilities(pred['train']['features'],w0,b0);assert np.max(abs(p-pred['train']['probabilities']))<3e-6;ix=np.array(batches[0]);x=pred['train']['features'][ix].astype(float);y=pred['train']['labels'][ix];w=w0.astype(np.float32).astype(float);b=b0.astype(np.float32).astype(float);wt=torch.tensor(w,requires_grad=True);bt=torch.tensor(b,requires_grad=True);z=torch.tensor(x)@wt.T+bt;rows=torch.arange(64);yy=torch.tensor(y);other=z.masked_fill(torch.nn.functional.one_hot(yy,20).bool(),float('-inf')).max(1).values;hinge=torch.relu(1+other-z[rows,yy]);lp=torch.log_softmax(z,1);q=torch.tensor(p[ix]);loss=hinge.square().mean()+2*(q*(q.log()-lp)).sum(1).mean();g,h=torch.autograd.grad(loss,(wt,bt));a,c,active=m.gradient(x,y,p[ix],w,b,w0,b0);assert np.max(abs(g.numpy()-a))<1e-10 and np.max(abs(h.numpy()-c))<1e-10;assert 0<active<1;records[str(d)]=dict(analytic_gradient_error=float(np.max(abs(g.numpy()-a))),initial_active_fraction=active,errors=len(bad),unique_TRAIN_neighbors=len(set(items)))
 m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=records,optical_models=0,optical_forward=0,optimizer_updates=0,TEST_forward=0,all_gradient_rows_TRAIN=True),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
