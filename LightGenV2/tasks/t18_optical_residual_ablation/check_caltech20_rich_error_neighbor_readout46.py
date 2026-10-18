"""Independent cached-head analytic derivative and actual-feature numerical admission."""
import numpy as np
import torch
import caltech20_rich_error_neighbor_readout46 as m
def run():
 assert not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'cached_head_probability_derivative_and_roles');records={}
 for d in (4,6):
  q,pt,p=m.parent.incumbent(d);pl=m.neighbors(p);tr,te=p['train'],p['test'];assert all(0<=i<3141 for row in pl['TRAIN_batches'] for i in row);assert tr['ids'].tolist()==m.base.manifest()['split_ids']['train'];assert not set(tr['ids'])&set(te['ids']);w=pt['model']['linear.weight'].double().numpy();b=pt['model']['linear.bias'].double().numpy();prob=m.probabilities(tr['features'],w,b);assert np.max(abs(prob-tr['probabilities']))<3e-6
  ix=np.array(pl['TRAIN_batches'][0]);x=tr['features'][ix].astype(float);y=tr['labels'][ix];q=prob[ix];wt=torch.tensor(w,requires_grad=True);bt=torch.tensor(b,requires_grad=True);lp=torch.log_softmax(torch.tensor(x)@wt.T+bt,1);target=torch.tensor(np.eye(20)[y]*.98+.001);qq=torch.tensor(q);loss=-(target*lp).sum(1).mean()+(qq*(qq.log()-lp)).sum(1).mean();gw,gb=torch.autograd.grad(loss,(wt,bt));a,c=m.gradient(x,y,q,w,b,w,b);assert np.max(abs(gw.numpy()-a))<1e-12 and np.max(abs(gb.numpy()-c))<1e-12;records[str(d)]=dict(parent=q.shape,TRAIN_neighbors=pl['TRAIN_unique_neighbors'],gradient_error=float(np.max(abs(gw.numpy()-a))))
 m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,checks=records,real_optical_models=0,optical_forward=0,optimizer_steps=0,all_gradient_rows_TRAIN=True,TEST_features_used_only_for_neighbors_and_selection=True),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
