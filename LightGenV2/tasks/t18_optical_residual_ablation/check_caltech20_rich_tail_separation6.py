import numpy as np
import torch
import caltech20_rich_tail_separation6 as m
def independent_plan(pred):
 c=m.config();tr,te=pred['train'],pred['test'];x=tr['features'].astype(float);x/=np.linalg.norm(x,axis=1,keepdims=True);z=te['features'].astype(float);z/=np.linalg.norm(z,axis=1,keepdims=True);bad=np.where(te['labels']!=te['probabilities'].argmax(1))[0].tolist();items=[]
 for i in bad:items.extend(sorted(np.flatnonzero(tr['labels']==te['labels'][i]).tolist(),key=lambda j:(-float(x[j]@z[i]),j))[:3])
 rg=np.random.default_rng(c['seed']);replay=[]
 for label in range(20):replay.extend(rg.choice([int(i) for i in np.flatnonzero(tr['labels']==label) if i not in items],4,replace=False).tolist())
 batches=[list(map(int,np.r_[rg.choice(items,8,replace=True),rg.choice(replay,8,replace=False)])) for _ in range(c['steps'])];required=sorted(set(items+replay+[int(np.flatnonzero(tr['labels']==j)[0]) for j in range(20)]));negative=np.full(3141,-1,int)
 for i in required:
  candidates=np.flatnonzero(tr['labels']!=tr['labels'][i]);negative[i]=int(candidates[np.argmax(x[candidates]@x[i])])
 centers=np.stack([np.sum(x[tr['labels']==j],axis=0)/sum(tr['labels']==j) for j in range(20)]);centers/=np.linalg.norm(centers,axis=1,keepdims=True);return bad,items,replay,batches,negative,centers
def run():
 assert not m.CPU.exists();m.claim(m.CPU.with_name(m.CPU.name+'.claim.json'),'independent_tail_scope_sampler_triplet_derivative');q,pt,pred=m.parent_data();pl=m.plan(pred);bad,items,replay,batches,negative,centers=independent_plan(pred);assert (bad,items,replay,batches)==(pl['TEST_error_indices'],pl['TRAIN_neighbors'],pl['replay'],pl['batches']);assert negative.tolist()==pl['negative_indices'] and np.allclose(centers,np.array(pl['centers']),atol=1e-7);assert len(bad)==78 and len(set(items))==217;assert not set(pred['train']['ids'])&set(pred['test']['ids'])
 model,_,_,_=m.model_for();assert set(n for n,p in model.named_parameters() if p.requires_grad)==set(m.TRAINABLE);assert m.frozen_digest(model.state_dict())==m.frozen_digest(pt['model']);assert model.stage_slices[5]==(14,15) and model.stages==['expert','global']*3
 r=np.random.default_rng(16);f=torch.tensor(r.normal(size=(5,256)),requires_grad=True);pos=torch.nn.functional.normalize(torch.tensor(r.normal(size=(5,256))),dim=1);neg=torch.nn.functional.normalize(torch.tensor(r.normal(size=(5,256))),dim=1);u=torch.nn.functional.normalize(f,dim=1);v=neg-pos;loss=torch.relu(.1+(u*v).sum(1)).mean();grad=torch.autograd.grad(loss,f)[0].numpy();ff=f.detach().numpy();uu=ff/np.linalg.norm(ff,axis=1,keepdims=True);vv=v.numpy();active=(.1+(uu*vv).sum(1)>0);expected=(vv-uu*(uu*vv).sum(1,keepdims=True))/np.linalg.norm(ff,axis=1,keepdims=True)*active[:,None]/5;assert np.max(abs(expected-grad))<1e-12
 scale=m.triplet_scale(pl,pred,pt);ix=np.array(sorted(set(items)));ff=pred['train']['features'][ix].astype(float);u=ff/np.linalg.norm(ff,axis=1,keepdims=True);neg=pred['train']['features'][negative[ix]].astype(float);neg/=np.linalg.norm(neg,axis=1,keepdims=True);v=neg-centers[pred['train']['labels'][ix]];active=(.1+(u*v).sum(1)>0);g=(v-u*(u*v).sum(1,keepdims=True))/np.linalg.norm(ff,axis=1,keepdims=True)*active[:,None];ce=(pred['train']['probabilities'][ix]-(np.eye(20)[pred['train']['labels'][ix]]*.98+.001))@pt['model']['linear.weight'].double().numpy();expected_beta=min(2.,.2*np.linalg.norm(ce)/max(np.linalg.norm(g),1e-12));assert abs(scale['beta']-expected_beta)<1e-5 and scale['triplet_active']>.1
 m.CPU.mkdir();m.s.save(m.CPU/'checks.json',dict(m.identity(),passed=True,scope=list(m.TRAINABLE),actual_parent_models=1,model_forward=0,optimizer=0,TEST_forward=0,independent_triplet_derivative_maxabs=float(np.max(abs(expected-grad))),scale=scale,all_gradient_indices_TRAIN=True,real_GPU_gate_required=True),exclusive=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  p=m.CPU.with_name(m.CPU.name+'.failure.json')
  if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
  raise
