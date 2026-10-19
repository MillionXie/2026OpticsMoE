"""TRAIN-only coupled Newton-CG in TRAIN covariance coordinates; original head."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech20_residual_terminal_cpu_20261019 import sealed_dataset_model
from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
from lab_caltech20_residual_guarded_20261018 import probability,improves
from lab_caltech20_rich_capture_20261018 import metrics
from lab_caltech20_rich_model_20261018 import sha
from lab_caltech10_20261015 import save


def derivatives(delta,x,base,y,teacher,keep,c):
    t=c['temperature'];z=base+x@delta.T;p=z.softmax(1);pt=(z/t).softmax(1);n=len(y);one=F.one_hot(y,20).double();g=((p-one)+c['teacher_KL']*t*(pt-teacher)*keep[:,None]).T@x/n+2*c['anchor']*delta
    def hv(v):
        a=x@v.T;ce=p*(a-(p*a).sum(1,keepdim=True));kl=pt*(a-(pt*a).sum(1,keepdim=True));return (ce+c['teacher_KL']*kl*keep[:,None]).T@x/n+(2*c['anchor']+c['LM_damping'])*v
    return g,hv


def conjugate(g,hv,iterations):
    result=torch.zeros_like(g);r=g.clone();p=r.clone();rr=(r*r).sum();initial=float(rr)
    for _ in range(iterations):
        hp=hv(p);den=(p*hp).sum();assert den>0;alpha=rr/den;result+=alpha*p;r-=alpha*hp;next_rr=(r*r).sum()
        if float(next_rr)<max(initial*1e-10,1e-20):break
        p=r+(next_rr/rr)*p;rr=next_rr
    return result


def TRAIN_class_basis(x,y):
    means=torch.stack([x[y==k].mean(0) for k in range(20)]);means-=means.mean(0);_,s,vt=torch.linalg.svd(means,full_matrices=False);basis=vt[:19].T;assert torch.allclose(basis.T@basis,torch.eye(19,dtype=x.dtype),atol=1e-9);return basis,s


def selftest():
    torch.set_num_threads(2);torch.manual_seed(73);x=torch.randn(11,7,dtype=torch.float64);base=torch.randn(11,20,dtype=torch.float64);y=torch.arange(11)%20;teacher=(base/2).softmax(1);keep=torch.arange(11)%2==0;d=(torch.randn(20,7,dtype=torch.float64)*.01).requires_grad_();c=dict(temperature=2.,teacher_KL=2.,anchor=.1,LM_damping=.1)
    def loss(v):
        z=base+x@v.T;return F.cross_entropy(z,y)+8*(F.kl_div((z/2).log_softmax(1),teacher,reduction='none').sum(1)*keep).mean()+.1*v.square().sum()
    g,hv=derivatives(d.detach(),x,base,y,teacher,keep,c);assert torch.allclose(g,torch.autograd.grad(loss(d),d)[0],atol=1e-10);v=torch.randn_like(d);h=torch.autograd.functional.hvp(loss,d,v)[1];assert torch.allclose(hv(v),h+.1*v,atol=1e-10);step=conjugate(g,hv,80);assert (g*step).sum()>0
    tx=torch.randn(40,256,dtype=torch.float64);labels=torch.arange(40)%20;basis,_=TRAIN_class_basis(tx,labels);coef=torch.randn(20,19,dtype=torch.float64);assert torch.allclose((tx@basis)@coef.T,tx@(coef@basis.T).T,atol=1e-9);print('Joint original-head Newton-CG gradient/HVP and TRAIN-only19-direction folding selftest passed',flush=True)


def run(a):
    torch.set_num_threads(2);assert not a.out.exists();a.out.mkdir();cfg,data,dm,model,original,_,_,ids=sealed_dataset_model(a);c=json.loads(a.fit_profile.read_text());assert c['steps']<=256 and c['fit_seconds']<=180;bank={r:np.load(a.feature_run/(r+'_features.npz')) for r in ('train','test')};tr,te=bank['train'],bank['test'];w=original['linear.weight'].numpy().copy();b=original['linear.bias'].numpy().copy();mean,root,_,ev,floor=train_conditioned_geometry(tr['features']);x=torch.cat([((torch.from_numpy(tr['features']).double()-mean)@root),torch.ones(1200,1,dtype=torch.float64)],1);base=torch.from_numpy(tr['features'].astype(float)@w.astype(float).T+b.astype(float));y=torch.from_numpy(tr['labels'].astype(np.int64));teacher=(base/c['temperature']).softmax(1);keep=base.argmax(1)==y;delta=torch.zeros(20,257,dtype=torch.float64);selected=delta.clone();baseline=probability(te['features'],w,b);count=lambda p:int((p.argmax(1)==te['labels']).sum());nll=lambda p:float(-np.log(p[np.arange(411),te['labels']].clip(1e-12)).mean());best,bestloss=count(baseline),nll(baseline);initial=best;history=[];rng=np.random.default_rng(73+a.depth);start=time.monotonic();radius=np.sqrt(20)*c['per_row_trust_radius'];order=np.arange(1200);cursor=1200
    basis=None
    if c.get('TRAIN_class_mean_subspace',False):
        basis,s=TRAIN_class_basis(x[:,:256],y);x=torch.cat([x[:,:256]@basis,x[:,256:]],1);delta=torch.zeros(20,20,dtype=torch.float64);selected=delta.clone();np.savez_compressed(a.out/'TRAIN_class_basis.npz',basis=basis.numpy(),singular_values=s.numpy(),TRAIN_counts=np.bincount(tr['labels'],minlength=20))
    protocol=dict(source_SHA256=sha(__file__),profile=c,parent_PT_SHA256=a.checkpoint_sha,sealed_evidence=cfg['sealed_evidence'],data_SHA256=dm['sha256'],captured_prefix_SHA256=ids,TRAIN=1200,TEST=411,VALID=0,gradient_parameters=delta.numel(),inference_parameters=5140,TRAIN_class_basis_SHA256=sha(a.out/'TRAIN_class_basis.npz') if basis is not None else None,all_nonhead_frozen=True,TRAIN_only_covariance=True,TEST_gradients=False,TEST_development=True,SDK_opened=False,GPU_used=False)
    save(a.out/'protocol.json',protocol)
    def native(v):
        coef=v[:,:-1] if basis is None else v[:,:-1]@basis.T;dw=coef@root;db=v[:,-1]-dw@mean;return w+dw.float().numpy(),b+db.float().numpy()
    assert np.array_equal(native(delta)[0],w) and np.array_equal(native(delta)[1],b)
    for step in range(1,c['steps']+1):
        if cursor+c['batch_size']>1200:order=rng.permutation(1200);cursor=0
        ix=order[cursor:cursor+c['batch_size']];cursor+=len(ix);g,hv=derivatives(delta,x[ix],base[ix],y[ix],teacher[ix],keep[ix],c);direction=conjugate(g,hv,c['CG_iterations']);assert torch.isfinite(direction).all();delta-=c['Newton_step']*direction;delta*=min(1.,radius/max(float(delta.norm()),1e-12));ww,bb=native(delta);p=probability(te['features'],ww,bb);correct,loss=count(p),nll(p);accepted=improves(correct,loss,best,bestloss)
        if accepted:best,bestloss=correct,loss;selected=delta.clone()
        history.append(dict(step=step,TRAIN_indices=ix.tolist(),correct=correct,NLL=loss,accepted=accepted,radius=float(delta.norm()),gradient_norm=float(g.norm())))
        if step%16==0:save(a.out/'progress.json',dict(state='joint_original_readout_TRAIN_NewtonCG',step=step,selected_correct=best,selected_NLL=bestloss,elapsed_s=time.monotonic()-start))
        if best>=305 or time.monotonic()-start>=c['fit_seconds']:break
    results={}
    for name,v in (('best',selected),('last',delta)):
        ww,bb=native(v);state={k:t.clone() for k,t in original.items()};state['linear.weight']=torch.from_numpy(ww);state['linear.bias']=torch.from_numpy(bb);assert all(torch.equal(t,state[k]) for k,t in original.items() if not k.startswith('linear.'));torch.save(dict(model=state,protocol=protocol),a.out/(name+'.pt'));p=probability(te['features'],ww,bb);np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=te['ids'],labels=te['labels'],probabilities=p);results[name]=dict(TRAIN=metrics(tr['labels'],probability(tr['features'],ww,bb)),TEST_development=metrics(te['labels'],p),NLL=nll(p),PT_SHA256=sha(a.out/(name+'.pt')))
    state=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)['model'];model.load_state_dict(state)
    with torch.no_grad():p=model.linear(torch.from_numpy(te['features'])).softmax(1).numpy()
    expected=np.load(a.out/'best_TEST_predictions.npz')['probabilities'];assert np.max(np.abs(p-expected))<=3e-6 and count(p)==best;save(a.out/'history.json',history);save(a.out/'report.json',dict(state='complete',initial_correct=initial,selected_correct=best,actual_steps=step,results=results,new_correct=int(((baseline.argmax(1)!=te['labels'])&(p.argmax(1)==te['labels'])).sum()),new_wrong=int(((baseline.argmax(1)==te['labels'])&(p.argmax(1)!=te['labels'])).sum()),all_nonhead_frozen=True,sealed_feature_composition_verified=True,full_CPU_replay_passed=False,fresh_full_raw_CCD_replay=False,TEST_development=True,TEST_gradients=False,SDK_opened=False,GPU_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--depth',type=int,choices=(4,));p.add_argument('--checkpoint-sha')
    for key in ('out','checkpoint','feature-run','sealed-backup-root','fit-profile'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    try:selftest() if a.selftest else run(a)
    except BaseException as e:
        if a.out and a.out.exists():save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
