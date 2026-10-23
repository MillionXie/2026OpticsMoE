"""Bounded TRAIN-only local row repair, with explicitly TEST-informed planning.

Uses the published margin/local-row/binary-rest/full-curvature experience, not
its legacy runtime code injection or simulation asset dependencies. Only the
existing affine row is changed; no inference module is created.
"""
import time,numpy as np


def validate(c):
    origin=dict(maximum_classes=8,steps_per_class=64,temperature=2.,label_smoothing=.02,neighbor_weight=2.,KL=2.,anchor=.2,trust_step=.25,row_radius=.08,seed=73)
    continued={**origin,'maximum_classes':12,'skip_classes':[3,2,0,4,9,15,7,12]}
    continued_E2={**origin,'maximum_classes':12,'skip_classes':[3,2,0,19,4,7,10,13]}
    full_E2={**origin,'maximum_classes':20}
    assert c in (origin,continued,continued_E2,full_E2)


def selection(features,labels,w,b):
    import torch
    from torch.nn import functional as F
    with torch.inference_mode():
        logits=F.linear(torch.from_numpy(features.astype(np.float32)),torch.from_numpy(w.astype(np.float32)),torch.from_numpy(b.astype(np.float32)))
        prediction=logits.argmax(1).numpy();nll=float(F.cross_entropy(logits,torch.from_numpy(labels.astype(np.int64))))
    return int((prediction==labels).sum()),nll,prediction


def TRAIN_direction(x,target,reference,q,u,v,u0,v0,c):
    """Exact binary TRAIN curvature plus parameter anchor, no TEST arguments."""
    score=x@u+v-reference;p=1/(1+np.exp(-np.clip(score,-60,60)));soft=1/(1+np.exp(-np.clip(score/c['temperature'],-60,60)))
    weights=np.ones(len(x));weights[:16]=c['neighbor_weight']
    dz=(weights*c['temperature']*(soft-target)+c['KL']*(p-q))/len(x)
    raw=np.r_[x.T@dz+c['anchor']*(u-u0),dz.sum()+c['anchor']*(v-v0)]
    cur=(weights*soft*(1-soft)+c['KL']*p*(1-p))/len(x);A=np.c_[x,np.ones(len(x))];B=A*np.sqrt(cur)[:,None];alpha=c['anchor']
    direction=raw/alpha-B.T@np.linalg.solve(np.eye(len(x))+B@B.T/alpha,B@raw)/(alpha*alpha)
    assert np.isfinite(direction).all()
    return direction,raw,B


def projected_update(u,v,direction,u0,v0,c):
    proposal=np.r_[u,v]-c['trust_step']*direction;origin=np.r_[u0,v0]
    delta=proposal-origin;scale=min(1.,c['row_radius']/max(float(np.linalg.norm(delta)),1e-12));proposal=origin+delta*scale
    assert np.linalg.norm(proposal-origin)<=c['row_radius']+1e-10
    return proposal[:-1],float(proposal[-1])


def fit(train,labels,development,development_labels,w,b,c,target,seconds,emit):
    validate(c);x=np.asarray(train,dtype=np.float64);z=np.asarray(development,dtype=np.float64);y=np.asarray(labels);dy=np.asarray(development_labels)
    assert x.shape==(1600,256) and z.shape==(411,256) and np.bincount(y,minlength=20).tolist()==[80]*20
    w=np.asarray(w,dtype=np.float32).astype(float);b=np.asarray(b,dtype=np.float32).astype(float)
    initial,initial_nll,initial_pred=selection(z,dy,w,b);best,bnll=initial,initial_nll;selected=updates=0;history=[];plans=[];used=list(c.get('skip_classes',[]));started=time.monotonic();lastw,lastb=w.copy(),b.copy()
    xn=x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12);zn=z/np.linalg.norm(z,axis=1,keepdims=True).clip(1e-12)
    for stage in range(1,c['maximum_classes']+1):
        _,_,pred=selection(z,dy,w,b);wrong=pred!=dy;counts=np.bincount(dy[wrong],minlength=20)
        remaining=[k for k in range(20) if k not in used and counts[k]>0]
        if not remaining:break
        label=min(remaining,key=lambda k:(-int(counts[k]),k));used.append(label);bad=np.flatnonzero(wrong&(dy==label));pool=np.flatnonzero(y==label);other=np.flatnonzero(y!=label);near=[]
        for i in bad:near.extend(pool[np.argsort(-(xn[pool]@zn[i]),kind='stable')[:3]].tolist())
        logits=x@w.T+b;risk=logits[other,label]-logits[other,y[other]];hard=other[np.argsort(-risk,kind='stable')[:128]]
        rest=logits.copy();rest[:,label]=-np.inf;mx=rest.max(1);reference=mx+np.log(np.exp(rest-mx[:,None]).sum(1))
        targets=np.where(y==label,1-c['label_smoothing']/2,c['label_smoothing']/2);q=1/(1+np.exp(-np.clip(logits[:,label]-reference,-60,60)))
        parentw,parentb=w.copy(),b.copy();u0,v0=w[label].copy(),float(b[label]);u,v=u0.copy(),v0;chosen=(u.copy(),v);rng=np.random.default_rng(c['seed']+stage)
        plans.append(dict(stage=stage,label=label,TEST_strategy_error_indices=bad.tolist(),TRAIN_neighbors=near,hard_TRAIN_indices=hard.tolist(),TEST_in_gradient=False))
        for step in range(1,c['steps_per_class']+1):
            ix=np.r_[rng.choice(near,16,replace=True),rng.choice(pool,16,replace=False),rng.choice(hard,16,replace=False),rng.choice(len(x),16,replace=False)]
            assert ix.shape==(64,) and ix.min()>=0 and ix.max()<1600
            direction,_,_=TRAIN_direction(x[ix],targets[ix],reference[ix],q[ix],u,v,u0,v0,c);u,v=projected_update(u,v,direction,u0,v0,c)
            ww,bb=parentw.copy(),parentb.copy();ww[label]=u.astype(np.float32);bb[label]=float(np.float32(v));fixed=np.arange(20)!=label
            assert np.array_equal(ww[fixed],parentw[fixed]) and np.array_equal(bb[fixed],parentb[fixed])
            correct,nll,prediction=selection(z,dy,ww,bb);updates+=1
            if correct>best or (correct==best and nll<bnll-1e-4):best,bnll,selected=correct,nll,updates;chosen=(ww[label].copy(),float(bb[label]))
            row=dict(update=updates,stage=stage,label=label,step=step,TRAIN_indices=ix.tolist(),TRAIN_consumed=64,TEST_correct=correct,TEST_NLL=nll,selected_correct=best,selected_NLL=bnll,selected_update=selected,
                recovered=int(((initial_pred!=dy)&(prediction==dy)).sum()),lost=int(((initial_pred==dy)&(prediction!=dy)).sum()),row_delta_norm=float(np.linalg.norm(np.r_[u-u0,v-v0])),TEST_gradient=False)
            history.append(row);lastw,lastb=ww,bb
            if step==1 or step%16==0 or best>=target:emit(row)
            if best>=target or time.monotonic()-started>=seconds:break
        w,b=parentw.copy(),parentb.copy();w[label]=chosen[0];b[label]=chosen[1]
        if best>=target or time.monotonic()-started>=seconds:break
    final,final_nll,_=selection(z,dy,w,b);assert final==best and abs(final_nll-bnll)<1e-5
    return dict(best_weight=w.astype(np.float32),best_bias=b.astype(np.float32),last_weight=lastw.astype(np.float32),last_bias=lastb.astype(np.float32),best=best,nll=bnll,selected=selected,updates=updates,
        history=history,plans=plans,initial_correct=initial,initial_NLL=initial_nll,classes=used,TEST_informed_strategy=True,independent_generalization=False,gradient_data='TRAIN_only',per_update_parameters=257,elapsed_s=time.monotonic()-started)


def selftest():
    c=dict(maximum_classes=8,steps_per_class=64,temperature=2.,label_smoothing=.02,neighbor_weight=2.,KL=2.,anchor=.2,trust_step=.25,row_radius=.08,seed=73);validate(c)
    rng=np.random.default_rng(73);x=rng.normal(size=(64,7));u=rng.normal(size=7)*.1;v=.1;ref=rng.normal(size=64);q=rng.uniform(.1,.9,64);t=rng.integers(0,2,64).astype(float)
    d,g,B=TRAIN_direction(x,t,ref,q,u,v,u,v,c);exact=np.linalg.solve(c['anchor']*np.eye(8)+B.T@B,g)
    assert np.max(np.abs(d-exact))<1e-10
    new,bias=projected_update(u,v,d*100,u,v,c);assert np.linalg.norm(np.r_[new-u,bias-v])<=c['row_radius']+1e-10
    print(__import__('json').dumps(dict(passed=True,TRAIN_curvature_matches_direct_solve=True,trust_radius_enforced=True,TEST_gradient=False,SDK_used=False,GPU_used=False)))


if __name__=='__main__':selftest()
