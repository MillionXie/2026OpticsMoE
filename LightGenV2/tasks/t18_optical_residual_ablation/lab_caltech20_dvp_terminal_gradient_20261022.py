"""Training-only RMS adjoint audit; original inference graph/values unchanged."""
import torch
from torch.nn import functional as F
from caltech20_alt246_biasfree_model import rms,features
from lab_caltech20_rich_tail_refine_20261018 import tail


def exact_RMS_training_tail(electronic,head,alpha,x,optical):
    """Same original terminal values, retain normalization derivatives in TRAIN."""
    e=electronic(x,478);re,ro=rms(e),rms(optical)
    mixed=(1-alpha.float())*(e/re)+alpha.float()*(optical/ro)
    return head(features(re*mixed/rms(mixed)))


def TRAIN_gradient_audit(electronic,head,alpha,x,optical,labels):
    """Finite differences on TRAIN only; no TEST inputs or parameter fitting."""
    params=[electronic.local.weight,alpha];saved=[p.detach().clone() for p in params]
    old_flags=[p.requires_grad for p in params]
    try:
        for p in params:p.requires_grad_(True)
        old=tail(electronic,head,alpha,x,optical);new=exact_RMS_training_tail(electronic,head,alpha,x,optical)
        assert torch.equal(old,new),'Training-only derivative must not alter original forward'
        lo=F.cross_entropy(old,labels);ln=F.cross_entropy(new,labels)
        go=torch.autograd.grad(lo,params);gn=torch.autograd.grad(ln,params)
        ov=torch.cat([g.detach().double().flatten() for g in go]);nv=torch.cat([g.detach().double().flatten() for g in gn])
        assert torch.isfinite(ov).all() and torch.isfinite(nv).all() and float(nv.norm())>1e-10
        direction=nv/nv.norm();rows=[]
        for eps in (.003,.001):
            scores=[]
            for sign in (1,-1):
                offset=0
                with torch.no_grad():
                    for p,v in zip(params,saved):
                        n=p.numel();p.copy_(v+sign*eps*direction[offset:offset+n].reshape_as(p).to(p.dtype));offset+=n
                    scores.append(float(F.cross_entropy(tail(electronic,head,alpha,x,optical),labels)))
            fd=(scores[0]-scores[1])/(2*eps)
            rows.append(dict(epsilon=eps,finite_difference=fd,original_detached_adjoint=float(ov@direction),full_RMS_adjoint=float(nv@direction),original_error=abs(fd-float(ov@direction)),full_RMS_error=abs(fd-float(nv@direction))))
    finally:
        with torch.no_grad():
            for p,v in zip(params,saved):p.copy_(v)
        for p,flag in zip(params,old_flags):p.requires_grad_(flag)
    assert all(torch.equal(p,v) for p,v in zip(params,saved))
    return dict(TRAIN_only=True,TEST_used=False,TRAIN_samples=len(labels),original_forward_bitwise_equal=True,parameters_restored=True,directional_checks=rows,original_gradient_norm=float(ov.norm()),full_RMS_gradient_norm=float(nv.norm()),gradient_cosine=float((ov@nv)/(ov.norm()*nv.norm()).clamp_min(1e-12)),new_inference_layers=0,not_proven_accuracy_cause=True)


def selftest():
    from caltech20_alt246_biasfree_model import SmallElectronicResidual
    torch.set_num_threads(2);torch.manual_seed(73)
    e=SmallElectronicResidual();h=torch.nn.Linear(256,20);a=torch.nn.Parameter(torch.tensor(.8,dtype=torch.float64))
    x=torch.rand(2,478,478)+.1;o=torch.rand_like(x)+.1;result=TRAIN_gradient_audit(e,h,a,x,o,torch.tensor([0,1]))
    assert result['original_forward_bitwise_equal'] and result['parameters_restored']
    assert all(r['full_RMS_error']<.01 for r in result['directional_checks'])
    print(result)


if __name__=='__main__':selftest()
