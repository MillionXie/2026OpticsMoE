"""One coupled original-affine TRUST-region correction, TRAIN gradients only.

Native W/b are folded exactly. No inference delta module or TEST derivatives.
"""
import math,time


def objective(weight,bias,origin,features,labels,weights,teacher):
    import torch
    from torch.nn import functional as F
    logits=F.linear(features,weight,bias);ce=F.cross_entropy(logits,labels,weight=weights,label_smoothing=.03)
    target=teacher.detach();gate=target.argmax(1)==labels
    per=F.kl_div(F.log_softmax(logits/2,dim=1),target,reduction='none').sum(1)*4
    kl=(per*gate).sum()/gate.sum().clamp_min(1);anchor=.2*((weight-origin['weight']).square().sum()+(bias-origin['bias']).square().sum())
    return ce+.5*kl+anchor,ce,kl,anchor


def project(head,origin):
    import torch
    with torch.no_grad():
        dw=head.weight-origin['weight'];db=head.bias-origin['bias'];norm=torch.cat([dw,db[:,None]],1).norm(dim=1)
        scale=(.075/norm.clamp_min(1e-12)).clamp_max(1);head.weight.copy_(origin['weight']+dw*scale[:,None]);head.bias.copy_(origin['bias']+db*scale)


def fold(head,origin,raw_parent):
    return {k:(raw_parent[k]+(v-origin[k])).detach().float().clone()for k,v in head.state_dict().items()}


def fit(head,train,labels,weights,raw_parent,development,development_labels,baseline,target,seconds,emit):
    import torch
    from torch.nn import functional as F
    assert train.shape==(1600,256) and development.shape==(411,256) and not development.requires_grad
    assert labels.shape==(1600,) and development_labels.shape==(411,)
    origin={k:v.detach().clone()for k,v in head.state_dict().items()};start=time.monotonic();history=[]
    with torch.no_grad():teacher=F.softmax(F.linear(train,head.weight,head.bias)/2,dim=1)
    initial=fold(head,origin,raw_parent);assert all(torch.equal(initial[k],raw_parent[k].float())for k in initial)
    best,bnll,selected=baseline['correct'],baseline['nll'],0;best_head=initial
    optimizer=torch.optim.Adam(head.parameters(),lr=.001)
    for step in range(1,801):
        lr=.001*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/799)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);loss,ce,kl,anchor=objective(head.weight,head.bias,origin,train,labels,weights,teacher)
        assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all()for v in head.parameters());torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step();project(head,origin)
        if step==1 or step%20==0 or step==800:
            candidate=fold(head,origin,raw_parent)
            native_delta=torch.cat([candidate['weight'].double()-raw_parent['weight'],(candidate['bias'].double()-raw_parent['bias'])[:,None]],1);radius=float(native_delta.norm(dim=1).max());assert radius<=.080001
            with torch.inference_mode():
                logits=F.linear(development,candidate['weight'],candidate['bias']);correct=int((logits.argmax(1)==development_labels).sum());nll=float(F.cross_entropy(logits,development_labels))
            if correct>best or(correct==best and nll<bnll-1e-4):best,bnll,selected=correct,nll,step;best_head={k:v.clone()for k,v in candidate.items()}
            row=dict(full_TRAIN_update=step,TRAIN_consumed=1600,TRAIN_CE=float(ce.detach()),TRAIN_parent_KL=float(kl.detach()),TRAIN_parameter_anchor=float(anchor.detach()),
                TEST_correct=correct,TEST_NLL=nll,selected_correct=best,selected_update=selected,native_row_delta_radius=radius,elapsed_s=time.monotonic()-start,TEST_gradient=False)
            history.append(row);emit(row,history)
        if best>=target or time.monotonic()-start>=seconds:break
    return dict(best_head=best_head,last_head=fold(head,origin,raw_parent),correct=best,nll=bnll,selected_update=selected,updates=step,history=history)


def selftest():
    import torch
    from torch.nn import functional as F
    torch.manual_seed(73);head=torch.nn.Linear(6,3).double();raw={k:v.detach().clone()for k,v in head.state_dict().items()};origin={k:v.detach().clone()for k,v in head.state_dict().items()}
    assert all(torch.equal(fold(head,origin,raw)[k],raw[k].float())for k in raw)
    x=torch.randn(20,6,dtype=torch.float64);y=torch.arange(20)%3;q=F.softmax(F.linear(x,head.weight,head.bias).detach()/2,1).requires_grad_();loss,*_=objective(head.weight,head.bias,origin,x,y,torch.ones(3,dtype=torch.float64),q);loss.backward()
    assert q.grad is None and all(torch.isfinite(v.grad).all()for v in head.parameters())
    with torch.no_grad():head.weight.add_(100);head.bias.add_(100)
    project(head,origin);radius=torch.cat([head.weight-origin['weight'],(head.bias-origin['bias'])[:,None]],1).norm(dim=1);assert float(radius.max().detach())<=.075001
    return dict(passed=True,zero_delta_native_identity_exact=True,TRAIN_teacher_detached=True,projected_row_radius_at_most=.075,new_inference_layers=0,TEST_gradient=False)
