"""Shared existing readout gradient sum and unchanged inference parameter proof."""
import types,numpy as np,torch
import caltech20_rich_L6_shared_supervision as m
import check_caltech20_rich_prototype_consistency as previous
import audit_caltech20_rich_L6_shared_supervision as audit
def run():
    m.completed_mild();ns=dict(previous.run.__globals__);ns.update(m=m,audit=audit);proof=types.FunctionType(previous.run.__code__,ns,'same_prototype_parent_CPU_math',previous.run.__defaults__,previous.run.__closure__);proof()
    rg=np.random.default_rng(17);w=torch.tensor(rg.normal(size=(20,7)),requires_grad=True);b=torch.tensor(rg.normal(size=20),requires_grad=True);x=[torch.tensor(rg.normal(size=(5,7))) for _ in range(3)];y=torch.tensor(np.eye(20)[[0,3,7,14,19]]*.95+.05/20);weights=torch.tensor(rg.uniform(.5,2,20));loss=sum(a*m.auxiliary_CE(torch.softmax(z@w.T+b,1),y,weights) for a,z in zip((1.,.05,.05),x));gw,gb=torch.autograd.grad(loss,(w,b));ew=np.zeros((20,7));eb=np.zeros(20)
    for a,z in zip((1.,.05,.05),x):
        logits=z.numpy()@w.detach().numpy().T+b.detach().numpy();p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True);target=y.numpy()*weights.numpy();g=(p*target.sum(1,keepdims=True)-target)/5;ew+=a*g.T@z.numpy();eb+=a*g.sum(0)
    assert np.max(np.abs(gw.numpy()-ew))<1e-12 and np.max(np.abs(gb.numpy()-eb))<1e-12
    model=m.initialize(6,False);assert sum(p.numel() for p in model.linear.parameters())==5140 and len(list(model.parameters()))==30 and not any('aux' in n for n in model.state_dict());q=m.s.read(m.CPU/'checks.json');q.update(shared_head_derivative_passed=True,additional_inference_heads=0,shared_readout_parameters=5140,real_model_forward_calls=0,actual_mild_review=m.completed_mild()['actual_new_epochs']);m.s.save(m.CPU/'checks.json',q)
if __name__=='__main__':
    try:run()
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
