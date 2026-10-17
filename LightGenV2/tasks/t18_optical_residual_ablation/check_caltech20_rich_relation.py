"""Analytic relation gradient and exact two-pass chain rule admission."""
import ast,inspect,os
import numpy as np
import torch
import caltech20_rich_relation as m
import check_caltech20_rich_robust as old
exec(compile(ast.parse(inspect.getsource(old.run)),__file__+':same_actual_paths','exec'),globals())

def analytic():
    g=torch.Generator().manual_seed(17);f=torch.randn((5,7),generator=g,dtype=torch.float64,requires_grad=True);t=torch.randn((5,9),generator=g,dtype=torch.float64)
    loss=m.relation_loss(f,t);actual=torch.autograd.grad(loss,f)[0]
    u=torch.nn.functional.normalize(f.detach(),dim=1);v=torch.nn.functional.normalize(t,dim=1);error=u@u.T-v@v.T;error.fill_diagonal_(0)
    du=4*error@u/(5*4);expected=(du-u*(du*u).sum(1,keepdim=True))/f.detach().norm(dim=1,keepdim=True)
    assert torch.allclose(actual,expected,atol=1e-13,rtol=1e-12)
    # Full graph vs cached feature-gradient surrogate, same complete macro pairs.
    w=torch.randn((3,7),generator=g,dtype=torch.float64,requires_grad=True);x=torch.randn((5,3),generator=g,dtype=torch.float64)
    direct=torch.autograd.grad(m.relation_loss(x@w,t),w)[0]
    leaf=(x@w).detach().requires_grad_(True);dg=torch.autograd.grad(m.relation_loss(leaf,t),leaf)[0]
    surrogate=sum(((x[i:i+2]@w)*dg[i:i+2]).sum() for i in range(0,5,2));got=torch.autograd.grad(surrogate,w)[0]
    assert torch.allclose(direct,got,atol=1e-13,rtol=1e-12)
    return dict(analytic_gradient_maxabs=float((actual-expected).abs().max()),two_pass_gradient_maxabs=float((direct-got).abs().max()))

if __name__=='__main__':
    try:
        m.resource_overlap();a=analytic();run();m.s.save(m.CPU/'relation_analytic.json',dict(passed=True,**a),exclusive=True)
    except BaseException as e:
        p=m.CPU.with_name(m.CPU.name+'.failure.json')
        if not p.exists():m.s.save(p,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
