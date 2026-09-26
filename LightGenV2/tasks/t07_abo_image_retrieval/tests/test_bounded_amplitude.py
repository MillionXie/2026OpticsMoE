import torch
from ..standalone.optics import bounded_amplitude,OpticalPath

def test_bounded_contract():
    spec=dict(kind='tanh',scale=.5)
    x=torch.tensor([0.,.001,.1,.5,1.,10.],requires_grad=True)
    y=bounded_amplitude(x,spec)
    assert y[0]==0 and y.min()>=0 and y.max()<=1
    assert torch.all(y[1:]>=y[:-1])
    y.sum().backward();assert torch.isfinite(x.grad).all() and x.grad[1]>0
    gray=(255*y.detach()).round().byte()
    assert (gray.float()/255-y.detach()).abs().max()<=.5/255+1e-7
    path=OpticalPath();path.bounded_amplitude=path.router.bounded_amplitude=spec
    path.eval();latent=torch.randn(1,38,192)
    expert,weights=path.expert(latent);output=path.global_stage(expert,weights)
    output.square().mean().backward()
    assert path.global_phase.grad is not None and torch.isfinite(path.global_phase.grad).all()

if __name__=='__main__':test_bounded_contract();print('bounded contract and optical gradients passed')
