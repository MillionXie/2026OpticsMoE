"""Training-only feature dropout; evaluation is the exact frozen hybrid graph."""
import torch
from torch import nn
from torch.nn import functional as F
from caltech20_alt246_biasfree_model import HybridMoE

class MaskedLinear(nn.Linear):
    def __init__(self):
        super().__init__(256,20);self.mask=None

    def forward(self,x):
        if self.training and self.mask is not None:
            assert self.mask.shape==x.shape;x=x*self.mask
        return F.linear(x,self.weight,self.bias)

class RegularizedMoE(HybridMoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho);linear=MaskedLinear();linear.load_state_dict(self.linear.state_dict(),strict=True);self.linear=linear

    def contract(self):
        # No additional inference parameters, layers or optical captures.
        return super().contract()
