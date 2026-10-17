"""Same current electronic residual graph in both arms; ten-output terminal head."""
import torch
from torch import nn
from torch.nn import functional as F
from caltech20_alt246_biasfree_model import HybridMoE

class MaskedLinear(nn.Linear):
    def __init__(self):super().__init__(256,10);self.mask=None
    def forward(self,x):
        if self.training and self.mask is not None:
            assert self.mask.shape==x.shape;x=x*self.mask
        return F.linear(x,self.weight,self.bias)

class RegularizedMoE(HybridMoE):
    def __init__(self,depth,rho):
        super().__init__(depth,rho)
        self.linear=MaskedLinear();nn.init.xavier_uniform_(self.linear.weight,gain=.1);nn.init.zeros_(self.linear.bias)
        assert sum(p.numel() for n,p in self.named_parameters() if n!='raw_router' and not n.startswith('phase_sets.'))==2570+10*depth
    def contract(self):
        c=super().contract();c.update(id='singleRouter_EG_cycles_RMSlearnedAlphaGT07_CCD256Linear10_biasfree_v1',class_count=10,electronic_parameters=2570+10*self.depth,terminal_linear_parameters=2570,electronic_architecture='Each E/G: biasfree1channel3x3 residual9params + learned scalar alpha; final pooled16/nonaffineLN/Linear256to10',capture_auxiliary='Retain same fixed20 optical capture windows as current20; classification uses full-CCD256 linear10, not window index labels',comparison='BOTH arms identical electronics; only optical coherent rho .3 versus0')
        return c
