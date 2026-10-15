"""Shared simulation/physical-CCD electronic readout; no device access on import."""
import torch
from torch import nn
from torch.nn import functional as F

CONTRACT='CCD478_mean_log1p_pool16_trainSTD_MLP256_32_10_plus_log_optical_v1'

def ccd_features(ccd):
    assert ccd.ndim==3 and ccd.shape[-2:]==(478,478)
    x=ccd.float();assert torch.isfinite(x).all() and (x>=0).all()
    x=torch.log1p(x/x.mean((-2,-1),keepdim=True).clamp_min(1e-6))
    return F.adaptive_avg_pool2d(x[:,None],(16,16)).flatten(1)

class CCDHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(256,32),nn.ReLU(),nn.Linear(32,10))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('mean',torch.zeros(256));self.register_buffer('std',torch.ones(256))
        assert sum(p.numel() for p in self.parameters())==8554
    def from_features(self,features,optical_probability):
        return optical_probability.clamp_min(1e-8).log()+self.net((features-self.mean)/self.std)
    def forward(self,ccd,optical_probability):return self.from_features(ccd_features(ccd),optical_probability)
