"""Identical parameter-free, nonnegative OEO transfer functions for both models."""
import torch
from torch.nn import functional as F

CANDIDATES=('softplus2','relu_tanh','relu_softsign')

def activate(z,name):
    if name=='softplus2':return F.softplus(z,beta=2.,threshold=20.)
    if name=='relu_tanh':return torch.tanh(F.relu(z))
    if name=='relu_softsign':return F.softsign(F.relu(z))
    raise ValueError(name)

def reference_activate(z,name):
    """Independent expressions used only in GPU preflight."""
    dtype=z.dtype;z=z.double()
    if name=='softplus2':return torch.where(2*z>20,z,torch.logaddexp(torch.zeros_like(z),2*z)/2).to(dtype)
    r=z.clamp_min(0)
    if name=='relu_tanh':return (2/(1+torch.exp(-2*r))-1).to(dtype)
    if name=='relu_softsign':return (r/(1+r)).to(dtype)
    raise ValueError(name)
