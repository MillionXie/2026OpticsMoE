"""Same alternating hybrid graph; explicitly asymmetric learned-alpha bounds."""
import math
import torch
from caltech20_alt246_genreg_model import RegularizedMoE
from caltech20_alt246_biasfree_model import rms,features,energies

BOUNDS={0.:(.8501,.95),.3:(.7001,.95)}
INITIAL=.9

def alpha_value(raw,rho):
    lo,hi=BOUNDS[rho];return lo+(hi-lo)*raw.sigmoid()

def fuse(e,o,raw,rho):
    e,o=e.float(),o.float();re,ro=rms(e).detach(),rms(o).detach();alpha=alpha_value(raw,rho)
    mixed=(1-alpha)*(e/re)+alpha*(o/ro)
    return re*mixed/rms(mixed).detach()

class PairModel(RegularizedMoE):
    def __init__(self,depth,rho):
        super().__init__(depth,rho);lo,hi=BOUNDS[rho];raw=math.log((INITIAL-lo)/(hi-INITIAL))
        with torch.no_grad():
            for value in self.fusion_logits:value.fill_(raw)

    def forward(self,amplitude,callback=None,ids=None):
        self.last={};self.trace=[];intensity=self.capture(amplitude,'router',callback=callback,ids=ids);self.trace.append('router')
        energy=energies(intensity,self.router_masks);centered=energy-energy.mean(1,keepdim=True)
        dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1)
        weights=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8);self.routing_weights=weights.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights,callback,ids);self.trace.append(self.stages[k])
            target=self.stages[k+1] if k<self.depth-1 else 'global';size=478 if target=='global' else 224
            optical=self.nonlinear(intensity,target);electronic=self.electronic[k](amplitude,size)
            amplitude=fuse(electronic,optical,self.fusion_logits[k],self.rho)
        f=features(amplitude);probability=torch.softmax(self.linear(f),1);energy=energies(intensity,self.detectors)
        self.readout=dict(features=f.detach(),amplitude=amplitude.detach(),ccd=intensity.detach())
        return probability,energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)

    def alphas(self):return [float(alpha_value(q,self.rho).detach()) for q in self.fusion_logits]

    def contract(self):
        c=super().contract();c.update(id='singleRouter_EG_cycles_RMSlearnedAlpha_asymmetric_pair50_20261017',alpha_bounds=list(BOUNDS[self.rho]),alpha_initial=INITIAL,comparison_scope='Optical rho plus asymmetric electronic fusion constraints; not pure optical-rho causality',state_key_shapes_unchanged=True);return c
