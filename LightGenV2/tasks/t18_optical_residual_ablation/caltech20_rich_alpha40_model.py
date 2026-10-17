"""Same physical graph; alpha>.4 in precise coordinates, no added parameters."""
import math
import torch
from caltech20_rich_robust_model import RobustMoE,rms,features,energies
LOW,HIGH=.4001,.95

def remap(state,old_low):
    z={n:v.detach().clone() for n,v in state.items()}
    for n,v in z.items():
        if n.startswith('fusion_logits.'):
            # Reproduce the parent arithmetic before changing raw coordinates.
            a=(old_low+(.95-old_low)*v.sigmoid()).float().double()
            assert LOW<a<HIGH
            z[n]=torch.log((a-LOW)/(HIGH-a))
    return z

class Alpha40MoE(RobustMoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho)
        for p in self.fusion_logits:p.data=p.data.double()
        self.branch_statistics=[]
        self.set_alpha(.45)

    def set_alpha(self,value):
        assert LOW<value<HIGH
        with torch.no_grad():
            for p in self.fusion_logits:p.fill_(math.log((value-LOW)/(HIGH-value)))

    def forward(self,amplitude,callback=None,ids=None):
        self.last={};self.trace=[];self.branch_statistics=[]
        intensity=self.capture(amplitude,'router',callback=callback,ids=ids);self.trace.append('router')
        energy=energies(intensity,self.router_masks);centered=energy-energy.mean(1,keepdim=True)
        dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1)
        weights=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8);self.routing_weights=weights.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights,callback,ids);self.trace.append(self.stages[k])
            target=self.stages[k+1] if k<self.depth-1 else 'global';size=478 if target=='global' else 224
            optical=self.nonlinear(intensity,target);electronic=self.electronic[k](amplitude,size)
            re,ro=rms(electronic).detach(),rms(optical).detach()
            a=(LOW+(HIGH-LOW)*self.fusion_logits[k].sigmoid()).float()
            en,on=electronic/re,optical/ro
            self.branch_statistics.append(dict(alpha=float(a.detach()),normalized_cosine=float((en.detach()*on.detach()).mean()),electronic_rms=float(re.mean()),optical_rms=float(ro.mean())))
            mixed=(1-a)*en+a*on;amplitude=re*mixed/rms(mixed).detach()
        f=features(amplitude);p=torch.softmax(self.linear(f),1);energy=energies(intensity,self.detectors)
        self.readout=dict(features=f.detach(),amplitude=amplitude.detach(),ccd=intensity.detach())
        return p,energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)

    def alphas(self):return [float((LOW+(HIGH-LOW)*p.sigmoid()).float().detach()) for p in self.fusion_logits]

    def contract(self):
        c=super().contract();c.update(id='singleRouter_EG_alpha04_preciseCoordinates_20261018',alpha_bounds=[LOW,HIGH],alpha_initial=.45,alpha_coordinate_dtype='float64, effective fusion float32',training_only_noise_and_dropout=True,unchanged_inference_parameters=True)
        return c
