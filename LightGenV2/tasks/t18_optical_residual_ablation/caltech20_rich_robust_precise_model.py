"""Only learned-alpha coordinates use float64; physical fusion stays float32."""
import torch
from caltech20_rich_robust_model import RobustMoE,LOW,HIGH,rms,features,energies

def remap(state):
    z={n:v.detach().clone() for n,v in state.items()}
    for n,v in z.items():
        if n.startswith('fusion_logits.'):
            # First reproduce the old FLOAT32 alpha, then invert in FLOAT64.
            alpha=(.7001+(.95-.7001)*v.sigmoid()).double()
            z[n]=torch.log((alpha-LOW)/(HIGH-alpha))
    return z

class PreciseMoE(RobustMoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho)
        for p in self.fusion_logits:p.data=p.data.double()

    def forward(self,amplitude,callback=None,ids=None):
        self.last={};self.trace=[]
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
            mixed=(1-a)*(electronic/re)+a*(optical/ro)
            amplitude=re*mixed/rms(mixed).detach()
        f=features(amplitude);p=torch.softmax(self.linear(f),1);energy=energies(intensity,self.detectors)
        self.readout=dict(features=f.detach(),amplitude=amplitude.detach(),ccd=intensity.detach())
        return p,energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)

    def alphas(self):return [float((LOW+(HIGH-LOW)*p.sigmoid()).float().detach()) for p in self.fusion_logits]

    def contract(self):
        c=super().contract();c.update(id='singleRouter_EG_alpha06_preciseCoordinates_20261018',alpha_coordinate_dtype='float64; effective fusion alpha float32; all other learned parameters float32',repair='Original L6 stopped before any optimizer/TEST; float32 inverse-logit rounding changed one initial alpha by5.96e-8. New coordinate precision preserves initial effective values without relaxing parity threshold.')
        return c
