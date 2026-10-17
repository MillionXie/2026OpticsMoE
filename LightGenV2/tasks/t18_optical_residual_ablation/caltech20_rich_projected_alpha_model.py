"""Same fusion function; effective alpha coordinates with projected AdamW."""
import torch
from caltech20_rich_alpha40_namespace_repair import Alpha40MoE as DeviceSafeMoE
from caltech20_rich_alpha40_model import rms,features,energies
LOW,HIGH=.4001,.95
def convert(state,old_low):
    result={n:v.detach().clone() for n,v in state.items()}
    for n,v in result.items():
        if n.startswith('fusion_logits.'):
            result[n]=(old_low+(.95-old_low)*v.sigmoid()).float().double()
    return result
class ProjectedMoE(DeviceSafeMoE):
    def set_alpha(self,value):
        assert LOW<value<HIGH
        with torch.no_grad():
            for p in self.fusion_logits:p.fill_(value)
    def project(self):
        with torch.no_grad():
            for p in self.fusion_logits:p.clamp_(LOW,HIGH)
    def alphas(self):return [float(p.float().detach()) for p in self.fusion_logits]
    def forward(self,amplitude,callback=None,ids=None):
        self.last={};self.trace=[];self.branch_statistics=[]
        intensity=self.capture(amplitude,'router',callback=callback,ids=ids);self.trace.append('router')
        energy=energies(intensity,self.router_masks);centered=energy-energy.mean(1,keepdim=True)
        dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1)
        weights=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8);self.routing_weights=weights.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights,callback,ids);self.trace.append(self.stages[k])
            target=self.stages[k+1] if k<self.depth-1 else 'global';size=478 if target=='global' else 224
            optical=self.nonlinear(intensity,target);electronic=self.electronic[k](amplitude,size);re,ro=rms(electronic).detach(),rms(optical).detach()
            a=self.fusion_logits[k].float();mixed=(1-a)*(electronic/re)+a*(optical/ro);amplitude=re*mixed/rms(mixed).detach()
        f=features(amplitude);p=torch.softmax(self.linear(f),1);energy=energies(intensity,self.detectors);self.readout=dict(features=f.detach(),amplitude=amplitude.detach(),ccd=intensity.detach())
        return p,energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)
    def contract(self):
        c=super().contract();c.update(id='singleRouter_EG_directProjectedAlpha04_curriculum_20261018',alpha_bounds=[LOW,HIGH],alpha_coordinate_dtype='float64 DIRECT effective coefficient, float32 fusion; project after each AdamW; NOT sigmoid logits',state_key_semantics='fusion_logits.* legacy names now store effective alpha; load ONLY ProjectedMoE class',unchanged_inference_function=True)
        return c
