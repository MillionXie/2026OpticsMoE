"""One router, alternating E/G, ABO-style normalized learned electronic fusion."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from caltech_repair234_model import CorrectedMoE

CONTRACT='singleRouter_EG_cycles_RMSlearnedAlphaGT07_CCD256Linear20_biasfree_v2'
ALPHA_BOUNDS=(.7001,.95)


def rms(value):
    return (value.square().sum((-2,-1),keepdim=True)/(value.shape[-2]*value.shape[-1])).sqrt().clamp_min(1e-6)


def alpha_value(raw):return ALPHA_BOUNDS[0]+(ALPHA_BOUNDS[1]-ALPHA_BOUNDS[0])*raw.sigmoid()


def fuse(e,o,raw):
    e,o=e.float(),o.float();re,ro=rms(e).detach(),rms(o).detach();alpha=alpha_value(raw)
    mixed=(1-alpha)*(e/re)+alpha*(o/ro)
    return re*mixed/rms(mixed).detach()


def features(amplitude):
    pooled=F.adaptive_avg_pool2d(amplitude.float()[:,None],(16,16)).flatten(1)
    return F.layer_norm(pooled,(256,),eps=1e-5)


def energies(intensity,masks):
    return torch.stack([torch.einsum('hw,khw->k',v,masks) for v in intensity])


class SmallElectronicResidual(nn.Module):
    def __init__(self):
        super().__init__();self.local=nn.Conv2d(1,1,3,padding=1,bias=False)
        nn.init.zeros_(self.local.weight)

    def forward(self,x,size):
        x=F.interpolate(x[:,None],(size,size),mode='bilinear',align_corners=False)
        normalized=F.layer_norm(x,x.shape[-2:],eps=1e-5)
        return (x+.1*torch.tanh(self.local(normalized))).clamp_min(0)[:,0]


class HybridMoE(CorrectedMoE):
    def __init__(self,depth,rho):
        assert depth in (2,4,6) and rho in (0.,.3)
        super().__init__(min(depth,4),rho)
        for k in range(self.depth,depth):
            lo=len(self.phase_sets);shapes=[(224,224)]*4 if k%2==0 else [(478,478)]
            for shape in shapes:self.phase_sets.append(nn.Parameter(torch.empty(shape).uniform_(-.1,.1)))
            self.stage_slices.append((lo,len(self.phase_sets)))
        self.depth=depth;self.stages=['expert' if k%2==0 else 'global' for k in range(depth)]
        self.electronic=nn.ModuleList([SmallElectronicResidual() for _ in range(depth)])
        raw=math.log((.8-ALPHA_BOUNDS[0])/(ALPHA_BOUNDS[1]-.8))
        self.fusion_logits=nn.ParameterList([nn.Parameter(torch.tensor(raw)) for _ in range(depth)])
        self.linear=nn.Linear(256,20);nn.init.xavier_uniform_(self.linear.weight,gain=.1);nn.init.zeros_(self.linear.bias)
        self.readout={};self.trace=[]
        assert sum(p.numel() for n,p in self.named_parameters() if n=='raw_router' or n.startswith('phase_sets.'))==50176+429188*(depth//2)
        assert sum(p.numel() for n,p in self.named_parameters() if n!='raw_router' and not n.startswith('phase_sets.'))==5140+10*depth

    def forward(self,amplitude,callback=None,ids=None):
        self.last={};self.trace=[];intensity=self.capture(amplitude,'router',callback=callback,ids=ids);self.trace.append('router')
        energy=energies(intensity,self.router_masks);centered=energy-energy.mean(1,keepdim=True)
        dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1)
        weights=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8);self.routing_weights=weights.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights,callback,ids);self.trace.append(self.stages[k])
            target=self.stages[k+1] if k<self.depth-1 else 'global';size=478 if target=='global' else 224
            optical=self.nonlinear(intensity,target);electronic=self.electronic[k](amplitude,size)
            amplitude=fuse(electronic,optical,self.fusion_logits[k])
        f=features(amplitude);probability=torch.softmax(self.linear(f),1);energy=energies(intensity,self.detectors)
        self.readout=dict(features=f.detach(),amplitude=amplitude.detach(),ccd=intensity.detach())
        return probability,energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)

    def phase_smoothness(self):
        terms=[]
        for name,raw in self.named_parameters():
            if name!='raw_router' and not name.startswith('phase_sets.'):continue
            phase=2*math.pi*torch.sigmoid(raw)
            terms.extend([(phase[1:]-phase[:-1]).square().mean(),(phase[:,1:]-phase[:,:-1]).square().mean()])
        return torch.stack(terms).mean()

    def alphas(self):return [float(alpha_value(q).detach()) for q in self.fusion_logits]

    def contract(self):
        return dict(id=CONTRACT,main_depth=self.depth,router_count=1,stages=self.stages,captures=self.depth+1,
                    experts=4,optical_phase_parameters=50176+429188*(self.depth//2),optical_phase_tensors=1+5*(self.depth//2),
                    electronic_parameters=5140+10*self.depth,terminal_linear_parameters=5140,alpha_parameters=self.depth,
                    electronic_architecture='Each E/G: 1channel3x3 residual9params + learned scalar alpha; final pooled16/nonaffineLN/Linear256to20',
                    alpha_bounds=list(ALPHA_BOUNDS),alpha_initial=.8,alpha_definition='per-stage RMS normalized feature weight, not global optical power or output probability',
                    fusion='ABO fuse: normalize e/o by detached per-sample RMS; (1-alpha)e+alpha o; normalize mixture and restore electronic reference RMS',
                    rho=self.rho,rho_definition='coherent phase amplitude leakage expert/global only; router unchanged',
                    quantization='joint amplitude and phase8bit inside train/val/test',
                    deployment='same learned residual/alpha/linear state during simulation and future physical adaptation')
