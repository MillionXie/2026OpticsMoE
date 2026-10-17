"""Training-only robustness; identical state keys and clean physical inference."""
import math
import torch
from torch.nn import functional as F
from caltech20_rich_pair50_alpha_model import PairModel
from caltech20_alt246_biasfree_model import rms, features, energies
from caltech_deploy234_model import APERTURES

LOW, HIGH = .6001, .95

def remap(state):
    state={k:v.detach().clone() for k,v in state.items()}
    for name,value in state.items():
        if name.startswith('fusion_logits.'):
            alpha=.7001+(.95-.7001)*value.sigmoid()
            state[name]=torch.log((alpha-LOW)/(HIGH-alpha))
    return state

class RobustMoE(PairModel):
    def __init__(self, depth, rho=.3):
        assert rho==.3
        super().__init__(depth,rho)
        self.robust=None
        self.offset=0
        self.phase_anchors={}

    def anchor(self):
        self.phase_anchors={n:p.detach().clone() for n,p in self.named_parameters()
                            if n=='raw_router' or n.startswith('phase_sets.')}

    def context(self, seed, count, cfg):
        """One complete macro context; microbatch and SAM passes share randomness."""
        gen=torch.Generator(device=self.raw_router.device).manual_seed(int(seed))
        masks={}
        for n,p in self.named_parameters():
            if n in self.phase_anchors:
                masks[n]=(torch.rand(p.shape,generator=gen,device=p.device)>=cfg['phase_dropout']).to(p.dtype)
        noise=torch.randn((self.depth+1,count,2,32,32),generator=gen,device=self.raw_router.device)
        self.robust=dict(masks=masks,noise=noise,config=cfg,seed=int(seed),count=count)
        self.offset=0

    def raw(self,name,value):
        if not self.training or self.robust is None:return value
        return self.phase_anchors[name]+self.robust['masks'][name]*(value-self.phase_anchors[name])

    def plane(self, stage):
        plane=torch.ones((518,518),dtype=torch.complex64,device=self.raw_router.device)
        if stage=='router':
            raw=self.raw('raw_router',self.raw_router)
            plane[147:371,147:371]=torch.exp(2j*math.pi*raw.sigmoid())
        else:
            index=int(stage);lo,hi=self.stage_slices[index]
            locations=APERTURES if self.stages[index]=='expert' else [(20,20)]
            for j,(y,x) in zip(range(lo,hi),locations):
                value=self.phase_sets[j];raw=self.raw('phase_sets.'+str(j),value)
                plane[y:y+value.shape[0],x:x+value.shape[1]]=(1-self.rho)*torch.exp(2j*math.pi*raw.sigmoid())+self.rho
        return plane

    def capture(self, amplitude, stage, weights=None, callback=None, ids=None):
        intensity=super().capture(amplitude,stage,weights,callback,ids)
        if self.training and self.robust is not None:
            assert callback is None,'Stochastic training cannot drive hardware'
            index=0 if stage=='router' else int(stage)+1
            cfg=self.robust['config']
            noise=self.robust['noise'][index,self.offset:self.offset+len(amplitude)]
            noise=F.interpolate(noise,(478,478),mode='bilinear',align_corners=False)
            scale=intensity.detach().mean((-2,-1),keepdim=True).clamp_min(1e-12)
            intensity=(intensity*(1+cfg['ccd_gain_noise']*noise[:,0])+cfg['ccd_read_noise']*scale*noise[:,1]).clamp_min(0)
        return intensity

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
            a=LOW+(HIGH-LOW)*self.fusion_logits[k].sigmoid()
            mixed=(1-a)*(electronic/re)+a*(optical/ro)
            amplitude=re*mixed/rms(mixed).detach()
        f=features(amplitude);p=torch.softmax(self.linear(f),1);energy=energies(intensity,self.detectors)
        self.readout=dict(features=f.detach(),amplitude=amplitude.detach(),ccd=intensity.detach())
        return p,energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)

    def alphas(self):return [float((LOW+(HIGH-LOW)*p.sigmoid()).detach()) for p in self.fusion_logits]

    def contract(self):
        c=super().contract();c.update(id='singleRouter_EG_learnedAlpha06_trainingRobustness_20261018',alpha_bounds=[LOW,HIGH],training_only_noise_and_dropout=True,clean_inference_state_shapes_unchanged=True,test_used_for_selection=True)
        return c
