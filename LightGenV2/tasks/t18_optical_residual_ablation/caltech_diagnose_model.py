"""Independent train/val ablations; sealed deployment and legacy graphs unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from caltech_deploy234_model import DeployMoE, bounded_amplitude, SPEC, rms_encode, input_amplitude


class DiagnosticMoE(DeployMoE):
    def __init__(self, variant, depth=4, rho=.3):
        super().__init__(depth, rho)
        self.variant = variant

    def physical_fields(self, amplitude, stage, weights=None):
        if self.variant != 'unquantized' and not (self.variant == 'preserve_global' and stage != 'router' and self.stages[int(stage)] == 'global'):
            return super().physical_fields(amplitude, stage, weights)
        if self.variant == 'preserve_global' and stage != 'router' and self.stages[int(stage)] == 'global':
            assert amplitude.shape[-2:] == (478,478)
            incident = bounded_amplitude(F.pad(amplitude,(20,20,20,20)), SPEC)
        else:
            incident = bounded_amplitude(self.incident(amplitude,weights), SPEC)
        plane = self.plane(stage)
        amp = incident * plane.abs()[None]
        phase = torch.remainder(torch.angle(plane),2*math.pi)/(2*math.pi)
        if self.variant == 'unquantized':
            field = amp.to(torch.complex64)*torch.exp(2j*math.pi*phase)[None]
            return field,amp[:,20:498,20:498],phase[20:498,20:498]
        exact = torch.floor(phase*256).clamp(0,255)/256
        phase = phase+(exact-phase).detach()
        q = torch.round(amp.clamp(0,1)*255)/255
        amp = amp+(q-amp).detach()
        return amp.to(torch.complex64)*torch.exp(2j*math.pi*phase)[None],amp[:,20:498,20:498],exact[20:498,20:498]

    def nonlinear(self, intensity, next_stage):
        size = 478 if self.variant == 'preserve_global' and next_stage == 'global' else 224
        v = intensity.float().clamp_min(0)
        relative = (v/v.mean((-2,-1),keepdim=True).clamp_min(1e-6)).clamp_max(12)
        pooled = F.adaptive_avg_pool2d(torch.log1p(relative)[:,None],(size,size))[:,0]
        axes = (size,size) if self.variant == 'fullfield_ln' else (size,)
        return rms_encode(F.relu(F.layer_norm(pooled,axes,eps=1e-5)))

    def forward(self, amplitude, callback=None, ids=None):
        if self.variant not in ('dense_router','fullfield_ln','preserve_global'):
            return super().forward(amplitude,callback,ids)
        self.last={}
        intensity=self.capture(amplitude,'router',callback=callback,ids=ids)
        energy=torch.einsum('bhw,khw->bk',intensity,self.router_masks)
        centered=energy-energy.mean(1,keepdim=True)
        dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1)
        soft=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)
        if self.variant == 'dense_router':weights=soft
        else:
            selected=dense.topk(2,dim=1).indices
            sparse=dense*torch.zeros_like(dense).scatter(1,selected,1)
            hard=sparse/sparse.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)
            weights=hard.detach()+soft-soft.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights,callback,ids)
            if k<self.depth-1:amplitude=self.nonlinear(intensity,self.stages[k+1])
        energy=torch.einsum('bhw,khw->bk',intensity,self.detectors)
        return (energy+1e-12)/(energy.sum(1,keepdim=True)+20e-12),energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)


class LegacyReference(nn.Module):
    """Original nine-expert bundle, not a one-factor architecture comparison."""
    def __init__(self, cfg, depth=4, rho=.3):
        super().__init__()
        import fast_optical as old
        profile=dict(cfg,depths=[2,4],detector=old.detector(20))
        profile.update(phase_init_raw_uniform=.1,residual_rho=rho)
        self.net=old.build(depth,profile,profile,batched=True,device='cpu')

    def forward(self, amplitude):
        import fast_optical as old
        return old.optical.forward(self.net,amplitude)
    def smoothness(self):
        import fast_optical as old
        return old.optical.b.phase_smoothness(self.net)


def encode(images,theta,variant):
    if variant == 'legacy_reference':
        import fast_optical as old
        return old.optical.encode(images,theta,device=str(images.device))
    if variant == 'rgb_tiles':
        # Retain the old fixed R/G/B/mean mosaic, then use current physical size/scale.
        import fast_optical as old
        a=old.optical.encode(images,theta,device=str(images.device))
        assert a.ndim==4 and a.shape[1:]==(1,100,100)
        a=a[:,0]
        return rms_encode(F.interpolate(a[:,None],(224,224),mode='bilinear',align_corners=False)[:,0])
    return input_amplitude(images,theta)


def build(variant,cfg):
    return LegacyReference(cfg) if variant=='legacy_reference' else DiagnosticMoE(variant)
