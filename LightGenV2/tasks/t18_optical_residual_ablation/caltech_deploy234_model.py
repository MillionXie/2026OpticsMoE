"""T18 deployable 2/3/4 main stages; audited ABO geometry, no device access."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.optics import (
    Propagation, bounded_amplitude, CANVAS, ACTIVE, EXPERT, APERTURES)

SPEC = dict(kind='tanh', scale=.5)


def quantize_ste(value, levels=255):
    exact = torch.round(value.clamp(0, 1)*levels)/levels
    return value + (exact-value).detach()


def rms_encode(value):
    value = value.clamp_min(0)
    return value * (.25/value.square().mean((-2,-1), keepdim=True).sqrt().clamp_min(1e-6))


def input_amplitude(images, theta=None):
    value = images.float()/255
    if theta is not None:
        grid = F.affine_grid(theta, value.shape, align_corners=False)
        value = F.grid_sample(value, grid, align_corners=False, padding_mode='zeros')
    value = (value[:,0]*.299 + value[:,1]*.587 + value[:,2]*.114).clamp(0,1)
    value = F.interpolate(value[:,None], (224,224), mode='bilinear', align_corners=False)[:,0]
    return rms_encode(value.sqrt())


def ccd_nonlinearity(intensity):
    # This is the ABO computational readout, applied identically to raw CCD.
    value = intensity.float().clamp_min(0)
    relative = (value/value.mean((-2,-1),keepdim=True).clamp_min(1e-6)).clamp_max(12)
    pooled = F.adaptive_avg_pool2d(torch.log1p(relative)[:,None],(224,224))[:,0]
    return rms_encode(F.relu(F.layer_norm(pooled,(224,),eps=1e-5)))


def masks20():
    masks = torch.zeros(20,478,478)
    # Equal support, non-overlapping 64x64 windows, fixed before any training.
    for k in range(20):
        y, x = 63+(k//5)*96, 15+(k%5)*96
        masks[k,y:y+64,x:x+64] = 1
    assert masks.sum((-2,-1)).eq(4096).all() and masks.sum(0).max()==1
    return masks


class DeployMoE(nn.Module):
    def __init__(self, depth, rho):
        super().__init__()
        assert depth in (2,3,4) and rho in (0.,.3)
        self.depth, self.rho = depth, rho
        self.stages = ['expert' if k%2==0 else 'global' for k in range(depth)]
        self.raw_router = nn.Parameter(torch.empty(224,224).uniform_(-.1,.1))
        self.phase_sets = nn.ParameterList()
        self.stage_slices=[]
        for kind in self.stages:
            start=len(self.phase_sets)
            shapes=[(224,224)]*4 if kind=='expert' else [(478,478)]
            for shape in shapes:self.phase_sets.append(nn.Parameter(torch.empty(shape).uniform_(-.1,.1)))
            self.stage_slices.append((start,len(self.phase_sets)))
        self.propagator = Propagation()
        self.register_buffer('detectors',masks20(),persistent=False)
        router_masks=torch.zeros(4,478,478)
        for k,(y,x) in enumerate([(y,x) for y in (164,255) for x in (164,255)]):router_masks[k,y:y+59,x:x+59]=1
        self.register_buffer('router_masks',router_masks,persistent=False)
        self.last = {}

    def plane(self, stage):
        plane = torch.ones((518,518),dtype=torch.complex64,device=self.raw_router.device)
        if stage=='router':
            plane[147:371,147:371]=torch.exp(2j*math.pi*torch.sigmoid(self.raw_router))
        else:
            index=int(stage);lo,hi=self.stage_slices[index]
            locations=APERTURES if self.stages[index]=='expert' else [(20,20)]
            for raw,(y,x) in zip(self.phase_sets[lo:hi],locations):
                modulation=(1-self.rho)*torch.exp(2j*math.pi*torch.sigmoid(raw))+self.rho
                plane[y:y+raw.shape[0],x:x+raw.shape[1]]=modulation
        return plane

    def incident(self, amplitude, weights=None):
        if weights is None:return F.pad(amplitude,(147,147,147,147))
        field=amplitude.new_zeros(len(amplitude),518,518)
        for k,(y,x) in enumerate(APERTURES):field[:,y:y+224,x:x+224]=amplitude*weights[:,k,None,None]
        return field

    def physical_fields(self, amplitude, stage, weights=None):
        incident=bounded_amplitude(self.incident(amplitude,weights),SPEC)
        plane=self.plane(stage)
        # Exact coherent rho field represented by amplitude AND phase SLMs.
        encoded_amplitude=incident*plane.abs()[None]
        phase=torch.remainder(torch.angle(plane),2*math.pi)/(2*math.pi)
        phase_exact=torch.floor(phase*256).clamp(0,255)/256
        phase_ste=phase+(phase_exact-phase).detach()
        qamp=quantize_ste(encoded_amplitude)
        field=qamp.to(torch.complex64)*torch.exp(2j*math.pi*phase_ste)[None]
        return field,qamp[:,20:498,20:498],phase_exact[20:498,20:498]

    def capture(self, amplitude, stage, weights=None, callback=None, ids=None):
        field,amp,phase=self.physical_fields(amplitude,stage,weights)
        if callback is None:intensity=self.propagator(field).abs().square()[:,20:498,20:498]
        else:
            if self.raw_router.device.type!='cpu':raise ValueError('Hardware replay is CPU only')
            intensity=callback(stage,amp.detach().numpy(),(phase.detach().numpy()*256).astype(np.uint8),ids)
            intensity=torch.as_tensor(intensity,dtype=torch.float32,device=field.device)
        if intensity.shape!=(len(amplitude),478,478) or not torch.isfinite(intensity).all() or (intensity<0).any():raise ValueError('CCD must be raw linear non-negative [B,478,478]')
        self.last[stage]=dict(amplitude=amp.detach(),phase_gray=(phase.detach()*256).to(torch.uint8),intensity=intensity.detach())
        return intensity

    def forward(self, amplitude, callback=None, ids=None):
        self.last={}
        intensity=self.capture(amplitude,'router',callback=callback,ids=ids)
        energy=torch.einsum('bhw,khw->bk',intensity,self.router_masks)
        centered=energy-energy.mean(1,keepdim=True)
        logits=centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()
        dense=torch.softmax(logits/2,1)
        chosen=dense.topk(2,dim=1).indices
        sparse=dense*torch.zeros_like(dense).scatter(1,chosen,1)
        hard=sparse/sparse.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)
        soft=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)
        weights=hard.detach()+soft-soft.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights,callback,ids)
            if k<self.depth-1:amplitude=ccd_nonlinearity(intensity)
        energy=torch.einsum('bhw,khw->bk',intensity,self.detectors)
        probability=(energy+1e-12)/(energy.sum(1,keepdim=True)+20e-12)
        capture=energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)
        return probability,capture

    def smoothness(self):
        terms=[]
        for raw in self.parameters():
            phase=2*math.pi*torch.sigmoid(raw)
            terms.extend([(phase[1:]-phase[:-1]).square().mean(),(phase[:,1:]-phase[:,:-1]).square().mean()])
        return torch.stack(terms).mean()


def deployment_contract(depth,rho):
    return dict(main_depth=depth,router_count=1,total_captures=depth+1,
        stages=['expert' if k%2==0 else 'global' for k in range(depth)],
        geometry=dict(wavelength_m=532e-9,distance_m=.1,logical_pitch_m=17e-6,
            numerical_canvas=518,active=478,expert=224,expert_array='2x2',gap=30,
            active_expert_origins=[[0,0],[0,254],[254,0],[254,254]],native_active=1016,
            amplitude_native=[1080,1920],phase_native=[1200,1920]),
        amplitude='RMS .25 -> tanh(abs/.5), zeros remain zero, round255; NO extra peak normalization',
        phase='floor256 modulo2pi; calibrated phase LUT required',
        rho=rho,residual='((1-rho)*exp(i*phi)+rho); expert/global only, router rho0',
        residual_export='joint amplitude*abs(modulation) and angle(modulation), not phase-only',
        nonlinearity='raw CCD -> relative mean clamp12 -> log1p -> pool224 -> row LayerNorm eps1e-5 -> ReLU -> RMS.25',
        readout='fixed20 equal64x64 intensity windows, normalized detector energy; no learned electronic classifier',
        quantization='8-bit amplitude AND phase in training/validation/export, straight-through gradients',
        physical_status='offline mathematical/export parity; fresh camera ROI/orientation/exposure/gain/LUT and physical rho calibration still require bench acceptance; no device connection')