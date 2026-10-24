"""Proposed native-pixel OEO adaptation, NOT equivalent to the ZIP's coherent graph.

No SDK is opened here. Only an admitted hardware session may supply measured
intensities. Original simulation results must never label this new graph.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .reference.optics import AngularSpectrumPropagator


class NativeOEOOptics(nn.Module):
    def __init__(self, architecture, *, expert_grid=2, expert_size=476, gap=64,
                 border=43, routing_temperature=1., readout='detectors'):
        super().__init__()
        if architecture not in ('moe', 'd2nn') or readout not in ('detectors', 'linear'):
            raise ValueError('Unknown architecture/readout')
        self.architecture, self.readout = architecture, readout
        self.expert_grid, self.expert_size, self.border = expert_grid, expert_size, border
        self.active = expert_grid * expert_size + (expert_grid - 1) * gap
        if self.active != 1016 or expert_grid not in (2, 4):
            raise ValueError('This proposal is restricted to the existing 1016-pixel aperture')
        self.canvas = self.active + 2 * border
        self.temperature = routing_temperature
        if self.temperature <= 0:
            raise ValueError('Nonpositive routing temperature')
        if expert_grid == 4:
            positions = [(r0+dr, c0+dc) for r0,c0 in ((0,0),(0,2),(2,0),(2,2))
                         for dr,dc in ((0,0),(0,1),(1,0),(1,1))]
            order = (3,6,9,12,1,4,11,14,2,7,8,13,0,5,10,15)
        else:
            positions = [(0,0),(0,1),(1,0),(1,1)]
            order = tuple(range(4))
        self.slots = [(border+r*(expert_size+gap), border+c*(expert_size+gap)) for r,c in positions]
        self.register_buffer('active_indices', torch.tensor(order))
        self.register_buffer('active_count', torch.tensor(4))
        if architecture == 'moe':
            self.router_phase = nn.Parameter(torch.zeros(expert_size, expert_size))
            self.first_phase = nn.ParameterList([nn.Parameter(torch.zeros(expert_size, expert_size)) for _ in positions])
        else:
            self.register_parameter('router_phase', None)
            self.first_phase = nn.Parameter(torch.zeros(self.active, self.active))
        self.global_phase = nn.Parameter(torch.zeros(self.active, self.active))
        self.shared_head = nn.Linear(784,10,bias=False) if readout == 'linear' else None
        self.propagator = AngularSpectrumPropagator(532e-9, 8e-6, self.canvas, .1)
        c = self.canvas // 2
        if expert_grid == 2:
            self.router_centers = [(c+y,c+x) for y in (-270,270) for x in (-270,270)]
            self.router_side = 128
        else:
            offsets = [-148,-49,49,148]
            self.router_centers = [(c+offsets[r],c+offsets[col]) for r,col in positions]
            self.router_side = 82
        scale = self.active/478
        self.class_centers = [(round(c+(y-259)*scale),round(c+(x-259)*scale))
                              for y in (160,358) for x in (80,170,259,348,438)]
        self.detector_side = 68

    @staticmethod
    def normalize(x):
        return x / x.square().sum((-2,-1),keepdim=True).clamp_min(1e-20).sqrt()

    @staticmethod
    def transmission(raw):
        return torch.exp(2j*torch.pi*torch.sigmoid(raw))

    @staticmethod
    def detect(intensity, centers, side):
        h = side//2
        return torch.stack([intensity[:,y-h:y+h,x-h:x+h].sum((-2,-1)) for y,x in centers],1)

    def configure_stage(self, stage):
        if stage not in range(1,5) or (self.expert_grid==2 and stage!=1):
            raise ValueError('Invalid stage for fixed geometry')
        self.active_count.fill_(4*stage)
        if self.architecture=='moe':
            new = set(self.active_indices[4*(stage-1):4*stage].tolist())
            for index, phase in enumerate(self.first_phase):
                phase.requires_grad_(index in new)

    def camera_intensity(self, simulated, measured, name):
        if name not in measured:
            return simulated.abs().square()
        value = measured[name]
        if value.shape != (len(simulated),self.active,self.active):
            raise ValueError('Camera must supply the admitted native active aperture')
        if not torch.isfinite(value).all() or (value<0).any():
            raise ValueError('Invalid camera intensity')
        return F.pad(value,(self.border,)*4)

    def forward(self, amplitude, *, measured=None):
        if amplitude.ndim!=3 or amplitude.shape[-2:]!=(224,224):
            raise ValueError('Preserve original 224-square encoded input')
        measured = {} if measured is None else measured
        if set(measured)-{'router','first','final'}:
            raise ValueError('Unknown measurement boundary')
        if self.architecture=='d2nn' and 'router' in measured:
            raise ValueError('D2NN has no router measurement')
        size = self.expert_size if self.architecture=='moe' else self.active
        amp = self.normalize(F.interpolate(amplitude[:,None],(size,size),mode='bilinear',align_corners=False)[:,0])
        route = None
        if self.architecture=='moe':
            pad = (self.canvas-size)//2
            routed = F.pad(amp*self.transmission(self.router_phase),(pad,)*4)
            router = self.camera_intensity(self.propagator(routed),measured,'router')
            energy = self.detect(router,self.router_centers,self.router_side)
            indices = self.active_indices[:int(self.active_count)]
            scores = (energy.index_select(1,indices)+1e-12).pow(1/self.temperature)
            weights = scores/scores.sum(1,keepdim=True)
            route = torch.zeros_like(energy).scatter(1,indices[None].expand(len(amp),-1),weights)
            field = torch.zeros((len(amp),self.canvas,self.canvas),dtype=torch.complex64,device=amp.device)
            for index in indices.tolist():
                y,x = self.slots[index]
                field[:,y:y+size,x:x+size] = amp*route[:,index,None,None].sqrt()*self.transmission(self.first_phase[index])
        else:
            field = F.pad(amp*self.transmission(self.first_phase),(self.border,)*4)
            router = None
        first = self.camera_intensity(self.propagator(field),measured,'first')
        b = self.border
        cropped = first[:,b:-b,b:-b]
        intensity = cropped/cropped.mean((-2,-1),keepdim=True).clamp_min(1e-20)
        reload = self.normalize(F.softsign(intensity))
        field = F.pad(reload*self.transmission(self.global_phase),(b,)*4)
        final = self.camera_intensity(self.propagator(field),measured,'final')
        ccd = final[:,b:-b,b:-b]
        if self.shared_head is not None:
            features = F.adaptive_avg_pool2d(ccd[:,None],(28,28))[:,0].flatten(1)
            features = features/features.mean(1,keepdim=True).clamp_min(1e-20)
            logits = self.shared_head(features)
            probabilities = logits.softmax(1)
        else:
            energy = self.detect(final,self.class_centers,self.detector_side)
            probabilities = (energy+1e-12)/(energy.sum(1,keepdim=True)+1e-11)
            logits = probabilities.log()
        return dict(logits=logits,probabilities=probabilities,route_power=route,
                    router=router,first=first,final=final)

    @torch.no_grad()
    def initialize_from_reference(self, state):
        """Explicit warm start by raw-phase interpolation; never a parity claim."""
        def copy(target, source):
            target.copy_(F.interpolate(source[None,None].float(),target.shape,mode='bilinear',align_corners=False)[0,0])
        if self.architecture=='moe':
            copy(self.router_phase,state['router_phase'])
            for i,target in enumerate(self.first_phase):
                source = state[f'first_phase.{i}'] if f'first_phase.{i}' in state else state['first_phase'][i]
                copy(target,source)
        else:
            copy(self.first_phase,state['first_phase'])
        copy(self.global_phase,state['global_phase'])
        if self.shared_head is not None:
            self.shared_head.load_state_dict({'weight':state['shared_head.weight']},strict=True)
        if 'active_count' in state:
            self.active_count.copy_(state['active_count'])
