"""Full-field optical scaling model on the project's 17-um logical grid.

The phase device's 8-um pitch is an equal-physical-size export mapping, not
an extra trainable high-resolution mask. Each feature phase is followed by OEO.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .plan import geometry, router_regions


def unit_power(x):
    return x / x.abs().square().sum((-2, -1), keepdim=True).clamp_min(1e-12).sqrt()


def encode_rgb(rgb, side=224):
    """Channel-wise interpolation prevents mixing across mosaic boundaries."""
    x = F.interpolate(rgb, (side//2, side//2), mode='bilinear', align_corners=False, antialias=True)
    r, g, b = x.unbind(1)
    return unit_power(torch.cat((torch.cat((r,g),-1), torch.cat((b,x.mean(1)),-1)),-2))


class ASM(nn.Module):
    def __init__(self, side, pitch=17e-6, wavelength=532e-9, distance=.1, padding=2):
        super().__init__()
        self.side = side
        self.grid = int(side * padding)
        assert (self.grid-side) % 2 == 0
        self.pad = (self.grid-side)//2
        # Float64 phase construction avoids catastrophic phase error at 2*pi*z/lambda.
        f = torch.fft.fftfreq(self.grid, d=pitch, dtype=torch.float64)
        rad = 1 - wavelength**2 * (f[:,None]**2+f[None,:]**2)
        angle = 2*math.pi*distance/wavelength * (rad.clamp_min(0).sqrt()-1)
        h = torch.polar(torch.ones_like(angle), angle).to(torch.complex64)
        h[rad < 0] = 0
        self.register_buffer('transfer', h, persistent=False)

    def forward(self, field):
        x = F.pad(field, (self.pad,)*4)
        y = torch.fft.ifft2(torch.fft.fft2(x)*self.transfer)
        return y[...,self.pad:self.pad+self.side,self.pad:self.pad+self.side]


class ScalingOptics(nn.Module):
    def __init__(self, cfg, n=4, k=4, arch='moe_oeo', classes=8, layers=6, padding=2,
                 checkpointing=True):
        super().__init__()
        import copy
        self.cfg=copy.deepcopy(cfg)
        self.cfg['model']['feature_layers_provisional']=layers
        self.geo=geometry(self.cfg,n)
        self.n,self.k,self.arch,self.classes,self.layers=n,k,arch,classes,layers
        assert 1 <= k <= n
        self.g=self.geo['active_side_px'];self.c=self.geo['canvas_side_px'];self.margin=(self.c-self.g)//2
        self.e=cfg['geometry']['expert_side_px'];self.pitch=self.e+cfg['geometry']['gap_px']
        self.checkpointing=checkpointing
        self.prop=ASM(self.c,pitch=cfg['geometry']['pixel_pitch_um']*1e-6,padding=padding)
        self.is_moe=arch=='moe_oeo'
        if self.is_moe:
            self.router_side=self.geo['router_side_px']
            self.router_phase=nn.Parameter(torch.randn(self.router_side,self.router_side)*.02)
            self.router_prop=ASM(self.router_side,pitch=cfg['geometry']['pixel_pitch_um']*1e-6,padding=padding)
            self.boxes=router_regions(cfg,n)
            self.expert_phases=nn.ParameterList([nn.Parameter(torch.randn(n,self.e,self.e)*.02) for _ in range(layers//2)])
            self.global_phases=nn.ParameterList([nn.Parameter(torch.randn(self.g,self.g)*.02) for _ in range(layers//2)])
        else:
            assert arch in {'d2nn_total_parameter','d2nn_same_aperture','d2nn_expert_global'}
            self.side=self.geo['d2nn_parameter_side_px'] if arch=='d2nn_total_parameter' else self.g
            if arch=='d2nn_expert_global':
                target=self.geo['expert_phase_parameters']+self.geo['global_phase_parameters']
                ideal=math.sqrt(target/layers)
                sides={2*math.floor(ideal/2),2*math.ceil(ideal/2)}
                self.side=min((s for s in sides if 0<s<=self.g),key=lambda s:abs(layers*s*s-target))
                self.geo['expert_global_matching_target']=target
                self.geo['expert_global_actual_parameters']=layers*self.side*self.side
                assert abs(layers*self.side*self.side-target)/target<.005
            self.dense_phases=nn.ParameterList([nn.Parameter(torch.randn(self.side,self.side)*.02) for _ in range(layers)])
        grid=math.ceil(math.sqrt(classes));extent=(grid-1)*48+32;start=(self.c-extent)//2
        self.detectors=[(start+(i//grid)*48,start+(i//grid)*48+32,
                         start+(i%grid)*48,start+(i%grid)*48+32) for i in range(classes)]

    def route(self, source, dense=False, ste=True):
        field=self.router_prop(source.to(torch.complex64)*torch.exp(1j*self.router_phase))
        intensity=field.abs().square()
        score=torch.stack([intensity[:,a:b,c:d].sum((-2,-1)) for a,b,c,d in self.boxes],1)
        prob=(score/intensity.sum((-2,-1))[:,None].clamp_min(1e-12)/.05).softmax(1)
        mask=torch.zeros_like(prob).scatter_(1,prob.topk(self.k,dim=1).indices,1)
        if dense:mask=torch.ones_like(prob)
        selected=prob*mask
        hard=(selected/selected.sum(1,keepdim=True).clamp_min(1e-12)).clamp_min(0).sqrt()
        # Avoid sqrt(0) derivatives: hard path is detached for the STE.
        soft=prob.clamp_min(1e-12).sqrt()
        amp=soft+(hard-soft).detach() if self.training and ste and not dense else (soft if dense else hard)
        return amp,prob,mask

    def assemble(self, tiles, fill=0):
        """Differentiable tiling without N full-canvas allocations."""
        rows=[];grid=math.isqrt(self.n);gap=self.pitch-self.e
        for y in range(grid):
            parts=[]
            for x in range(grid):
                t=tiles[:,y*grid+x]
                parts.append(F.pad(t,(0,gap if x<grid-1 else 0,0,0),value=fill))
            row=torch.cat(parts,-1)
            rows.append(F.pad(row,(0,0,0,gap if y<grid-1 else 0),value=fill))
        return F.pad(torch.cat(rows,-2),(self.margin,)*4,value=fill)

    def oeo(self, field):
        m=self.margin
        i=field[:,m:m+self.g,m:m+self.g].abs().square()
        i=i/i.mean((-2,-1),keepdim=True).clamp_min(1e-12)
        t=F.layer_norm(i,(self.g,self.g),eps=1e-6).relu()
        a=unit_power(t/(1+t))
        return F.pad(a,(m,)*4).to(torch.complex64)

    def phase_step(self, field, phase, expert):
        if expert:
            full=self.assemble(phase[None],fill=0)[0]
        else:
            pad=(self.c-phase.shape[-1])//2
            full=F.pad(phase,(pad,)*4)
        return self.oeo(self.prop(field*torch.exp(1j*full)))

    def forward(self,rgb,dense=False):
        if self.is_moe:
            expert_source=encode_rgb(rgb)
            router_source=encode_rgb(rgb,self.router_side)
            amp,prob,mask=self.route(router_source,dense=dense)
            field=self.assemble(expert_source[:,None]*amp[:,:,None,None]).to(torch.complex64)
            phases=[(p,e) for pair in zip(self.expert_phases,self.global_phases) for p,e in zip(pair,(True,False))]
        else:
            source=encode_rgb(rgb,self.side)
            field=F.pad(source,((self.c-self.side)//2,)*4).to(torch.complex64)
            prob=mask=None
            phases=[(p,False) for p in self.dense_phases]
        for phase,expert in phases:
            if self.training and self.checkpointing:
                field=checkpoint(self.phase_step,field,phase,expert,use_reentrant=False)
            else:field=self.phase_step(field,phase,expert)
        intensity=self.prop(field).abs().square()
        energy=torch.stack([intensity[:,a:b,c:d].sum((-2,-1)) for a,b,c,d in self.detectors],1)
        probs=(energy+1e-12)/(energy.sum(1,keepdim=True)+self.classes*1e-12)
        capture=energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)
        return dict(probabilities=probs,capture=capture,route_probabilities=prob,route_mask=mask)

    def regularizer(self):
        terms=[]
        for p in self.parameters():
            terms.append(((1-(p[...,1:,:]-p[...,:-1,:]).cos()).mean()+
                          (1-(p[...,:,1:]-p[...,:,:-1]).cos()).mean())/2)
        return torch.stack(terms).mean()
