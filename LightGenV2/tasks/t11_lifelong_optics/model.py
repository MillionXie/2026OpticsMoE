"""Phase-only coherent optical MoE with fixed preallocated geometry."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .optics import AngularSpectrumPropagator

class OpticalMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        size, gap, border = cfg['expert_size'], cfg['gap'], cfg['border']
        self.size = size
        self.num_experts = int(cfg.get('num_experts', 12))
        if self.num_experts not in (12, 16):
            raise ValueError('num_experts must be 12 or 16')
        rows = self.num_experts // 4
        self.height, self.width = rows*size+(rows-1)*gap+2*border, 4*size+3*gap+2*border
        # Each consecutive group spans the aperture symmetrically; coordinates never change.
        if self.num_experts == 12:
            order = [(0,0),(0,3),(2,0),(2,3),(0,1),(0,2),(2,1),(2,2),(1,0),(1,1),(1,2),(1,3)]
        else:
            order = [(0,0),(0,3),(3,0),(3,3),(0,1),(0,2),(3,1),(3,2),
                     (1,0),(1,3),(2,0),(2,3),(1,1),(1,2),(2,1),(2,2)]
        self.slots = [(border+r*(size+gap),border+c*(size+gap)) for r,c in order]
        self.router_centers = [(y+size//2,x+size//2) for y,x in self.slots]
        if cfg.get('router_layout','slot_centers')=='ring':
            # Sixteen ports need the larger radius so adjacent square CCD windows do not overlap.
            radius=(1.0 if self.num_experts==16 else .8)*size
            groups=self.num_experts//4
            angles=[offset+groups*quadrant for offset in range(groups) for quadrant in range(4)]
            self.router_centers=[(round(self.height/2+radius*math.sin(2*k*math.pi/self.num_experts)),round(self.width/2+radius*math.cos(2*k*math.pi/self.num_experts))) for k in angles]
        elif cfg.get('router_layout','slot_centers')!='slot_centers':
            raise ValueError('Unknown router layout')
        self.num_classes=cfg.get('num_classes',8)
        if self.num_classes<2 or self.num_classes>8: raise ValueError('num_classes must be 2..8')
        candidates=[(round(self.height*y),round(self.width*x)) for y in (.32,.68) for x in (.16,.38,.62,.84)]
        self.class_centers=candidates[:self.num_classes]
        for centers,side in [(self.router_centers,cfg['router_detector_size']),(self.class_centers,cfg['detector_size'])]:
            if side<=0 or side%2: raise ValueError('Detector size must be positive and even')
            for i,(y,x) in enumerate(centers):
                if not (side//2<=y<=self.height-side//2 and side//2<=x<=self.width-side//2): raise ValueError('Detector out of bounds')
                if any(abs(y-yy)<side and abs(x-xx)<side for yy,xx in centers[:i]): raise ValueError('Overlapping detectors')
        self.register_buffer('active_count', torch.tensor(4))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed'])
            self.experts = nn.ParameterList([nn.Parameter(torch.randn(size,size)*.02) for _ in self.slots])
            self.router = nn.Parameter(torch.randn(size,size)*.02)
            self.global_phase = nn.Parameter(torch.randn(self.height-2*border,self.width-2*border)*.02)
        self.propagator = AngularSpectrumPropagator(cfg['wavelength_m'],cfg['pixel_size_m'],(self.height,self.width),cfg['distance_m'])

    def configure(self, stage):
        mapping={'A':(0,False),'warmup':(1,True),'B':(1,False),'warmup_C':(2,True),'C':(2,False),'warmup_D':(3,True),'D':(3,False)}
        if stage not in mapping: raise ValueError(stage)
        group,warmup=mapping[stage]; self.configure_group(group,warmup)

    def configure_group(self, group, warmup=False):
        if group not in range(self.num_experts//4): raise ValueError('expert group is outside the preallocated geometry')
        start=4*group; self.active_count.fill_(start+4)
        for i,p in enumerate(self.experts): p.requires_grad_(start<=i<start+4)
        self.router.requires_grad_(not warmup)
        self.global_phase.requires_grad_(not warmup)

    @staticmethod
    def transmission(p): return torch.exp(2j*torch.pi*torch.sigmoid(p))

    def encode(self, images):
        if images.dtype != torch.uint8 or images.ndim!=4 or images.shape[-1]!=3:
            raise ValueError('Input must be NHWC uint8 RGB; no prior normalization')
        rgb=F.interpolate(images.permute(0,3,1,2).float()/255,(self.size//2,)*2,mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
        a=torch.cat((torch.cat((rgb[:,0],rgb[:,1]),-1),torch.cat((rgb[:,2],torch.zeros_like(rgb[:,2])),-1)),-2)
        power=a.square().sum((-2,-1),keepdim=True)
        if (power<=1e-12).any(): raise ValueError('Zero input power')
        return a/power.sqrt()

    @staticmethod
    def detect(intensity, centers, side):
        half=side//2
        return torch.stack([intensity[:,y-half:y+half,x-half:x+half].sum((-2,-1)) for y,x in centers],1)

    def forward(self, images, mask=None, warmup=False):
        a=self.encode(images); n=int(self.active_count)
        allowed=torch.arange(self.num_experts,device=a.device)<n
        if mask is not None:
            mask=torch.as_tensor(mask,device=a.device,dtype=torch.bool)
            if mask.shape!=(self.num_experts,): raise ValueError(f'Expected {self.num_experts}-slot mask')
            allowed=allowed & mask
        if warmup: allowed=allowed & (torch.arange(self.num_experts,device=a.device)>=n-4)
        if not allowed.any(): raise ValueError('Empty expert mask')
        if warmup:
            q=allowed.float().expand(len(a),-1)/allowed.sum()
        else:
            dy,dx=self.height-self.size,self.width-self.size
            rf=F.pad(a*self.transmission(self.router),(dx//2,dx-dx//2,dy//2,dy-dy//2))
            energy=self.detect(self.propagator(rf).abs().square(),self.router_centers,self.cfg['router_detector_size'])
            weights=(energy+1e-12)*allowed
            q=weights/weights.sum(1,keepdim=True)
        field=torch.zeros((len(a),self.height,self.width),device=a.device,dtype=torch.complex64)
        for i in range(n):
            y,x=self.slots[i]
            # Exclude zero gates before sqrt, avoiding undefined derivative at zero.
            if bool(allowed[i]):
                field[:,y:y+self.size,x:x+self.size]=a*q[:,i,None,None].sqrt()*self.transmission(self.experts[i])
        b=self.cfg['border']
        field=self.propagator(self.propagator(field)*F.pad(self.transmission(self.global_phase),(b,)*4,value=1))
        intensity=field.abs().square()
        e=self.detect(intensity,self.class_centers,self.cfg['detector_size'])+1e-12
        return {'probabilities':e/e.sum(1,keepdim=True),'routes':q,'output_power':intensity.sum((-2,-1))}


def loss(output, labels):
    return F.nll_loss(output['probabilities'].clamp_min(1e-12).log(),labels)
