"""Phase-only coherent optical MoE; fixed 3x4 geometry across all stages."""
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
        self.height, self.width = 3*size+2*gap+2*border, 4*size+3*gap+2*border
        # Activate balanced groups of four corners first; coordinates never change.
        order = [(0,0),(0,3),(2,0),(2,3),(0,1),(0,2),(2,1),(2,2),(1,0),(1,1),(1,2),(1,3)]
        self.slots = [(border+r*(size+gap),border+c*(size+gap)) for r,c in order]
        self.router_centers = [(y+size//2,x+size//2) for y,x in self.slots]
        if cfg.get('router_layout','slot_centers')=='ring':
            radius=.8*size
            angles=[0,3,6,9,1,4,7,10,2,5,8,11]
            self.router_centers=[(round(self.height/2+radius*math.sin(k*math.pi/6)),round(self.width/2+radius*math.cos(k*math.pi/6))) for k in angles]
        elif cfg.get('router_layout','slot_centers')!='slot_centers':
            raise ValueError('Unknown router layout')
        self.class_centers = [(round(self.height*y),round(self.width*x)) for y in (.32,.68) for x in (.16,.38,.62,.84)]
        self.register_buffer('active_count', torch.tensor(4))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed'])
            self.experts = nn.ParameterList([nn.Parameter(torch.randn(size,size)*.02) for _ in self.slots])
            self.router = nn.Parameter(torch.randn(size,size)*.02)
            self.global_phase = nn.Parameter(torch.randn(self.height-2*border,self.width-2*border)*.02)
        self.propagator = AngularSpectrumPropagator(cfg['wavelength_m'],cfg['pixel_size_m'],(self.height,self.width),cfg['distance_m'])

    def configure(self, stage):
        if stage not in ('A','warmup','B'): raise ValueError(stage)
        self.active_count.fill_(4 if stage=='A' else 8)
        for i,p in enumerate(self.experts): p.requires_grad_(i<4 if stage=='A' else 4<=i<8)
        self.router.requires_grad_(stage!='warmup')
        self.global_phase.requires_grad_(stage!='warmup')

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
        allowed=torch.arange(12,device=a.device)<n
        if mask is not None:
            mask=torch.as_tensor(mask,device=a.device,dtype=torch.bool)
            if mask.shape!=(12,): raise ValueError('Expected 12-slot mask')
            allowed=allowed & mask
        if warmup: allowed=allowed & (torch.arange(12,device=a.device)>=4)
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
