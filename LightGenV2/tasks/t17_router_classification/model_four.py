"""Original four-slot PhaseOnly geometry, two OEO stages, strict top-2 routing."""
import torch
from torch import nn
from torch.nn import functional as F
from LightGenV2.demo_check.pure_optical.models import PhaseOnly, encode as encode_rgb

CONTRACT = dict(seed=17,architectures=['dynamic_four','full_d2nn'],input_size=224,
    active_size=478,canvas_size=518,wavelength_m=5.32e-7,pixel_size_m=1.7e-5,
    propagation_m=.1,main_input_power=1.,detector_size=32,router_detector_size=60,
    phase_init_std=0.,oeo_activation='intensity_softsign')

def sparse_top2(weights):
    chosen=weights.topk(2,dim=1).indices
    mask=torch.zeros_like(weights).scatter(1,chosen,1.)
    q=weights*mask
    return q/q.sum(1,keepdim=True).clamp_min(1e-20)

class FourRouterClassification(PhaseOnly):
    def __init__(self,variant,readout='linear'):
        if variant not in ('optical','electronic','d2nn'):raise ValueError(variant)
        if readout not in ('linear','ccd_grid'):raise ValueError(readout)
        super().__init__('full_d2nn' if variant=='d2nn' else 'dynamic_four',dict(CONTRACT))
        self.variant=variant
        self.readout=readout
        self.height=self.width=518
        self.active_height=self.active_width=478
        self.max_experts=4
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(122)
            head=nn.Linear(784,9,bias=False)
            if readout=='linear':self.shared_head=head
            if variant=='electronic':
                self.electronic_router=nn.Sequential(nn.Linear(784,64),nn.GELU(),nn.Linear(64,4))
                nn.init.zeros_(self.electronic_router[-1].weight)
                nn.init.zeros_(self.electronic_router[-1].bias)
                self.register_parameter('router_phase',None)

    def dense_route(self,amplitude):
        if self.variant=='d2nn':return None,None
        if self.variant=='optical':
            dense,capture=super().route(amplitude)
        else:
            x=F.adaptive_avg_pool2d(amplitude[:,None],28).flatten(1)
            x=x/x.mean(1,keepdim=True).clamp_min(1e-20)
            dense=self.electronic_router(x).softmax(1)
            capture=None
        return dense,capture

    def route(self,amplitude):
        dense,capture=self.dense_route(amplitude)
        return (sparse_top2(dense) if dense is not None else None),capture

    def forward(self,amplitude,return_debug=False,return_dense_route=False):
        if amplitude.ndim!=3 or amplitude.shape[-2:]!=(224,224):raise ValueError(amplitude.shape)
        amplitude=amplitude/amplitude.square().sum((-2,-1),keepdim=True).clamp_min(1e-20).sqrt()
        if return_dense_route:
            dense,capture=self.dense_route(amplitude)
            q=sparse_top2(dense) if dense is not None else None
        else:q,capture=self.route(amplitude)
        if self.variant=='d2nn':
            expanded=F.interpolate(amplitude[:,None],(478,478),mode='bilinear',align_corners=False)[:,0]
            expanded=expanded/expanded.square().sum((-2,-1),keepdim=True).clamp_min(1e-20).sqrt()
            first=F.pad(expanded*self.transmission(self.first_phase),(20,)*4)
        else:
            first=torch.zeros((len(amplitude),518,518),dtype=torch.complex64,device=amplitude.device)
            # Avoid sqrt(0)'s infinite derivative on masked experts.
            coefficients=torch.where(q>0,q.clamp_min(1e-20).sqrt(),torch.zeros_like(q))
            for i,(y,x) in enumerate(self.apertures):
                first[:,y:y+224,x:x+224]=(amplitude*coefficients[:,i,None,None]
                                                         *self.transmission(self.first_phase[i]))
        first_ccd=self.propagator(first)
        intermediate=self.oeo(first_ccd)
        mask=F.pad(self.transmission(self.global_phase),(20,)*4,value=1)
        final_ccd=self.propagator(intermediate*mask)
        reencoded=self.oeo(final_ccd)
        intensity=reencoded.abs().square()
        active=intensity[:,20:498,20:498]
        if self.readout=='linear':
            features=F.adaptive_avg_pool2d(active[:,None],28).flatten(1)
            features=features/features.mean(1,keepdim=True).clamp_min(1e-20)
            logits=self.shared_head(features)
            energies=None
        else:
            energies=ccd_grid_energies(active)
            # Relative additive epsilon preserves gradients in dark regions.
            total=energies.sum(1,keepdim=True)
            probabilities=(energies+1e-12*total)/(total*(1+9e-12))
            logits=probabilities.log()
        output=dict(logits=logits,route_power=q,router_capture=capture)
        if return_dense_route:output['dense_route_power']=dense
        if energies is not None:output['class_energies']=energies
        if return_debug:
            output.update(first_ccd=first_ccd.abs().square(),final_ccd=final_ccd.abs().square(),
                          post_oeo_intensity=intensity)
        return output

CCD_EDGES=(0,159,318,478)

def ccd_grid_energies(active):
    """Nine disjoint, full-coverage CCD regions, numbered row-major 0..8."""
    if active.shape[-2:]!=(478,478):raise ValueError(active.shape)
    return torch.stack([active[:,CCD_EDGES[r]:CCD_EDGES[r+1],
                                     CCD_EDGES[c]:CCD_EDGES[c+1]].sum((-2,-1))
                        for r in range(3) for c in range(3)],dim=1)
