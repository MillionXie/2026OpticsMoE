"""Composite RGB/dense4/preserve478 repair; sealed graphs are imported unchanged."""
import torch
from caltech_diagnose_model import DiagnosticMoE, encode

class CorrectedMoE(DiagnosticMoE):
    def __init__(self,depth,rho):
        super().__init__('preserve_global',depth,rho)
        self.routing_weights=None

    def forward(self,amplitude,callback=None,ids=None):
        self.last={}
        intensity=self.capture(amplitude,'router',callback=callback,ids=ids)
        energy=torch.einsum('bhw,khw->bk',intensity,self.router_masks)
        centered=energy-energy.mean(1,keepdim=True)
        dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1)
        weights=dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)
        self.routing_weights=weights.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights,callback,ids)
            if k<self.depth-1:amplitude=self.nonlinear(intensity,self.stages[k+1])
        energy=torch.einsum('bhw,khw->bk',intensity,self.detectors)
        return (energy+1e-12)/(energy.sum(1,keepdim=True)+20e-12),energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)

def input_amplitude(images,theta=None):return encode(images,theta,'rgb_tiles')


def deployment_contract(depth,rho):
    from caltech_deploy234_model import deployment_contract as base
    contract=base(depth,rho)
    contract.update(graph='RGB fixed input + continuous all4 routing + preserve478 expert-to-global',input_encoding='Fixed R/G/B/mean mosaic ->224 ->RMS.25; no learned input head',routing='Optical router energy -> centered RMS logits /temperature2 ->softmax ->L2 amplitude weights; all4 enabled, input dependent',global_input='Expert CCD478 retained when next is global; no224 pool/refanout before global; expert transitions224',nonlinearity='relative mean clamp12/log1p ->pool478 for global or224 for expert ->rowLN/ReLU/RMS.25',static_export='Per-stage angle(M) and abs(M); dynamic amplitude uses actual478 global or224 expert; old224-only exporter incompatible',physical_status='Offline contract only; native amplitude1920x1080/phase1920x1200, fresh LUT/ROI/exposure/orientation calibration required')
    return contract
