"""Rich20 inference with explicit, non-interchangeable alpha coordinates.

No SDK or training-controller imports.  Both classes reproduce evaluation only;
the manifest must choose the class, never infer the semantics from a filename.
"""
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from caltech20_alt246_biasfree_model import HybridMoE,rms,features,energies
from caltech_deploy234_model import rms_encode

CONTRACT='rich20_single_router_EG_explicit_alpha_existing_5140_linear_lab_v1'
DIRECT='direct_effective_f64'
BOUNDED='bounded_sigmoid_f32'
BOUNDS={0.:(.8501,.95),.3:(.7001,.95)}


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):result.update(block)
    return result.hexdigest()


def encode_images(images):
    assert images.ndim==4 and images.shape[1:]==(3,150,150)
    x=F.interpolate(images.float()/255,(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    mosaic=torch.cat([torch.cat([x[:,0:1],x[:,1:2]],3),torch.cat([x[:,2:3],x.mean(1,keepdim=True)],3)],2)
    return rms_encode(F.interpolate(mosaic,(224,224),mode='bilinear',align_corners=False)[:,0])


class DirectAlphaHybridMoE(HybridMoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho)
        for parameter in self.fusion_logits:parameter.data=parameter.data.double()

    def alphas(self):return [float(p.float().detach()) for p in self.fusion_logits]

    def effective_alpha(self,k):return self.fusion_logits[k].float()

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
            alpha=self.effective_alpha(k)
            mixed=(1-alpha)*(electronic/re)+alpha*(optical/ro)
            amplitude=re*mixed/rms(mixed).detach()
        f=features(amplitude);probability=torch.softmax(self.linear(f),1);energy=energies(intensity,self.detectors)
        self.readout=dict(features=f.detach(),amplitude=amplitude.detach(),ccd=intensity.detach())
        return probability,energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)

    def contract(self):
        c=super().contract();c.update(id=CONTRACT,alpha_schema=DIRECT,alpha_coordinate='fusion_logits.* store DIRECT effective alpha in float64; float32 fusion, NEVER sigmoid',alpha_bounds=[.4001,.95],alpha_initial=None,readout='last fused amplitude ->pool16->per-sample non-affine LN->existing trained Linear256to20',physical_capture_release=False,comparison_scope='Retained models have different learned-alpha constraints and training histories; not isolated optical-rho causality')
        return c


class BoundedAlphaHybridMoE(DirectAlphaHybridMoE):
    def __init__(self,depth,rho=0.):
        # Preserve the original raw float32 state; no logit conversion on load.
        HybridMoE.__init__(self,depth,rho)

    def effective_alpha(self,k):
        lo,hi=BOUNDS[self.rho]
        return lo+(hi-lo)*self.fusion_logits[k].sigmoid()

    def alphas(self):return [float(self.effective_alpha(k).detach()) for k in range(self.depth)]

    def contract(self):
        c=super().contract();c.update(alpha_schema=BOUNDED,alpha_coordinate='fusion_logits.* store RAW float32 logits; lo+(hi-lo)*sigmoid(raw)',alpha_bounds=list(BOUNDS[self.rho]),comparison_scope='Optical rho plus asymmetric electronic fusion constraints; not pure optical-rho causality')
        return c


def load_checkpoint(path,expected_sha,depth=2,rho=.3,*,alpha_schema):
    assert sha(path)==expected_sha and len(expected_sha)==64
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert 'model' in saved,'Require explicit complete model state, not an EMA/last fallback'
    state=saved['model'];assert isinstance(state,dict)
    assert alpha_schema in (DIRECT,BOUNDED)
    for k in range(depth):
        p=state[f'fusion_logits.{k}']
        assert p.numel()==1 and bool(torch.isfinite(p).all())
        if alpha_schema==DIRECT:
            assert p.dtype==torch.float64 and .4001<float(p.float())<=.95,'Incompatible direct alpha state'
        else:
            assert p.dtype==torch.float32,'Incompatible raw-logit alpha state'
    assert state['linear.weight'].shape==(20,256) and state['linear.bias'].shape==(20,)
    cls=DirectAlphaHybridMoE if alpha_schema==DIRECT else BoundedAlphaHybridMoE
    model=cls(depth,rho).eval();model.load_state_dict(state,strict=True);model.requires_grad_(False)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
    assert sum(p.numel() for p in model.linear.parameters())==5140
    return model,saved
