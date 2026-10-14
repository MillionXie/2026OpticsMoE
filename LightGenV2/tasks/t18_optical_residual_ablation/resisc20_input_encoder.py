"""Fixed input-detail control; no learned encoder or optical-network changes."""
import torch
from torch.nn import functional as F
from caltech_deploy234_model import rms_encode
from caltech_repair234_model import deployment_contract as physical_contract
import fast_optical as old

def encode(images,theta=None,tile=50):
    assert tile in (50,100) and images.ndim==4 and images.shape[1:]==(3,150,150)
    x=images.float()/255
    x=F.interpolate(x,(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    if theta is not None:x=old.optical.b.augment(x,theta)
    if tile==50:x=F.interpolate(x,(50,50),mode='bilinear',align_corners=False,antialias=True)
    mosaic=torch.cat([torch.cat([x[:,0:1],x[:,1:2]],3),torch.cat([x[:,2:3],x.mean(1,keepdim=True)],3)],2)
    return rms_encode(F.interpolate(mosaic,(224,224),mode='bilinear',align_corners=False)[:,0])

def contract(depth,rho,tile):
    v=physical_contract(depth,rho)
    v.update(input_encoding=f'Fixed RGB150 ->bicubicAA100/clamp/affine100 ->channel tile{tile} ->R/G/B/mean mosaic{2*tile} ->bilinear224/RMS.25; no learned encoder',
             input_channel_tile=tile,input_network_aperture_unchanged=224)
    return v
