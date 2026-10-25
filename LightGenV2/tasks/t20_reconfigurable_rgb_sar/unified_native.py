"""T20 single-sensor classification using the immutable T21 optical graph.

T21 EuroSAT fuses RGB+SAR for ONE location. Here each sensor is evaluated
separately; missing sensor tiles are zero. Neither class labels nor sample IDs
enter the encoding/router. The existing CLEVR-pretrained visual feature strip
is kept for RGB; SAR uses the SAME raw VV/VH averaging as T21, no added CNN.
"""
import math
import torch
from torch.nn import functional as F
from LightGenV2.tasks.t21_continual_learning_hardware.model import build_model
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.frozen_vision import visual_tiles
from LightGenV2.tasks.t13_four_modal_lifelong.model import normalize_power

FRONTEND_SHA256='d20273e1828fec4e5a7a5545fd892a722f3909f7205a791e0e682b437526aa11'
CONTRACT='T20_single_sensor_T21_native8_A_graph_v1'


def encode_single(images,domains,feature_rows):
    if images.dtype!=torch.uint8 or images.ndim!=4 or images.shape[1:]!=(56,56,3):raise ValueError('Original uint8 NHWC56 input required')
    if domains.shape!=(len(images),) or feature_rows.shape!=(len(images),128) or not torch.all((domains==0)|(domains==1)):raise ValueError('Declared observed sensor, not label, required')
    device=images.device;out=torch.zeros((len(images),224,224),device=device,dtype=torch.float32);rgb=domains==0;sar=~rgb
    if rgb.any():
        tiles=visual_tiles(images[rgb].cpu().numpy(),feature_rows[rgb].float(),device)
        out[rgb,:112,:112]=tiles[:,0];out[rgb,:112,112:]=tiles[:,1];out[rgb,112:,:112]=tiles[:,2]
    if sar.any():
        raw=images[sar].permute(0,3,1,2).float()/255.;raw=F.interpolate(raw,(112,112),mode='bilinear',align_corners=False)
        out[sar,112:,112:]=normalize_power((raw[:,0]+raw[:,1])/2,1.)
    if not torch.isfinite(out).all() or torch.any(out.square().sum((-2,-1))<=1e-12):raise ValueError('Finite nonzero sensor input required')
    return normalize_power(out,1.)


def build_unified(architecture,seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);model=build_model(architecture);model.configure_stage(1)
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                if 'phase' in name:parameter.normal_(0,.02)
            torch.manual_seed(seed+909)
            torch.nn.init.xavier_uniform_(model.shared_head.weight,gain=.1)
    # Static A capacity: same four center-out active slots as T21 A. Not the
    # ZIP's two-to-four phase expansion and not a new16-active-expert trial.
    if architecture=='moe':
        active=set(model.active_indices[:4].tolist())
        for i,p in enumerate(model.first_phase):p.requires_grad_(i in active)
    return model
