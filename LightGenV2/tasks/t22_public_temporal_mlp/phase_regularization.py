"""Periodic phase losses and initialization utilities for the retained globals."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def circular_difference(value):
    return torch.atan2(value.sin(), value.cos())


def phase_penalties(phase):
    horizontal = phase[..., :, 1:] - phase[..., :, :-1]
    vertical = phase[..., 1:, :] - phase[..., :-1, :]
    horizontal_curvature = phase[..., :, 2:] - 2*phase[..., :, 1:-1] + phase[..., :, :-2]
    vertical_curvature = phase[..., 2:, :] - 2*phase[..., 1:-1, :] + phase[..., :-2, :]
    slope = ((1-horizontal.cos()).mean() + (1-vertical.cos()).mean()) / 2
    curvature = ((1-horizontal_curvature.cos()).mean()
                 + (1-vertical_curvature.cos()).mean()) / 2
    return {'slope': slope, 'curvature': curvature}


def model_phase_penalties(model):
    values = [phase_penalties(2*math.pi*torch.sigmoid(stage.raw_global_phase))
              for stage in (model.frame_global, model.video_global)]
    return {name: torch.stack([item[name] for item in values]).mean()
            for name in ('slope', 'curvature')}


def gaussian_phase(phase, sigma):
    if sigma <= 0:
        return phase.clone()
    radius = math.ceil(3*sigma)
    coordinates = torch.arange(-radius, radius+1, device=phase.device, dtype=phase.dtype)
    kernel = torch.exp(-coordinates.square()/(2*sigma*sigma))
    kernel = kernel/kernel.sum()
    shape = phase.shape
    phasors = torch.stack((phase.cos(), phase.sin()), dim=-3).reshape(-1,2,*shape[-2:])
    horizontal = kernel.reshape(1,1,1,-1).expand(2,1,1,-1)
    vertical = kernel.reshape(1,1,-1,1).expand(2,1,-1,1)
    smoothed = F.conv2d(F.pad(phasors,(radius,radius,0,0),mode='replicate'), horizontal, groups=2)
    smoothed = F.conv2d(F.pad(smoothed,(0,0,radius,radius),mode='replicate'), vertical, groups=2)
    return torch.atan2(smoothed[:,1],smoothed[:,0]).remainder(2*math.pi).reshape(shape)


def initialize_phase(phase, config, geometry):
    smoothed = gaussian_phase(phase, config.get('sigma_pixels', 0.))
    if config['kind'] == 'gaussian':
        return smoothed
    if config['kind'] != 'lens':
        raise ValueError('Unknown phase initialization')
    focal_length = config['focal_length_m']
    if focal_length <= 0:
        raise ValueError('Focal length must be positive')
    coordinates = (torch.arange(phase.shape[-1], device=phase.device)-55)*geometry.pitch_um*1e-6
    y,x = torch.meshgrid(coordinates,coordinates,indexing='ij')
    lens = -math.pi*(x.square()+y.square())/(geometry.wavelength_nm*1e-9*focal_length)
    offset = torch.atan2(phase.sin().mean((-2,-1),keepdim=True),
                         phase.cos().mean((-2,-1),keepdim=True))
    residual = config.get('residual_scale',0.) * circular_difference(smoothed-offset)
    return (offset+lens+residual).remainder(2*math.pi)


def phase_to_raw(phase):
    probability = (phase.remainder(2*math.pi)/(2*math.pi)).clamp(1e-4,1-1e-4)
    return torch.logit(probability)


def illumination_support(frame_level):
    support = torch.zeros(111,111,dtype=torch.bool)
    if frame_level:
        for y,x in [(12,12),(12,71),(71,12),(71,71)]:
            support[y:y+27,x:x+27] = True
    else:
        # The 42 video/text tokens occupy 42 rows; the remaining 14 rows are padding.
        support[27:69,27:83] = True
    return support


def phase_statistics(stage):
    phase = (2*math.pi*torch.sigmoid(stage.raw_global_phase.detach())).cpu()
    support = illumination_support(stage.frame_level)
    edge_x = support[:,1:] & support[:,:-1]
    edge_y = support[1:,:] & support[:-1,:]
    dx = circular_difference(phase[:,:,1:]-phase[:,:,:-1])[:,edge_x]
    dy = circular_difference(phase[:,1:,:]-phase[:,:-1,:])[:,edge_y]
    active_rms = torch.cat((dx.flatten(),dy.flatten())).square().mean().sqrt()
    penalties = phase_penalties(phase)
    return {'slope_penalty':float(penalties['slope']),
            'curvature_penalty':float(penalties['curvature']),
            'active_neighbor_difference_RMS_rad':float(active_rms),
            'geometric_illumination_fraction':float(support.float().mean())}
