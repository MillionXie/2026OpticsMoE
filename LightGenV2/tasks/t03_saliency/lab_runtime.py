"""Pinned SALICON body replay at its three physical detector boundaries.

Cache = exact frozen Qwen patch/position output BEFORE replacement block 0.
No learned optical/electronic computation is replaced by a cached prediction.
"""
from types import SimpleNamespace, MethodType
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import read, write, sha
from .modeling import (build_dense_core, MeanOnlyCCDNormalizer, architecture_label,
                       OpticalDetectorTopKRouter, SaliencyDensityDecoder,
                       restore_qwen_block_major_spatial)

STAGES = ('vision_router', 'vision_expert', 'vision_global')
CHECKPOINT_SHA = '036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe'
REFERENCE_CC = 0.8624925081777596
TEST_IDS_SHA = '625dec6bc15b2d737d39bc252cfa0c354de217fec0266dcda568913f4a3496d0'


class CachedStudent(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.core = build_dense_core(settings, 'cpu')
        branch = self.core.optical_branch
        branch.core.router = OpticalDetectorTopKRouter(branch.core.geometry, settings)
        if settings.ccd_normalization != 'mean_only':
            raise ValueError('Pinned release requires the original mean-only normalization')
        branch.ccd_normalizer = MeanOnlyCCDNormalizer(settings.active_size)
        self.head = SaliencyDensityDecoder(input_dim=settings.electronic_width,
                                          output_size=settings.image_size)

    def forward(self, batch):
        device = next(self.parameters()).device
        grid = batch['grid'].to(device)
        tokens = batch['tokens'].to(device)
        shapes = [tuple(map(int, row)) for row in grid.tolist()]
        lengths = [t*h*w for t, h, w in shapes]
        groups = list(tokens.split(lengths))
        self.core.forward_groups(groups, shapes)
        spatial = restore_qwen_block_major_spatial(torch.cat(self.core.last_latent_groups), grid)
        return self.head(spatial)


def load_model(project, device='cpu'):
    project = Path(project)
    checkpoint = project/'weights/best_checkpoint.pt'
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Wrong SALICON 0.8625 checkpoint')
    settings = SimpleNamespace(**read(project/'settings.json'))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['architecture'] != architecture_label(settings):
        raise ValueError('SALICON settings/architecture mismatch')
    model = CachedStudent(settings)
    model.core.load_state_dict(payload['core'], strict=True)
    model.head.load_state_dict(payload['saliency_head'], strict=True)
    model.core.set_phase_dropout_active(False)
    return model.eval().to(device)


def phase_planes(model):
    branch = model.core.optical_branch
    g = branch.core.geometry
    a = g.active_aperture
    field = next(model.parameters()).new_ones(1, g.canvas_size, g.canvas_size)
    with torch.no_grad():
        planes = {'vision_router': branch.core.router.active_phase()}
        for stage, modulation in [('vision_expert', branch._expert_phase_modulation(field)),
                                  ('vision_global', branch._global_phase_modulation(field))]:
            planes[stage] = torch.remainder(torch.angle(modulation[0, a.y0:a.y1, a.x0:a.x1]), 2*math.pi)
    return {k:v.detach().cpu().numpy() for k,v in planes.items()}


class StopAtPlane(Exception):
    pass


class OpticalBoundary:
    def __init__(self, model, measured=None, stop_before=None):
        self.model, self.measured, self.stop_before = model, measured or {}, stop_before
        if tuple(self.measured) != STAGES[:len(self.measured)]:
            raise ValueError('Measured frames must be a contiguous three-stage prefix')
        self.amplitudes, self.detectors, self.originals = {}, {}, []
        self.index = 0
        self.planes = phase_planes(model)

    def __enter__(self):
        core = self.model.core.optical_branch.core
        for owner, prop in [('router', core.router.propagator), ('body', core.propagator)]:
            original = prop.forward
            self.originals.append((prop, original))
            def forward(module, field, owner=owner, original=original):
                if self.index >= 3 or owner != ('router','body','body')[self.index]:
                    raise RuntimeError('Unexpected SALICON optical propagation order')
                stage = STAGES[self.index]
                self.index += 1
                a = core.geometry.active_aperture
                active = field[:, a.y0:a.y1, a.x0:a.x1]
                phase = torch.as_tensor(self.planes[stage], device=field.device)
                amplitude = active * torch.exp(-1j*phase)
                if amplitude.imag.abs().max() > 2e-5 or amplitude.real.min() < -2e-5:
                    raise RuntimeError('Phase layout does not recover nonnegative real input')
                self.amplitudes[stage] = amplitude.real.clamp_min(0).detach()
                if stage in self.measured:
                    intensity = self.measured[stage].to(field.device).float()
                    if intensity.shape != active.shape or not torch.isfinite(intensity).all() or intensity.min() < 0:
                        raise ValueError('Measured CCD must be finite nonnegative [B,478,478]')
                    self.detectors[stage] = intensity.detach()
                    output = torch.zeros_like(field)
                    output[:, a.y0:a.y1, a.x0:a.x1] = intensity.sqrt().to(torch.complex64)
                    return output
                output = original(field)
                self.detectors[stage] = output[:,a.y0:a.y1,a.x0:a.x1].abs().square().detach()
                if stage == self.stop_before:
                    raise StopAtPlane(stage)
                return output
            prop.forward = MethodType(forward, prop)
        return self

    def __exit__(self, *args):
        for prop, original in self.originals:
            prop.forward = original


def replay(model, batch, measured=None, stop_before=None):
    result = None
    with torch.inference_mode(), OpticalBoundary(model, measured, stop_before) as tap:
        try:
            result = model(batch)
        except StopAtPlane:
            pass
    return result, tap
