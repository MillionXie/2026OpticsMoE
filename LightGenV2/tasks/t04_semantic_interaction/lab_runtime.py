"""Pinned standard-head OpenMoji replay at all six real CCD boundaries."""
from pathlib import Path
from types import MethodType
import math
import numpy as np
import torch
from . import standalone
from .modeling import build_model
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
    OpenMojiEditingDataset, collate_samples, load_prompt_cache)

STAGES = ('language_router', 'language_expert', 'language_global',
          'vision_router', 'vision_expert', 'vision_global')
CHECKPOINT_SHA = 'a69ddcee827749fb9202f9aef11ea45011e433d8b2f0151be2eec3db7dbff9eb'
REFERENCE = .8715


def load_model(project, device='cpu'):
    project = Path(project).resolve()
    if standalone.digest(project/'weights/best_checkpoint.pt') != CHECKPOINT_SHA:
        raise ValueError('Require pinned routerfill_shared epoch40, not historical98%')
    standalone.ROOT = project
    cfg = standalone.settings(project/'outputs')
    model = build_model(cfg, torch.device(device))
    payload = torch.load(project/'weights/best_checkpoint.pt', map_location='cpu', weights_only=False)
    if payload['architecture'] != model.checkpoint_architecture or payload['epoch'] != 40:
        raise ValueError('Architecture/epoch mismatch')
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    for path in model._optical_paths():
        path.set_phase_dropout_active(False)
    return model, cfg


def dataset(cfg):
    return OpenMojiEditingDataset(cfg.test_manifest, cfg, load_prompt_cache(cfg.prompt_cache_path))


def batch_for(data, index, device):
    batch = collate_samples([data[index]])
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def phase_planes(model):
    planes = {}
    with torch.inference_mode():
        for modality in ('language', 'vision'):
            branch = getattr(model, modality+'_core').optical_branch
            core = branch.core
            g, a = core.geometry, core.geometry.active_aperture
            ones = next(model.parameters()).new_ones(1, g.canvas_size, g.canvas_size).to(torch.complex64)
            planes[modality+'_router'] = core.router.active_phase()
            for suffix, mod in [('expert', core.expert_layers[0](ones)), ('global', core.global_phase(ones))]:
                planes[modality+'_'+suffix] = torch.remainder(torch.angle(mod[0,a.y0:a.y1,a.x0:a.x1]), 2*math.pi)
    return {name: value.detach().cpu().numpy() for name,value in planes.items()}


class StopAtPlane(Exception):
    pass


class OpticalBoundary:
    def __init__(self, model, measured=None, stop_before=None):
        self.model, self.measured, self.stop_before = model, measured or {}, stop_before
        if tuple(self.measured) != STAGES[:len(self.measured)]:
            raise ValueError('Measured frames must be a contiguous language-first prefix')
        if stop_before is not None and stop_before not in STAGES:
            raise ValueError('Unknown optical plane')
        self.originals, self.amplitudes, self.detectors = [], {}, {}
        self.index = 0
        self.planes = phase_planes(model)

    def __enter__(self):
        for modality in ('language', 'vision'):
            core = getattr(self.model, modality+'_core').optical_branch.core
            for kind, prop in [('router',core.router.propagator), ('body',core.propagator)]:
                original = prop.forward
                self.originals.append((prop, original))
                def forward(module, field, modality=modality, kind=kind, core=core, original=original):
                    if self.index >= 6:
                        raise RuntimeError('Unexpected extra optical propagation')
                    stage = STAGES[self.index]
                    if not stage.startswith(modality+'_') or (stage.endswith('router')) != (kind=='router'):
                        raise RuntimeError('Unexpected optical plane order')
                    self.index += 1
                    a = core.geometry.active_aperture
                    active = field[:,a.y0:a.y1,a.x0:a.x1]
                    phase = torch.as_tensor(self.planes[stage], device=field.device)
                    amplitude = active * torch.exp(-1j*phase)
                    if amplitude.imag.abs().max() > 3e-5 or amplitude.real.min() < -3e-5:
                        raise RuntimeError('Exported phase fails real nonnegative input recovery')
                    self.amplitudes[stage] = amplitude.real.clamp_min(0).detach()
                    if stage in self.measured:
                        intensity = self.measured[stage].to(field.device).float()
                        if intensity.shape != active.shape or not torch.isfinite(intensity).all() or intensity.min()<0:
                            raise ValueError('CCD must be finite nonnegative [B,478,478]')
                        output = torch.zeros_like(field)
                        output[:,a.y0:a.y1,a.x0:a.x1] = intensity.sqrt().to(torch.complex64)
                    else:
                        output = original(field)
                        intensity = output[:,a.y0:a.y1,a.x0:a.x1].abs().square().float()
                    self.detectors[stage] = intensity.detach()
                    if stage == self.stop_before:
                        raise StopAtPlane(stage)
                    return output
                prop.forward = MethodType(forward, prop)
        return self

    def __exit__(self, *args):
        for prop, original in self.originals:
            prop.forward = original


def replay(model, batch, measured=None, stop_before=None):
    output = None
    with torch.inference_mode(), OpticalBoundary(model, measured, stop_before) as tap:
        try:
            output = model(batch['source_image'], batch['prompt_hidden'])
        except StopAtPlane:
            pass
    return output, tap
