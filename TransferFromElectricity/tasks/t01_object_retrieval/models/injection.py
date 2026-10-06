"""Use PyTorch parametrization to consume external raw phases without detaching."""
import torch
from torch import nn
from torch.nn.utils import parametrize


class ExternalRawPhase(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.current = original.detach().clone()

    def forward(self, original):
        return self.current


def expert_planes(replacement):
    planes = []
    for surrogate in (replacement.vision_surrogate, replacement.language_surrogate):
        core = surrogate.core.optical_branch.core
        local = [expert for layer in core.expert_layers for expert in layer.experts]
        if len(local) != 4 or any(tuple(x.raw_phase.shape) != (224, 224) for x in local):
            raise RuntimeError('Expected four 224x224 expert planes per modality')
        planes.extend(local)
    return planes


def global_planes(replacement):
    planes=[s.core.optical_branch.core.global_phase.phase for s in
            (replacement.vision_surrogate,replacement.language_surrogate)]
    if any(tuple(p.raw_phase.shape)!=(478,478) for p in planes):
        raise RuntimeError('Expected one 478x478 global per modality')
    return planes


class ExpertInjection:
    def __init__(self, planes):
        self.planes = list(planes)
        self.adapters = []
        for plane in self.planes:
            adapter = ExternalRawPhase(plane.raw_phase)
            parametrize.register_parametrization(plane, 'raw_phase', adapter)
            plane.parametrizations.raw_phase.original.requires_grad_(False)
            self.adapters.append(adapter)

    def bind(self, raw):
        if tuple(raw.shape[:2]) != (2, 4) or len(self.adapters) != 8:
            raise ValueError('Expected [2,4,H,W] static expert bank')
        flat = raw.flatten(0, 1)
        for adapter, value, plane in zip(self.adapters, flat, self.planes):
            if value.shape != plane.parametrizations.raw_phase.original.shape:
                raise ValueError('Generated phase geometry mismatch')
            adapter.current = value

    def materialize(self):
        for plane in self.planes:
            parametrize.remove_parametrizations(plane, 'raw_phase', leave_parametrized=True)
            plane.raw_phase.requires_grad_(False)


class GlobalInjection(ExpertInjection):
    def bind(self,raw):
        if tuple(raw.shape)!=(2,478,478) or len(self.adapters)!=2:
            raise ValueError('Expected [2,478,478] static global planes')
        for adapter,value in zip(self.adapters,raw):adapter.current=value
