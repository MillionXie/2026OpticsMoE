"""T03-only phase coordinates: identical optics, no additional parameters.

Direct radians avoid the sigmoid Jacobian near 0/2pi. Propagation already uses
exp(i*phase); hardware encoding wraps modulo 2pi. Never reinterpret an old
raw logit as radians without the explicit checkpoint conversion below.
"""
import math
import torch
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import OpticalDetectorTopKRouter

RADIANS_SUFFIX = '_router_radians'
ROUTER_KEY = 'hybrid.optical_branch.core.router.raw_router_phase'


class RadiansOpticalRouter(OpticalDetectorTopKRouter):
    def __init__(self, geometry, settings):
        super().__init__(geometry,settings)
        with torch.no_grad():
            # Reuse the one existing Parameter: optimizer grouping stays intact.
            self.raw_router_phase.copy_(2*math.pi*self.raw_router_phase.sigmoid())

    def phase(self):
        return self.raw_router_phase


def checkpoint_phase(name, raw, architecture):
    """Physical radians, possibly unwrapped; features retain sigmoid semantics."""
    if 'raw_router_phase' in name and str(architecture).endswith(RADIANS_SUFFIX):
        return raw.float()
    return 2*math.pi*raw.float().sigmoid()


def convert_router_checkpoint(source, target, source_arch, target_arch):
    """One strict coordinate conversion; allow reloading an already-direct run."""
    if not target_arch.endswith(RADIANS_SUFFIX):
        raise RuntimeError('Router radians transfer requires radians target architecture')
    if source_arch not in (target_arch,target_arch.removesuffix(RADIANS_SUFFIX)):
        raise RuntimeError('Router transfer cannot combine unrelated architecture changes')
    if source.keys()!=target.keys() or ROUTER_KEY not in source:
        raise RuntimeError('Router transfer requires identical complete state keys')
    if any(source[k].shape!=target[k].shape for k in source):
        raise RuntimeError('Router transfer cannot change tensor shapes')
    if not torch.isfinite(source[ROUTER_KEY]).all():
        raise RuntimeError('Nonfinite source router phase')
    if source_arch==target_arch:
        return source,False
    result=dict(source)
    result[ROUTER_KEY]=checkpoint_phase(ROUTER_KEY,source[ROUTER_KEY],source_arch)
    return result,True
