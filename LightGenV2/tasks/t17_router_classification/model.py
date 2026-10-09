"""Reuse the published T16 optical path; only replace the routing computation."""
import torch
from torch import nn
from torch.nn import functional as F
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.model import DirectCCDOptics


class RouterClassification(DirectCCDOptics):
    def __init__(self, architecture):
        if architecture not in ('optical', 'electronic', 'd2nn'):
            raise ValueError(architecture)
        super().__init__('d2nn' if architecture == 'd2nn' else 'moe')
        self.variant = architecture
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(122)
            self.shared_head = nn.Linear(784, 9, bias=False)
            if architecture == 'electronic':
                self.electronic_router = nn.Sequential(nn.Linear(784, 64), nn.GELU(),
                                                      nn.Linear(64, 16))
                nn.init.zeros_(self.electronic_router[-1].weight)
                nn.init.zeros_(self.electronic_router[-1].bias)
                self.register_parameter('router_phase', None)
        self.active_count.fill_(16)
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def route_with_efficiency(self, amplitude, return_debug=False):
        if self.variant != 'electronic':
            return super().route_with_efficiency(amplitude, return_debug)
        x = F.adaptive_avg_pool2d(amplitude[:, None], 28).flatten(1)
        x = x / x.mean(1, keepdim=True).clamp_min(1e-20)
        q = (self.electronic_router(x) / self.routing_temperature).softmax(1)
        return q, None, None
