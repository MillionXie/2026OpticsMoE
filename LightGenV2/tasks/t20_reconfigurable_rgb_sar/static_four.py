"""T20-only four physical experts, preserving the original aperture in native8.

Uses the existing quantized CCD/OEO operations, NOT T21's16-slot layout.
No source or checkpoint in T21 is changed.
"""
import torch
from .hardware_aware import NativeOEOOptics
from LightGenV2.tasks.t21_continual_learning_hardware.model import HardwareOptics

CONTRACT = 'T20_static4_fullaperture_native8_v5'


class FourExpertOptics(HardwareOptics):
    def __init__(self, architecture):
        NativeOEOOptics.__init__(self, architecture, expert_grid=2,
            expert_size=476, gap=64, border=43,
            routing_temperature=1.25, readout='linear')
        self.register_buffer('display_gains', torch.tensor([32.,128.,384.]))


def build_static_four(architecture, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = FourExpertOptics(architecture)
        model.configure_stage(1)
        with torch.no_grad():
            for name, value in model.named_parameters():
                if 'phase' in name:
                    value.normal_(0,.02)
            torch.manual_seed(seed+909)
            torch.nn.init.xavier_uniform_(model.shared_head.weight,gain=.1)
    return model


def geometry(model):
    return dict(active_pixels=model.active, canvas_pixels=model.canvas,
        input_and_phase_pitch_um=8, expert_grid=model.expert_grid,
        physical_experts=4 if model.architecture=='moe' else 0,
        active_experts=4 if model.architecture=='moe' else 0,
        expert_pixels=model.expert_size, gap_pixels=64, border_pixels=model.border,
        expert_aperture_mm=476*.008, aperture_mm=1016*.008,
        optical_parameters=sum(p.numel() for n,p in model.named_parameters() if 'phase' in n),
        readout_parameters=model.shared_head.weight.numel(), extra_inference_layers=False)
