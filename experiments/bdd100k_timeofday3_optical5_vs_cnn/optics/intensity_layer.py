from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .angular_spectrum import AngularSpectrumPropagator


class OpticalDetectionIntensityLayer(nn.Module):
    """Trainable modulation, propagation, and intensity-to-intensity detection."""
    def __init__(self,field_size:int,padding_size:int,wavelength_nm:float,pixel_pitch_um:float,distance_cm:float,
                 phase_init:str="uniform",amplitude_mask_enabled:bool=True,eps:float=1e-6)->None:
        super().__init__();self.field_size=field_size;self.eps=eps;self.phase_mask=nn.Parameter(torch.empty(field_size,field_size))
        if phase_init=="uniform":nn.init.uniform_(self.phase_mask,0,2*math.pi)
        elif phase_init=="zeros":nn.init.zeros_(self.phase_mask)
        else:raise ValueError("phase_init must be uniform or zeros")
        self.amplitude_mask_logits=nn.Parameter(torch.full((field_size,field_size),4.0)) if amplitude_mask_enabled else None
        self.detector_bias=nn.Parameter(torch.zeros(()));self.propagator=AngularSpectrumPropagator(field_size,padding_size,wavelength_nm,pixel_pitch_um,distance_cm)
    def forward(self,intensity:torch.Tensor)->torch.Tensor:
        if intensity.ndim!=3 or tuple(intensity.shape[-2:])!=(self.field_size,self.field_size):raise ValueError(f"Expected [B,{self.field_size},{self.field_size}]")
        # Every conversion receives a nonnegative intensity-like field. Normalize
        # per sample before modulation to prevent scale drift across five layers.
        value=F.relu(intensity.float());value=value/value.mean(dim=(-2,-1),keepdim=True).clamp_min(self.eps)
        field=torch.complex(value,torch.zeros_like(value));modulation=torch.exp(1j*self.phase_mask.float()).to(torch.complex64)
        if self.amplitude_mask_logits is not None:modulation=modulation*torch.sigmoid(self.amplitude_mask_logits.float())
        # Square-law detection is followed by per-sample intensity normalization
        # and a trainable biased ReLU detector nonlinearity. The resulting
        # intensity is passed directly to the next optical layer (no square root).
        detected=self.propagator(field*modulation).abs().square().float();detected=detected/detected.mean(dim=(-2,-1),keepdim=True).clamp_min(self.eps)
        return F.relu(detected+self.detector_bias.float())
    def wrapped_phase(self)->torch.Tensor:return torch.remainder(self.phase_mask,2*math.pi)
