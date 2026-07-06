from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .angular_spectrum import AngularSpectrumPropagator


class OpticalDetectionIntensityLayer(nn.Module):
    """Trainable modulation, propagation, and intensity-to-intensity detection."""
    def __init__(self,field_size:int,padding_size:int,wavelength_nm:float,pixel_pitch_um:float,distance_cm:float,
                 phase_init:str="uniform",amplitude_mask_enabled:bool=True,phase_dropout:object|None=None,eps:float=1e-6)->None:
        super().__init__();self.field_size=field_size;self.eps=eps;self.phase_dropout=phase_dropout;self.phase_dropout_active=False;self.last_phase_dropout_mask=None;self.phase_mask=nn.Parameter(torch.empty(field_size,field_size))
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
        field=torch.complex(value,torch.zeros_like(value));modulation=self._phase_modulation(len(value))
        if self.amplitude_mask_logits is not None:modulation=modulation*torch.sigmoid(self.amplitude_mask_logits.float())
        # Square-law detection is followed by per-sample intensity normalization
        # and a trainable biased ReLU detector nonlinearity. The resulting
        # intensity is passed directly to the next optical layer (no square root).
        detected=self.propagator(field*modulation).abs().square().float();detected=detected/detected.mean(dim=(-2,-1),keepdim=True).clamp_min(self.eps)
        return F.relu(detected+self.detector_bias.float())
    def wrapped_phase(self)->torch.Tensor:return torch.remainder(self.phase_mask,2*math.pi)
    def set_phase_dropout_active(self,active:bool)->None:self.phase_dropout_active=bool(active)
    def _phase_modulation(self,batch_size:int)->torch.Tensor:
        modulation=torch.exp(1j*self.phase_mask.float()).to(torch.complex64).unsqueeze(0);cfg=self.phase_dropout
        enabled=cfg is not None and cfg.enabled and self.training and self.phase_dropout_active and cfg.p>0 and cfg.mode!="none"
        if not enabled:self.last_phase_dropout_mask=None;return modulation
        mask_batch=1 if cfg.batch_shared else batch_size;keep_probability=1-float(cfg.p)
        if cfg.mode=="phase_bypass":keep=(torch.rand(mask_batch,self.field_size,self.field_size,device=self.phase_mask.device)<keep_probability).float()
        elif cfg.mode=="block_phase_bypass":
            block=max(1,int(cfg.block_size));low=math.ceil(self.field_size/block);keep=(torch.rand(mask_batch,low,low,device=self.phase_mask.device)<keep_probability).float();keep=keep.repeat_interleave(block,-2).repeat_interleave(block,-1)[:,:self.field_size,:self.field_size]
        else:raise RuntimeError(f"Unexpected phase dropout mode: {cfg.mode}")
        self.last_phase_dropout_mask=keep.detach();keep=keep.to(torch.complex64)
        return keep*modulation+(1-keep)
