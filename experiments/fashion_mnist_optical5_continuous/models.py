from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .optics import ContinuousOpticalPropagationLayer,GridClassRegionDetector


def _norm(channels:int)->nn.Module:
    groups=min(8,channels)
    while channels%groups:groups-=1
    return nn.GroupNorm(groups,channels)


class DetectorReadout(nn.Module):
    def __init__(self,channels:list[int],pool_size:int,hidden_dim:int,dropout:float,num_classes:int)->None:
        super().__init__();c1,c2=channels;self.stem=nn.Sequential(nn.Conv2d(1,c1,3,padding=1),_norm(c1),nn.GELU(),nn.Conv2d(c1,c2,3,stride=2,padding=1),_norm(c2),nn.GELU());self.pool=nn.AdaptiveAvgPool2d((pool_size,pool_size));feature_dim=c2*pool_size*pool_size+num_classes;self.head=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,num_classes))
    def forward(self,intensity:torch.Tensor,region_distribution:torch.Tensor)->torch.Tensor:
        features=self.pool(self.stem(torch.log1p(intensity).unsqueeze(1))).flatten(1);return self.head(torch.cat((features,region_distribution.float()),1))


class FashionMNISTOptical5Continuous(nn.Module):
    def __init__(self,settings:object)->None:
        super().__init__();self.field_size=settings.optical_field_size;self.layers=nn.ModuleList([ContinuousOpticalPropagationLayer(settings.optical_field_size,settings.optical_padding_size,settings.wavelength_nm,settings.pixel_pitch_um,settings.mask_distance_cm,settings.phase_init,settings.amplitude_mask_enabled,settings.phase_dropout) for _ in range(5)]);self.final_detector_bias=nn.Parameter(torch.zeros(()));self.class_detector=GridClassRegionDetector(settings.optical_field_size,settings.class_names,settings.detector_region_size,settings.detector_region_temperature);self.readout=DetectorReadout(settings.readout_channels,settings.readout_pool_size,settings.readout_hidden_dim,settings.readout_dropout,settings.num_classes);self.detector_region_loss_weight=settings.detector_region_loss_weight;self.detector_concentration_loss_weight=settings.detector_concentration_loss_weight;self.phase_smoothness_weight=settings.phase_smoothness_weight;self.eps=1e-6
    def encode(self,grayscale:torch.Tensor)->torch.Tensor:
        value=grayscale.float().clamp(0,1);value=value/value.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(self.eps);return F.interpolate(value,size=(self.field_size,self.field_size),mode="bilinear",align_corners=False)[:,0]
    def forward(self,grayscale:torch.Tensor,return_aux:bool=False,return_diagnostics:bool=False):
        initial=self.encode(grayscale);field=torch.complex(initial,torch.zeros_like(initial));intermediates=[]
        for layer in self.layers:
            field=layer(field)
            if return_diagnostics:intermediates.append(field.abs().square().float())
        intensity=field.abs().square().float();intensity=intensity/intensity.mean((-2,-1),keepdim=True).clamp_min(self.eps);intensity=F.relu(intensity+self.final_detector_bias.float());detector=self.class_detector(intensity);logits=self.readout(intensity,detector["region_distribution"])
        if return_diagnostics:return logits,{"input_intensity":initial,"after_layers":intermediates,"detector_input":intensity,**detector}
        if return_aux:return logits,detector
        return logits
    def set_epoch(self,epoch:int)->None:
        cfg=self.layers[0].phase_dropout;active=bool(cfg.enabled and epoch>=cfg.start_epoch)
        for layer in self.layers:layer.set_phase_dropout_active(active)
    def phase_tv_loss(self)->torch.Tensor:
        losses=[]
        for layer in self.layers:
            phase=layer.phase_mask.float();losses.append((phase[1:]-phase[:-1]).abs().mean()+(phase[:,1:]-phase[:,:-1]).abs().mean())
        return torch.stack(losses).mean()


def parameter_report(model:FashionMNISTOptical5Continuous)->dict:
    return {"model_name":type(model).__name__,"parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"optical_layers":5,"continuous_complex_field":True,"inter_layer_detection":False,"inter_layer_nonlinearity":False,"final_square_law_detection":True,"phase_mask_parameters":sum(layer.phase_mask.numel() for layer in model.layers),"amplitude_mask_parameters":sum(layer.amplitude_mask_logits.numel() for layer in model.layers if layer.amplitude_mask_logits is not None),"readout_parameters":sum(p.numel() for p in model.readout.parameters()),"class_region_detector":model.class_detector.specification(),"phase_dropout":vars(model.layers[0].phase_dropout),"phase_smoothness_weight":model.phase_smoothness_weight}
