from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .optics import ClassRegionDetector,OpticalDetectionIntensityLayer


class ElectronicDetectorReadout(nn.Module):
    """Two small electronic convolutions followed by average pooling and an MLP."""
    def __init__(self,channels:list[int],pool_size:int,hidden_dim:int,dropout:float,num_classes:int)->None:
        super().__init__();c1,c2=channels
        self.features=nn.Sequential(
            nn.Conv2d(1,c1,3,padding=1),_norm(c1),nn.GELU(),
            nn.Conv2d(c1,c2,3,stride=2,padding=1),_norm(c2),nn.GELU(),
            nn.AdaptiveAvgPool2d((pool_size,pool_size)),nn.Flatten(),
        )
        feature_dim=c2*pool_size*pool_size+num_classes
        self.head=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,num_classes))
    def forward(self,intensity:torch.Tensor,region_distribution:torch.Tensor)->torch.Tensor:
        features=self.features(torch.log1p(intensity).unsqueeze(1))
        return self.head(torch.cat((features,region_distribution.float()),dim=1))


class Optical5ElectronicReadoutQualityClassifier(nn.Module):
    """Grayscale image -> five O-E-O conversions -> small electronic readout."""
    def __init__(self,input_size:int,field_size:int,padding_size:int,wavelength_nm:float,pixel_pitch_um:float,
                 distance_cm:float,phase_init:str,amplitude_mask_enabled:bool,readout_channels:list[int],
                 readout_pool_size:int,readout_hidden_dim:int,readout_dropout:float,num_classes:int,
                 optical_layers:int,phase_dropout:object|None,class_names:list[str],detector_region_size:int,
                 detector_region_temperature:float,detector_region_loss_weight:float,
                 detector_concentration_loss_weight:float)->None:
        super().__init__()
        if optical_layers!=5:raise ValueError("Exactly five optical layers are required")
        self.input_size=int(input_size);self.field_size=int(field_size)
        self.layers=nn.ModuleList([OpticalDetectionIntensityLayer(field_size,padding_size,wavelength_nm,pixel_pitch_um,distance_cm,phase_init,amplitude_mask_enabled,phase_dropout) for _ in range(optical_layers)])
        self.class_detector=ClassRegionDetector(field_size,class_names,detector_region_size,detector_region_temperature)
        self.readout=ElectronicDetectorReadout(readout_channels,readout_pool_size,readout_hidden_dim,readout_dropout,num_classes)
        self.detector_region_loss_weight=float(detector_region_loss_weight);self.detector_concentration_loss_weight=float(detector_concentration_loss_weight)

    def encode(self,grayscale:torch.Tensor)->torch.Tensor:
        if grayscale.ndim!=4 or grayscale.shape[1]!=1:raise ValueError("Expected [B,1,H,W] grayscale")
        value=grayscale.float().clamp(0,1);value=value/value.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        return F.interpolate(value,size=(self.field_size,self.field_size),mode="bilinear",align_corners=False)[:,0]

    def forward(self,grayscale:torch.Tensor,return_diagnostics:bool=False,return_aux:bool=False):
        value=self.encode(grayscale);initial=value;intermediates=[]
        for layer in self.layers:
            value=layer(value)
            if return_diagnostics:intermediates.append(value)
        detector=self.class_detector(value);logits=self.readout(value,detector["region_distribution"])
        if return_diagnostics:return logits,{"input_intensity":initial,"after_layers":intermediates,"detector_input":value,**detector}
        if return_aux:return logits,detector
        return logits

    def set_epoch(self,epoch:int)->None:
        cfg=self.layers[0].phase_dropout;active=bool(cfg is not None and cfg.enabled and epoch>=cfg.start_epoch)
        for layer in self.layers:layer.set_phase_dropout_active(active)


def build_model(settings:object)->nn.Module:
    return Optical5ElectronicReadoutQualityClassifier(settings.input_size,settings.optical_field_size,settings.optical_padding_size,
        settings.wavelength_nm,settings.pixel_pitch_um,settings.mask_distance_cm,settings.phase_init,settings.amplitude_mask_enabled,
        settings.readout_channels,settings.readout_pool_size,settings.readout_hidden_dim,settings.readout_dropout,settings.num_classes,
        settings.optical_layers,settings.phase_dropout,settings.class_names,settings.detector_region_size,
        settings.detector_region_temperature,settings.detector_region_loss_weight,settings.detector_concentration_loss_weight)


def parameter_report(model:nn.Module)->dict:
    return {"model_name":type(model).__name__,"parameters":sum(p.numel() for p in model.parameters()),
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"optical_layers":len(model.layers),
        "phase_mask_parameters":sum(layer.phase_mask.numel() for layer in model.layers),
        "amplitude_mask_parameters":sum(layer.amplitude_mask_logits.numel() for layer in model.layers if layer.amplitude_mask_logits is not None),
        "electronic_readout_parameters":sum(p.numel() for p in model.readout.parameters()),"inter_layer_detection":True,
        "inter_layer_intensity_normalization":True,"inter_layer_nonlinearity":True,"electronic_readout_convolutions":2,
        "class_region_detector":model.class_detector.specification(),"detector_region_loss_weight":model.detector_region_loss_weight,
        "detector_concentration_loss_weight":model.detector_concentration_loss_weight}


def _norm(channels:int)->nn.Module:
    groups=min(8,channels)
    while channels%groups:groups-=1
    return nn.GroupNorm(groups,channels)
