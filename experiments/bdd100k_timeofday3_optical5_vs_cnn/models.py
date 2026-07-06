from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .optics import OpticalDetectionIntensityLayer


class EnhancedDetectorReadout(nn.Module):
    def __init__(self,channels:list[int],pool_size:int,hidden_dim:int,dropout:float,num_classes:int)->None:
        super().__init__();c1,c2=channels
        self.stem=nn.Sequential(
            nn.Conv2d(1,c1,3,padding=1),_norm(c1),nn.GELU(),
            nn.Conv2d(c1,c2,3,stride=2,padding=1),_norm(c2),nn.GELU(),
        )
        self.pool=nn.AdaptiveAvgPool2d((pool_size,pool_size))
        feature_dim=c2*pool_size*pool_size
        self.head=nn.Sequential(
            nn.LayerNorm(feature_dim),nn.Linear(feature_dim,hidden_dim),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(hidden_dim,num_classes),
        )
    def forward(self,intensity:torch.Tensor)->torch.Tensor:
        value=torch.log1p(intensity).unsqueeze(1);features=self.stem(value)
        return self.head(self.pool(features).flatten(1))


class Optical5EnhancedTimeOfDayClassifier(nn.Module):
    def __init__(self,input_size:int=224,field_size:int=256,padding_size:int=400,wavelength_nm:float=532,
                 pixel_pitch_um:float=17,distance_cm:float=5,phase_init:str="uniform",amplitude_mask_enabled:bool=True,
                 readout_channels:list[int]|None=None,readout_pool_size:int=8,readout_hidden_dim:int=256,
                 readout_dropout:float=.2,num_classes:int=3,optical_layers:int=5)->None:
        super().__init__();
        if optical_layers!=5:raise ValueError("Exactly five optical layers are required")
        self.field_size=field_size;self.layers=nn.ModuleList([OpticalDetectionIntensityLayer(field_size,padding_size,wavelength_nm,pixel_pitch_um,distance_cm,phase_init,amplitude_mask_enabled) for _ in range(5)])
        self.readout=EnhancedDetectorReadout(readout_channels or [16,32],readout_pool_size,readout_hidden_dim,readout_dropout,num_classes)
        self.last_diagnostics:dict|None=None
    def encode(self,grayscale:torch.Tensor)->torch.Tensor:
        if grayscale.ndim!=4 or grayscale.shape[1]!=1:raise ValueError("Expected [B,1,H,W] grayscale")
        value=grayscale.float().clamp(0,1);rms=value.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6);value=value/rms
        return F.interpolate(value,size=(self.field_size,self.field_size),mode="bilinear",align_corners=False)[:,0]
    def forward(self,grayscale:torch.Tensor,return_diagnostics:bool=False):
        value=self.encode(grayscale);initial=value;intermediates=[]
        for layer in self.layers:
            value=layer(value)
            if return_diagnostics:intermediates.append(value)
        logits=self.readout(value)
        if return_diagnostics:return logits,{"input_intensity":initial,"after_layers":intermediates,"detector_input":value}
        return logits


class ConvBlock(nn.Module):
    def __init__(self,input_channels:int,output_channels:int)->None:
        super().__init__();self.block=nn.Sequential(nn.Conv2d(input_channels,output_channels,3,padding=1,bias=False),_norm(output_channels),nn.GELU())
    def forward(self,value:torch.Tensor)->torch.Tensor:return self.block(value)


class ElectronicCNNTimeOfDayBaseline(nn.Module):
    def __init__(self,channels:list[int]|None=None,dropout:float=.2,num_classes:int=3)->None:
        super().__init__();c0,c1,c2,c3=channels or [32,64,128,256]
        self.features=nn.Sequential(ConvBlock(1,c0),ConvBlock(c0,c1),ConvBlock(c1,c1),nn.MaxPool2d(2),ConvBlock(c1,c2),ConvBlock(c2,c2),nn.MaxPool2d(2),ConvBlock(c2,c3),ConvBlock(c3,c3),nn.MaxPool2d(2),ConvBlock(c3,c3),nn.Dropout2d(.1))
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.LayerNorm(c3),nn.Linear(c3,256),nn.GELU(),nn.Dropout(dropout),nn.Linear(256,num_classes))
    def forward(self,value:torch.Tensor)->torch.Tensor:return self.head(self.features(value))


def build_model(settings:object)->nn.Module:
    if settings.model_type=="electronic_cnn":return ElectronicCNNTimeOfDayBaseline(settings.cnn_channels,settings.cnn_dropout,settings.num_classes)
    return Optical5EnhancedTimeOfDayClassifier(settings.input_size,settings.optical_field_size,settings.optical_padding_size,settings.wavelength_nm,settings.pixel_pitch_um,settings.mask_distance_cm,settings.phase_init,settings.amplitude_mask_enabled,settings.readout_channels,settings.readout_pool_size,settings.readout_hidden_dim,settings.readout_dropout,settings.num_classes,settings.optical_layers)


def parameter_report(model:nn.Module)->dict:
    total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    report={"model_name":type(model).__name__,"parameters":total,"trainable_parameters":trainable}
    if isinstance(model,Optical5EnhancedTimeOfDayClassifier):
        report.update({"optical_layers":5,"phase_mask_parameters":sum(layer.phase_mask.numel() for layer in model.layers),"amplitude_mask_parameters":sum(layer.amplitude_mask_logits.numel() for layer in model.layers if layer.amplitude_mask_logits is not None),"readout_parameters":sum(p.numel() for p in model.readout.parameters()),"intensity_forward":True})
    return report


def _norm(channels:int)->nn.Module:
    groups=min(8,channels)
    while channels%groups:groups-=1
    return nn.GroupNorm(groups,channels)
