from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .optics import ClassRegionDetector,ContinuousOpticalPropagationLayer,OpticalDetectionIntensityLayer


class EnhancedDetectorReadout(nn.Module):
    def __init__(self,channels:list[int],pool_size:int,hidden_dim:int,dropout:float,num_classes:int,region_feature_dim:int=0)->None:
        super().__init__();c1,c2=channels
        self.stem=nn.Sequential(
            nn.Conv2d(1,c1,3,padding=1),_norm(c1),nn.GELU(),
            nn.Conv2d(c1,c2,3,stride=2,padding=1),_norm(c2),nn.GELU(),
        )
        self.pool=nn.AdaptiveAvgPool2d((pool_size,pool_size))
        self.region_feature_dim=int(region_feature_dim);feature_dim=c2*pool_size*pool_size+self.region_feature_dim
        self.head=nn.Sequential(
            nn.LayerNorm(feature_dim),nn.Linear(feature_dim,hidden_dim),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(hidden_dim,num_classes),
        )
    def forward(self,intensity:torch.Tensor,region_features:torch.Tensor|None=None)->torch.Tensor:
        value=torch.log1p(intensity).unsqueeze(1);features=self.stem(value)
        features=self.pool(features).flatten(1)
        if self.region_feature_dim:
            if region_features is None or region_features.shape!=(len(intensity),self.region_feature_dim):raise ValueError(f"Expected region features [B,{self.region_feature_dim}]")
            features=torch.cat((features,region_features.float()),1)
        return self.head(features)


class OpticalTimeOfDayClassifierBase(nn.Module):
    def __init__(self,field_size:int,readout_channels:list[int]|None,readout_pool_size:int,readout_hidden_dim:int,
                 readout_dropout:float,num_classes:int,class_names:list[str]|None,detector_region_size:int,
                 detector_region_temperature:float,detector_region_loss_weight:float,
                 detector_concentration_loss_weight:float)->None:
        super().__init__();names=class_names or ["daytime","night","dawn_dusk"]
        if len(names)!=num_classes:raise ValueError("class_names and num_classes must agree")
        self.field_size=int(field_size);self.class_detector=ClassRegionDetector(field_size,names,detector_region_size,detector_region_temperature);self.detector_region_loss_weight=float(detector_region_loss_weight);self.detector_concentration_loss_weight=float(detector_concentration_loss_weight)
        self.readout=EnhancedDetectorReadout(readout_channels or [16,32],readout_pool_size,readout_hidden_dim,readout_dropout,num_classes,num_classes)

    def encode(self,grayscale:torch.Tensor)->torch.Tensor:
        if grayscale.ndim!=4 or grayscale.shape[1]!=1:raise ValueError("Expected [B,1,H,W] grayscale")
        value=grayscale.float().clamp(0,1);rms=value.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6);value=value/rms
        return F.interpolate(value,size=(self.field_size,self.field_size),mode="bilinear",align_corners=False)[:,0]

    def finish(self,value:torch.Tensor,initial:torch.Tensor,intermediates:list[torch.Tensor],return_diagnostics:bool,return_aux:bool):
        detector=self.class_detector(value);logits=self.readout(value,detector["region_distribution"])
        if return_diagnostics:return logits,{"input_intensity":initial,"after_layers":intermediates,"detector_input":value,**detector}
        if return_aux:return logits,detector
        return logits

    def set_epoch(self,epoch:int)->None:
        cfg=self.layers[0].phase_dropout;active=bool(cfg is not None and cfg.enabled and epoch>=cfg.start_epoch)
        for layer in self.layers:layer.set_phase_dropout_active(active)


class Optical5EnhancedTimeOfDayClassifier(OpticalTimeOfDayClassifierBase):
    """Five O-E-O stages: every layer detects, normalizes, applies ReLU, and reloads."""
    def __init__(self,input_size:int=224,field_size:int=256,padding_size:int=400,wavelength_nm:float=532,
                 pixel_pitch_um:float=17,distance_cm:float=5,phase_init:str="uniform",amplitude_mask_enabled:bool=True,
                 readout_channels:list[int]|None=None,readout_pool_size:int=8,readout_hidden_dim:int=256,
                 readout_dropout:float=.2,num_classes:int=3,optical_layers:int=5,phase_dropout:object|None=None,
                 class_names:list[str]|None=None,detector_region_size:int=48,detector_region_temperature:float=1,
                 detector_region_loss_weight:float=1,detector_concentration_loss_weight:float=.1)->None:
        if optical_layers!=5:raise ValueError("Exactly five optical layers are required")
        super().__init__(field_size,readout_channels,readout_pool_size,readout_hidden_dim,readout_dropout,num_classes,class_names,detector_region_size,detector_region_temperature,detector_region_loss_weight,detector_concentration_loss_weight)
        self.layers=nn.ModuleList([OpticalDetectionIntensityLayer(field_size,padding_size,wavelength_nm,pixel_pitch_um,distance_cm,phase_init,amplitude_mask_enabled,phase_dropout) for _ in range(5)])

    def forward(self,grayscale:torch.Tensor,return_diagnostics:bool=False,return_aux:bool=False):
        value=self.encode(grayscale);initial=value;intermediates=[]
        for layer in self.layers:
            value=layer(value)
            if return_diagnostics:intermediates.append(value)
        return self.finish(value,initial,intermediates,return_diagnostics,return_aux)


class Optical5ContinuousTimeOfDayClassifier(OpticalTimeOfDayClassifierBase):
    """Five mask/propagation stages with one final square-law detector only."""
    def __init__(self,input_size:int=224,field_size:int=256,padding_size:int=400,wavelength_nm:float=532,
                 pixel_pitch_um:float=17,distance_cm:float=5,phase_init:str="uniform",amplitude_mask_enabled:bool=True,
                 readout_channels:list[int]|None=None,readout_pool_size:int=8,readout_hidden_dim:int=256,
                 readout_dropout:float=.2,num_classes:int=3,optical_layers:int=5,phase_dropout:object|None=None,
                 class_names:list[str]|None=None,detector_region_size:int=48,detector_region_temperature:float=1,
                 detector_region_loss_weight:float=1,detector_concentration_loss_weight:float=.1)->None:
        if optical_layers!=5:raise ValueError("Exactly five optical layers are required")
        super().__init__(field_size,readout_channels,readout_pool_size,readout_hidden_dim,readout_dropout,num_classes,class_names,detector_region_size,detector_region_temperature,detector_region_loss_weight,detector_concentration_loss_weight)
        self.layers=nn.ModuleList([ContinuousOpticalPropagationLayer(field_size,padding_size,wavelength_nm,pixel_pitch_um,distance_cm,phase_init,amplitude_mask_enabled,phase_dropout) for _ in range(5)]);self.final_detector_bias=nn.Parameter(torch.zeros(()));self.eps=1e-6

    def forward(self,grayscale:torch.Tensor,return_diagnostics:bool=False,return_aux:bool=False):
        initial=self.encode(grayscale);field=torch.complex(initial,torch.zeros_like(initial));intermediates=[]
        for layer in self.layers:
            field=layer(field)
            if return_diagnostics:intermediates.append(field.abs().square().float())
        detected=field.abs().square().float();detected=detected/detected.mean((-2,-1),keepdim=True).clamp_min(self.eps);value=F.relu(detected+self.final_detector_bias.float())
        return self.finish(value,initial,intermediates,return_diagnostics,return_aux)


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
    model_class=Optical5ContinuousTimeOfDayClassifier if settings.model_type=="optical5_continuous" else Optical5EnhancedTimeOfDayClassifier
    return model_class(settings.input_size,settings.optical_field_size,settings.optical_padding_size,settings.wavelength_nm,settings.pixel_pitch_um,settings.mask_distance_cm,settings.phase_init,settings.amplitude_mask_enabled,settings.readout_channels,settings.readout_pool_size,settings.readout_hidden_dim,settings.readout_dropout,settings.num_classes,settings.optical_layers,settings.phase_dropout,settings.class_names,settings.detector_region_size,settings.detector_region_temperature,settings.detector_region_loss_weight,settings.detector_concentration_loss_weight)


def parameter_report(model:nn.Module)->dict:
    total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    report={"model_name":type(model).__name__,"parameters":total,"trainable_parameters":trainable}
    if isinstance(model,OpticalTimeOfDayClassifierBase):
        continuous=isinstance(model,Optical5ContinuousTimeOfDayClassifier);report.update({"optical_layers":5,"phase_mask_parameters":sum(layer.phase_mask.numel() for layer in model.layers),"amplitude_mask_parameters":sum(layer.amplitude_mask_logits.numel() for layer in model.layers if layer.amplitude_mask_logits is not None),"readout_parameters":sum(p.numel() for p in model.readout.parameters()),"inter_layer_detection":not continuous,"inter_layer_nonlinearity":not continuous,"continuous_complex_field":continuous,"final_square_law_detection":True,"class_region_detector":model.class_detector.specification(),"detector_region_loss_weight":model.detector_region_loss_weight,"detector_concentration_loss_weight":model.detector_concentration_loss_weight})
    return report


def _norm(channels:int)->nn.Module:
    groups=min(8,channels)
    while channels%groups:groups-=1
    return nn.GroupNorm(groups,channels)
