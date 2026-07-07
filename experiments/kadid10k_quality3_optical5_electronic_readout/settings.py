from __future__ import annotations

import json
import os
import re
from dataclasses import asdict,dataclass,field,fields
from pathlib import Path
from typing import Any


PROJECT_DIR=Path(__file__).resolve().parent
QWEN_EXPERIMENT="qwen3_vl_2b_kadid10k_quality3_optical_fullstack4_token64_residual"
ENV_REFERENCE=re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


@dataclass
class PhaseDropoutSettings:
    enabled:bool=False;p:float=0.05;mode:str="block_phase_bypass";block_size:int=8;batch_shared:bool=True;start_epoch:int=10


@dataclass
class Settings:
    dataset:str="kadid10k_quality3"
    data_root:Path=PROJECT_DIR.parent/QWEN_EXPERIMENT/"data"/"kadid10k"
    download:bool=True
    dataset_download_url:str="https://files.osf.io/v1/resources/xkqjh/providers/osfstorage/5eafe5bf0ffc0500ec6f6c94/?zip="
    metadata_csv:str="dmos.csv"
    image_dir:str="images"
    quality_label_mode:str="score_tertile"
    quality_score_higher_is_better:bool|None=None
    train_reference_fraction:float=0.8
    test_reference_fraction:float=0.2
    input_size:int=224
    num_classes:int=3
    class_names:list[str]|None=None
    validation_fraction:float=0.1
    epochs:int=100
    batch_size:int=32
    num_workers:int=8
    learning_rate:float=1e-3
    weight_decay:float=5e-4
    seed:int=42
    device:str="cuda"
    log_interval_batches:int=50
    save_interval_epochs:int=10
    save_predictions_interval_epochs:int=1
    train_limit:int|None=None
    test_limit:int|None=None
    train_limit_per_class:int|None=None
    test_limit_per_class:int|None=None
    train_samples_per_class_per_epoch:int|None=None
    model_type:str="optical5_electronic_readout"
    output_dir:Path=PROJECT_DIR/"runs"/"kadid10k_quality3_optical5_electronic_readout"
    optical_layers:int=5
    optical_field_size:int=64
    optical_padding_size:int=128
    wavelength_nm:float=532.0
    pixel_pitch_um:float=8.0
    mask_distance_cm:float=5.0
    phase_init:str="zeros"
    amplitude_mask_enabled:bool=False
    intensity_forward:bool=True
    readout_channels:list[int]|None=None
    readout_pool_size:int=8
    readout_hidden_dim:int=256
    readout_dropout:float=0.2
    detector_region_size:int=12
    detector_region_temperature:float=1.0
    detector_region_loss_weight:float=1.0
    detector_concentration_loss_weight:float=0.1
    regularization:dict[str,Any]=field(default_factory=lambda:{"phase_dropout":asdict(PhaseDropoutSettings())})
    quality_score_column:str|None=None

    def __post_init__(self)->None:
        if self.class_names is None:self.class_names=["high_quality","medium_quality","low_quality"]
        if self.readout_channels is None:self.readout_channels=[16,32]

    @property
    def phase_dropout(self)->PhaseDropoutSettings:
        return PhaseDropoutSettings(**(self.regularization.get("phase_dropout",{}) or {}))

    def validate(self)->None:
        if self.dataset!="kadid10k_quality3":raise ValueError("dataset must be kadid10k_quality3")
        if self.class_names!=["high_quality","medium_quality","low_quality"] or self.num_classes!=3:raise ValueError("Quality-3 class order is fixed")
        if self.quality_label_mode not in {"score_tertile","distortion_level_3class"}:raise ValueError("Unsupported quality_label_mode")
        if not self.metadata_csv.strip() or not self.image_dir.strip():raise ValueError("metadata_csv and image_dir must be non-empty")
        if self.download and not self.dataset_download_url.strip():raise ValueError("dataset_download_url must be non-empty when download=true")
        if self.model_type!="optical5_electronic_readout":raise ValueError("This baseline only implements optical5_electronic_readout")
        if self.optical_layers!=5 or not self.intensity_forward:raise ValueError("The baseline requires five intensity-forward optical layers")
        if self.optical_padding_size<self.optical_field_size:raise ValueError("optical padding must be >= field size")
        if len(self.readout_channels)!=2:raise ValueError("Electronic readout requires two convolution channel values")
        if not 0<self.detector_region_size<=self.optical_field_size//4:raise ValueError("detector regions must be non-overlapping")
        if not 0<self.validation_fraction<1:raise ValueError("validation_fraction must be between 0 and 1")
        if abs(self.train_reference_fraction+self.test_reference_fraction-1)>1e-9:raise ValueError("reference fractions must sum to 1")
        for name in ("epochs","batch_size","input_size","log_interval_batches","save_interval_epochs","save_predictions_interval_epochs"):
            if int(getattr(self,name))<=0:raise ValueError(f"{name} must be positive")
        for name in ("train_limit","test_limit","train_limit_per_class","test_limit_per_class","train_samples_per_class_per_epoch"):
            value=getattr(self,name)
            if value is not None and int(value)<=0:raise ValueError(f"{name} must be positive when set")
        dropout=self.phase_dropout
        if dropout.mode not in {"none","phase_bypass","block_phase_bypass"}:raise ValueError("Unsupported phase dropout mode")
        if not 0<=dropout.p<1:raise ValueError("phase dropout p must satisfy 0 <= p < 1")

    def to_dict(self)->dict[str,Any]:return asdict(self)


def load_settings(path:str|Path)->Settings:
    config=resolve_path(path,Path.cwd(),"config");raw=json.loads(config.read_text(encoding="utf-8"));allowed={item.name for item in fields(Settings)};unknown=sorted(set(raw)-allowed)
    if unknown:raise ValueError(f"Unknown config keys: {unknown}")
    for name in ("data_root","output_dir"):
        if name in raw:raw[name]=resolve_path(raw[name],config.parent,name)
    settings=Settings(**raw);settings.validate();return settings


def resolve_path(value:str|Path,base:Path,field_name:str)->Path:
    expanded=os.path.expandvars(os.path.expanduser(str(value)));unresolved={a or b for a,b in ENV_REFERENCE.findall(expanded)}
    if unresolved:raise ValueError(f"{field_name} has unset environment variables: {sorted(unresolved)}")
    path=Path(expanded);return (path if path.is_absolute() else base/path).resolve()
