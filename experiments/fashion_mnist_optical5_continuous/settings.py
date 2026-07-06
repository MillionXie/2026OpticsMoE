from __future__ import annotations

import json
import os
import re
from dataclasses import asdict,dataclass,field,fields
from pathlib import Path
from typing import Any


PROJECT_DIR=Path(__file__).resolve().parent
ENV_REFERENCE=re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


@dataclass
class PhaseDropoutSettings:
    enabled:bool=False
    p:float=.05
    mode:str="block_phase_bypass"
    block_size:int=8
    batch_shared:bool=True
    start_epoch:int=10


@dataclass
class Settings:
    dataset:str="fashion_mnist"
    data_root:Path=PROJECT_DIR/"data"/"fashion_mnist"
    download:bool=True
    output_dir:Path=PROJECT_DIR/"runs"/"fashion_mnist_optical5_continuous_uniform"
    input_size:int=224
    num_classes:int=10
    class_names:list[str]|None=None
    validation_fraction:float=.1
    epochs:int=50
    batch_size:int=64
    num_workers:int=8
    learning_rate:float=1e-3
    weight_decay:float=5e-4
    seed:int=42
    device:str="cuda"
    log_interval_batches:int=100
    save_interval_epochs:int=5
    train_limit:int|None=None
    test_limit:int|None=None
    train_samples_per_class_per_epoch:int|None=None
    optical_layers:int=5
    optical_field_size:int=256
    optical_padding_size:int=400
    wavelength_nm:float=532.0
    pixel_pitch_um:float=17.0
    mask_distance_cm:float=5.0
    phase_init:str="uniform"
    amplitude_mask_enabled:bool=True
    readout_channels:list[int]|None=None
    readout_pool_size:int=8
    readout_hidden_dim:int=256
    readout_dropout:float=.2
    detector_region_size:int=28
    detector_region_temperature:float=1.0
    detector_region_loss_weight:float=1.0
    detector_concentration_loss_weight:float=.1
    phase_smoothness_weight:float=0.0
    regularization:dict[str,Any]=field(default_factory=lambda:{"phase_dropout":asdict(PhaseDropoutSettings())})

    def __post_init__(self)->None:
        if self.class_names is None:self.class_names=["t_shirt_top","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle_boot"]
        if self.readout_channels is None:self.readout_channels=[16,32]

    @property
    def phase_dropout(self)->PhaseDropoutSettings:return PhaseDropoutSettings(**(self.regularization.get("phase_dropout",{}) or {}))

    def validate(self)->None:
        if self.dataset!="fashion_mnist":raise ValueError("dataset must be fashion_mnist")
        if len(self.class_names)!=10 or self.num_classes!=10:raise ValueError("Fashion-MNIST requires ten classes")
        if self.optical_layers!=5:raise ValueError("Exactly five continuous propagation layers are required")
        if self.optical_padding_size<self.optical_field_size:raise ValueError("padding must be >= field size")
        if self.phase_init not in {"uniform","zeros"}:raise ValueError("phase_init must be uniform or zeros")
        if not 0<self.validation_fraction<1:raise ValueError("validation_fraction must be between 0 and 1")
        for name in ("epochs","batch_size","input_size","num_workers","log_interval_batches","save_interval_epochs","detector_region_size"):
            if int(getattr(self,name))<0 or (name!="num_workers" and int(getattr(self,name))==0):raise ValueError(f"{name} has an invalid value")
        for name in ("train_limit","test_limit","train_samples_per_class_per_epoch"):
            value=getattr(self,name)
            if value is not None and value<=0:raise ValueError(f"{name} must be positive")
        if self.detector_region_temperature<=0:raise ValueError("detector region temperature must be positive")
        for name in ("detector_region_loss_weight","detector_concentration_loss_weight","phase_smoothness_weight"):
            if float(getattr(self,name))<0:raise ValueError(f"{name} must be nonnegative")
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

