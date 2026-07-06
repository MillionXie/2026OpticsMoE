from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
ENV_REFERENCE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


@dataclass
class PhaseDropoutSettings:
    enabled: bool = False
    p: float = 0.05
    mode: str = "block_phase_bypass"
    block_size: int = 8
    batch_shared: bool = True
    start_epoch: int = 10


@dataclass
class Settings:
    dataset: str = "bdd100k_timeofday3"
    data_root: Path = PROJECT_DIR / "data" / "bdd100k_timeofday3"
    download: bool = True
    imagefolder_train: str = "train"
    imagefolder_test: str = "test"
    input_size: int = 224
    num_classes: int = 3
    class_names: list[str] | None = None
    validation_fraction: float = 0.1
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 8
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    seed: int = 42
    device: str = "cuda"
    progress: bool = True
    log_interval_batches: int = 50
    save_interval_epochs: int = 10
    save_predictions_interval_epochs: int = 1
    train_limit: int | None = None
    test_limit: int | None = None
    train_limit_per_class: int | None = None
    test_limit_per_class: int | None = None
    model_type: str = "optical5_enhanced"
    output_dir: Path = PROJECT_DIR / "runs" / "optical5"
    optical_layers: int = 5
    optical_field_size: int = 256
    optical_padding_size: int = 400
    wavelength_nm: float = 532.0
    pixel_pitch_um: float = 17.0
    mask_distance_cm: float = 5.0
    phase_init: str = "uniform"
    amplitude_mask_enabled: bool = True
    intensity_forward: bool = True
    readout_channels: list[int] | None = None
    readout_pool_size: int = 8
    readout_hidden_dim: int = 256
    readout_dropout: float = 0.2
    detector_region_size: int = 48
    detector_region_temperature: float = 1.0
    detector_region_loss_weight: float = 1.0
    detector_concentration_loss_weight: float = 0.1
    cnn_channels: list[int] | None = None
    cnn_dropout: float = 0.2
    regularization: dict[str, Any] = field(
        default_factory=lambda: {"phase_dropout": asdict(PhaseDropoutSettings())}
    )

    def __post_init__(self) -> None:
        if self.class_names is None: self.class_names=["daytime","night","dawn_dusk"]
        if self.readout_channels is None: self.readout_channels=[16,32]
        if self.cnn_channels is None: self.cnn_channels=[32,64,128,256]

    def validate(self) -> None:
        if self.dataset!="bdd100k_timeofday3": raise ValueError("dataset must be bdd100k_timeofday3")
        if self.class_names!=["daytime","night","dawn_dusk"] or self.num_classes!=3: raise ValueError("TimeOfDay-3 class order must be daytime, night, dawn_dusk")
        if self.model_type not in {"optical5_enhanced","electronic_cnn"}: raise ValueError("Unsupported model_type")
        if self.optimizer!="adamw" or self.scheduler!="cosine": raise ValueError("Only AdamW + cosine are supported")
        if not 0<self.validation_fraction<1: raise ValueError("validation_fraction must be between 0 and 1")
        for name in ("epochs","batch_size","input_size","log_interval_batches","save_interval_epochs","save_predictions_interval_epochs"):
            if int(getattr(self,name))<=0: raise ValueError(f"{name} must be positive")
        for name in ("train_limit","test_limit","train_limit_per_class","test_limit_per_class"):
            value=getattr(self,name)
            if value is not None and value<=0: raise ValueError(f"{name} must be positive")
        if self.model_type=="optical5_enhanced":
            if self.optical_layers!=5 or not self.intensity_forward: raise ValueError("Optical model requires five intensity-forward layers")
            if self.optical_padding_size<self.optical_field_size: raise ValueError("padding must be >= field size")
            if len(self.readout_channels)!=2: raise ValueError("Simplified optical readout requires exactly two convolution channel values")
            max_nonoverlap_size=self.optical_field_size//(self.num_classes+1)
            if not 0<self.detector_region_size<=max_nonoverlap_size: raise ValueError(f"detector_region_size must be between 1 and {max_nonoverlap_size} for non-overlapping horizontal regions")
            if self.detector_region_temperature<=0: raise ValueError("detector_region_temperature must be positive")
            if self.detector_region_loss_weight<0 or self.detector_concentration_loss_weight<0: raise ValueError("detector loss weights must be nonnegative")
            dropout=self.phase_dropout
            if dropout.mode not in {"none","phase_bypass","block_phase_bypass"}: raise ValueError("Unsupported phase dropout mode")
            if not 0<=dropout.p<1: raise ValueError("phase dropout p must satisfy 0 <= p < 1")
            if dropout.block_size<=0 or dropout.start_epoch<=0: raise ValueError("phase dropout block_size and start_epoch must be positive")

    def to_dict(self)->dict[str,Any]: return asdict(self)

    @property
    def phase_dropout(self)->PhaseDropoutSettings:
        return PhaseDropoutSettings(**(self.regularization.get("phase_dropout",{}) or {}))


def load_settings(path:str|Path)->Settings:
    config=resolve_path(path,Path.cwd(),"config"); raw=json.loads(config.read_text(encoding="utf-8"))
    allowed={item.name for item in fields(Settings)}; unknown=sorted(set(raw)-allowed)
    if unknown: raise ValueError(f"Unknown config keys: {unknown}")
    for name in ("data_root","output_dir"):
        if name in raw: raw[name]=resolve_path(raw[name],config.parent,name)
    settings=Settings(**raw); settings.validate(); return settings


def resolve_path(value:str|Path,base:Path,field_name:str)->Path:
    expanded=os.path.expandvars(os.path.expanduser(str(value))); unresolved={a or b for a,b in ENV_REFERENCE.findall(expanded)}
    if unresolved: raise ValueError(f"{field_name} has unset environment variables: {sorted(unresolved)}")
    path=Path(expanded); return (path if path.is_absolute() else base/path).resolve()
