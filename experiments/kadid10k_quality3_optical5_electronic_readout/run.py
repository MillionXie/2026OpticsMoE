from __future__ import annotations

import argparse
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import load_data
from .metrics import write_json
from .models import build_model,parameter_report
from .settings import load_settings,resolve_path
from .training import test_model,train_model


def build_parser()->argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="KADID-10k Quality-3 five-layer optical front-end plus electronic readout baseline")
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--phase",choices=["prepare_data","train","test","all"],default="all")
    parser.add_argument("--device");parser.add_argument("--epochs",type=int);parser.add_argument("--output-dir",type=Path)
    return parser


def main(argv:list[str]|None=None)->int:
    args=build_parser().parse_args(argv);settings=load_settings(args.config)
    if args.device:settings.device=args.device
    if args.epochs:settings.epochs=args.epochs
    if args.output_dir:settings.output_dir=resolve_path(args.output_dir,Path.cwd(),"output_dir")
    settings.validate();_seed(settings.seed);_dirs(settings.output_dir);data=load_data(settings)
    write_json(settings.output_dir/"config_resolved.json",settings.to_dict());write_json(settings.output_dir/"environment.json",_environment());write_json(settings.output_dir/"dataset.json",data.metadata)
    if args.phase=="prepare_data":print(f"KADID Quality-3 ready: train={len(data.train)} validation={len(data.validation)} test={len(data.test)}");return 0
    device=torch.device(settings.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    model=build_model(settings).to(device);report=parameter_report(model);report.update({"model_type":settings.model_type,"input_size":settings.input_size,
        "class_names":settings.class_names,"optical_field_size":settings.optical_field_size,"optical_padding_size":settings.optical_padding_size,
        "wavelength_nm":settings.wavelength_nm,"pixel_pitch_um":settings.pixel_pitch_um,"mask_distance_cm":settings.mask_distance_cm,
        "phase_init":settings.phase_init,"amplitude_mask_enabled":settings.amplitude_mask_enabled,"phase_dropout":settings.regularization.get("phase_dropout",{}),
        "pipeline":"grayscale -> optical intensity layer x5 -> two-convolution electronic readout -> MLP"})
    write_json(settings.output_dir/"model.json",report)
    if args.phase in {"train","all"}:train_model(model,data,settings,device)
    if args.phase in {"test","all"}:
        metrics=test_model(model,data,settings,device);print(f"[test] top1={metrics['top1_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} balanced={metrics['balanced_accuracy']:.4f}")
    return 0


def _dirs(root:Path)->None:
    for name in ("metrics","checkpoints","figures/phase_masks","figures/light_fields","figures/detector_outputs","figures/detector_regions"):(root/name).mkdir(parents=True,exist_ok=True)
def _seed(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def _environment()->dict[str,Any]:return {"python":platform.python_version(),"platform":platform.platform(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


if __name__=="__main__":raise SystemExit(main())
