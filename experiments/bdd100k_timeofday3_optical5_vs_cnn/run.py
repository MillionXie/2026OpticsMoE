from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
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
    parser=argparse.ArgumentParser(description="BDD100K TimeOfDay-3 optical5 O-E-O, continuous optical, and electronic CNN")
    parser.add_argument("--config",type=Path);parser.add_argument("--phase",choices=["prepare_data","train","test","compare","compare_optical","all"],default="all")
    parser.add_argument("--device");parser.add_argument("--epochs",type=int);parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--optical-output-dir",type=Path);parser.add_argument("--continuous-output-dir",type=Path);parser.add_argument("--cnn-output-dir",type=Path);return parser


def main(argv:list[str]|None=None)->int:
    args=build_parser().parse_args(argv)
    if args.phase=="compare":return _compare(args)
    if args.phase=="compare_optical":return _compare_optical(args)
    if args.config is None:raise SystemExit("--config is required except for --phase compare")
    settings=load_settings(args.config)
    if args.device:settings.device=args.device
    if args.epochs:settings.epochs=args.epochs
    if args.output_dir:settings.output_dir=resolve_path(args.output_dir,Path.cwd(),"output_dir")
    settings.validate();_seed(settings.seed);_dirs(settings.output_dir);write_json(settings.output_dir/"config_resolved.json",settings.to_dict());write_json(settings.output_dir/"environment.json",_environment())
    data=load_data(settings);write_json(settings.output_dir/"dataset.json",data.metadata)
    if args.phase=="prepare_data":print(f"TimeOfDay-3 ready: train={len(data.train)} validation={len(data.validation)} test={len(data.test)}");return 0
    device=torch.device(settings.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    model=build_model(settings).to(device);report=parameter_report(model);report.update({"model_type":settings.model_type,"input_size":settings.input_size,"class_names":settings.class_names})
    if settings.model_type in {"optical5_enhanced","optical5_continuous"}:report.update({"optical_field_size":settings.optical_field_size,"optical_padding_size":settings.optical_padding_size,"wavelength_nm":settings.wavelength_nm,"pixel_pitch_um":settings.pixel_pitch_um,"mask_distance_cm":settings.mask_distance_cm,"phase_dropout":settings.regularization.get("phase_dropout",{}),"detector_region_objective":{"region_loss_weight":settings.detector_region_loss_weight,"concentration_loss_weight":settings.detector_concentration_loss_weight,"readout_uses_region_distribution":True}})
    write_json(settings.output_dir/"model.json",report)
    if args.phase in {"train","all"}:train_model(model,data,settings,device)
    if args.phase in {"test","all"}:
        metrics=test_model(model,data,settings,device);print(f"[test] top1={metrics['top1_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} balanced={metrics['balanced_accuracy']:.4f}")
    return 0


def _compare(args:argparse.Namespace)->int:
    if args.optical_output_dir is None or args.cnn_output_dir is None:raise SystemExit("compare requires --optical-output-dir and --cnn-output-dir")
    optical=resolve_path(args.optical_output_dir,Path.cwd(),"optical_output_dir");cnn=resolve_path(args.cnn_output_dir,Path.cwd(),"cnn_output_dir")
    om=_read(optical/"metrics"/"test_metrics.json");cm=_read(cnn/"metrics"/"test_metrics.json");omodel=_read(optical/"model.json");cmodel=_read(cnn/"model.json")
    comparison={"optical":{"top1_accuracy":om["top1_accuracy"],"macro_f1":om["macro_f1"],"balanced_accuracy":om["balanced_accuracy"],"per_class":om["per_class"],"parameters":omodel["parameters"],"training_time_sec":_training_time(optical)},"cnn":{"top1_accuracy":cm["top1_accuracy"],"macro_f1":cm["macro_f1"],"balanced_accuracy":cm["balanced_accuracy"],"per_class":cm["per_class"],"parameters":cmodel["parameters"],"training_time_sec":_training_time(cnn)}}
    write_json(optical/"metrics"/"comparison.json",comparison);print(json.dumps(comparison,indent=2));return 0


def _compare_optical(args:argparse.Namespace)->int:
    if args.optical_output_dir is None or args.continuous_output_dir is None:raise SystemExit("compare_optical requires --optical-output-dir and --continuous-output-dir")
    oeo=resolve_path(args.optical_output_dir,Path.cwd(),"optical_output_dir");continuous=resolve_path(args.continuous_output_dir,Path.cwd(),"continuous_output_dir");oeo_metrics=_read(oeo/"metrics"/"test_metrics.json");continuous_metrics=_read(continuous/"metrics"/"test_metrics.json");oeo_model=_read(oeo/"model.json");continuous_model=_read(continuous/"model.json")
    keys=("top1_accuracy","macro_f1","balanced_accuracy","detector_region_accuracy","mean_detector_energy_fraction","mean_target_region_energy_fraction")
    comparison={"oeo_optical5":{**{key:oeo_metrics.get(key) for key in keys},"parameters":oeo_model["parameters"],"training_time_sec":_training_time(oeo)},"continuous_optical5":{**{key:continuous_metrics.get(key) for key in keys},"parameters":continuous_model["parameters"],"training_time_sec":_training_time(continuous)},"continuous_minus_oeo":{key:continuous_metrics[key]-oeo_metrics[key] for key in keys if continuous_metrics.get(key) is not None and oeo_metrics.get(key) is not None}}
    write_json(oeo/"metrics"/"continuous_optical_comparison.json",comparison);write_json(continuous/"metrics"/"oeo_optical_comparison.json",comparison);print(json.dumps(comparison,indent=2));return 0


def _training_time(root:Path)->float|None:
    path=root/"metrics"/"training_history.csv"
    if not path.is_file():return None
    import csv
    with path.open(encoding="utf-8") as handle:return sum(float(row["epoch_time_sec"]) for row in csv.DictReader(handle))
def _read(path:Path)->dict[str,Any]:
    if not path.is_file():raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
def _dirs(root:Path)->None:
    for name in ("metrics","checkpoints","figures/phase_masks","figures/light_fields","figures/detector_outputs","figures/detector_regions"):(root/name).mkdir(parents=True,exist_ok=True)
def _seed(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def _environment()->dict[str,Any]:return {"python":platform.python_version(),"platform":platform.platform(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


if __name__=="__main__":raise SystemExit(main())
