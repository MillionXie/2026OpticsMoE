from __future__ import annotations

import argparse
import platform
import random
from pathlib import Path

import numpy as np
import torch

from .data import load_data
from .metrics import write_json
from .models import FashionMNISTOptical5Continuous,parameter_report
from .settings import load_settings,resolve_path
from .training import test_model,train_model


def parser()->argparse.ArgumentParser:
    value=argparse.ArgumentParser(description="Fashion-MNIST five-layer continuous optical propagation mask control");value.add_argument("--config",type=Path,required=True);value.add_argument("--phase",choices=["prepare_data","train","test","all"],default="all");value.add_argument("--device");value.add_argument("--epochs",type=int);value.add_argument("--output-dir",type=Path);return value


def main(argv:list[str]|None=None)->int:
    args=parser().parse_args(argv);settings=load_settings(args.config)
    if args.device:settings.device=args.device
    if args.epochs:settings.epochs=args.epochs
    if args.output_dir:settings.output_dir=resolve_path(args.output_dir,Path.cwd(),"output_dir")
    _seed(settings.seed);_dirs(settings.output_dir);write_json(settings.output_dir/"config_resolved.json",settings.to_dict());write_json(settings.output_dir/"environment.json",{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]});data=load_data(settings);write_json(settings.output_dir/"dataset.json",data.metadata)
    if args.phase=="prepare_data":print(f"Fashion-MNIST ready train={len(data.train)} validation={len(data.validation)} test={len(data.test)}");return 0
    device=torch.device(settings.device);model=FashionMNISTOptical5Continuous(settings).to(device);write_json(settings.output_dir/"model.json",parameter_report(model))
    if args.phase in {"train","all"}:train_model(model,data,settings,device)
    if args.phase in {"test","all"}:metrics=test_model(model,data,settings,device);print(f"[test] top1={metrics['top1_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} region={metrics['detector_region_accuracy']:.4f}")
    return 0


def _dirs(root:Path)->None:
    for name in ("metrics","checkpoints","figures/phase_masks","figures/light_fields","figures/detector_outputs"):(root/name).mkdir(parents=True,exist_ok=True)
def _seed(seed:int)->None:random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None


if __name__=="__main__":raise SystemExit(main())
