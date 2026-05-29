import argparse
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opticalmoe.data import create_dataloaders
from opticalmoe.optics import OpticalClassifier
from opticalmoe.training import fit, load_checkpoint
from opticalmoe.utils import cm_to_m, load_config, nm_to_m, save_json, set_seed, um_to_m
from opticalmoe.utils.parameters import count_trainable_parameters
from opticalmoe.utils.run import create_run_dir
from opticalmoe.visualization import save_detector_layout


def build_model(config, num_classes: int) -> OpticalClassifier:
    optics_cfg = config["optics"]
    detector_cfg = config.get("detector", {})
    readout_cfg = config.get("readout", {})
    distances_cm = optics_cfg["distances_cm"]
    distances_m = {
        "input_to_prompt": cm_to_m(distances_cm["input_to_prompt"]),
        "prompt_to_first_layer": cm_to_m(distances_cm["prompt_to_first_layer"]),
        "inter_layer": cm_to_m(distances_cm["inter_layer"]),
        "last_layer_to_detector": cm_to_m(distances_cm["last_layer_to_detector"]),
    }

    return OpticalClassifier(
        num_classes=num_classes,
        wavelength_m=nm_to_m(optics_cfg.get("wavelength_nm", 532.0)),
        pixel_size_m=um_to_m(optics_cfg.get("pixel_size_um", 8.0)),
        input_size=int(optics_cfg.get("input_size", 200)),
        padding=int(optics_cfg.get("padding", 200)),
        grid_size=int(optics_cfg.get("grid_size", 600)),
        num_layers=int(optics_cfg.get("num_layers", 5)),
        distances_m=distances_m,
        phase_param=optics_cfg.get("phase_param", "unconstrained"),
        phase_init=optics_cfg.get("phase_init", "uniform"),
        detector_size=int(detector_cfg.get("detector_size", 32)),
        detector_layout=detector_cfg.get("layout", "grid"),
        readout_type=readout_cfg.get("type", "optical_only"),
        normalize_detector_energy=bool(readout_cfg.get("normalize_detector_energy", True)),
        logit_scale=float(readout_cfg.get("logit_scale", 10.0)),
        readout_hidden_dim=int(readout_cfg.get("hidden_dim", 64)),
        readout_activation=readout_cfg.get("activation", "relu"),
        evanescent_mode=optics_cfg.get("evanescent_mode", "zero"),
    )


def get_fixed_batch(loader):
    for batch in loader:
        return batch
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Train an OpticalMoE optical classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run_name", required=True, help="Run directory name under runs/.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint, usually runs/<run>/last.pt.")
    parser.add_argument("--lr", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="When resuming, load model weights but start with a fresh optimizer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    set_seed(seed)

    run_dir = create_run_dir(args.run_name, base_dir=str(PROJECT_ROOT / "runs"))
    shutil.copyfile(args.config, run_dir / "config.yaml")

    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        config["dataset"], seed=seed
    )

    device_cfg = config.get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    model = build_model(config, num_classes=num_classes).to(device)
    optimizer_cfg = config.get("optimizer", {})
    lr = float(optimizer_cfg.get("lr", 1e-3))
    if args.lr is not None:
        lr = float(args.lr)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )

    start_epoch = 1
    best_val_acc = -1.0
    best_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer=None if args.reset_optimizer else optimizer,
            map_location=str(device),
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        metrics = checkpoint.get("metrics", {})
        best_val_acc = float(metrics.get("best_val_acc", metrics.get("val_acc", -1.0)))
        best_epoch = int(metrics.get("best_epoch", checkpoint.get("epoch", 0)))
        if args.lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        print(f"resumed checkpoint: {args.resume}")
        print(f"continuing from epoch {start_epoch}")
        print(f"optimizer lr: {optimizer.param_groups[0]['lr']:.3e}")

    print(f"device: {device}")
    print(f"num_classes: {num_classes}")
    print(f"optical parameters: {model.optical_parameter_count()}")
    print(f"electronic readout parameters: {model.electronic_parameter_count()}")
    print(f"total trainable parameters: {count_trainable_parameters(model)}")
    print(f"propagation segments: {model.num_propagation_segments}")

    save_detector_layout(model.detector, str(run_dir / "detector_layout.png"))

    fixed_vis_batch = get_fixed_batch(val_loader)
    results = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        run_dir=run_dir,
        num_epochs=int(config.get("training", {}).get("epochs", 1)),
        num_classes=num_classes,
        visualization_cfg=config.get("visualization", {}),
        fixed_vis_batch=fixed_vis_batch,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
    )

    optics_cfg = config["optics"]
    summary = {
        "dataset_name": config["dataset"].get("name", "mnist"),
        "best_validation_accuracy": results["best_val_acc"],
        "best_epoch": results["best_epoch"],
        "final_test_accuracy": results["final_test_acc"],
        "final_test_loss": results["final_test_loss"],
        "optical_parameter_count": model.optical_parameter_count(),
        "electronic_parameter_count": model.electronic_parameter_count(),
        "total_parameter_count": count_trainable_parameters(model),
        "num_optical_layers": optics_cfg.get("num_layers", 5),
        "wavelength_nm": optics_cfg.get("wavelength_nm", 532.0),
        "pixel_size_um": optics_cfg.get("pixel_size_um", 8.0),
        "input_size": optics_cfg.get("input_size", 200),
        "padding": optics_cfg.get("padding", 200),
        "simulation_grid_size": optics_cfg.get("grid_size", 600),
        "propagation_distances_cm": optics_cfg.get("distances_cm", {}),
        "phase_parameterization": optics_cfg.get("phase_param", "unconstrained"),
        "readout_type": config.get("readout", {}).get("type", "optical_only"),
        "seed": seed,
        "run_name": args.run_name,
    }
    save_json(summary, str(run_dir / "summary.json"))
    print(f"saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
