import argparse
import sys
import time
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common.data.datasets import create_dataloaders
from common.data.loader_utils import apply_smoke_loader_overrides, loader_summary_from_loaders, print_loader_summary
from common.optics.optical_models import GeneralD2NNClassifier
from common.reporting.metrics_writer import write_rows
from common.training.checkpointing import save_checkpoint
from common.training.eval_loop import evaluate
from common.utils.config import load_yaml, save_json, save_yaml
from common.utils.filesystem import make_run_dir, write_text
from common.utils.seed import choose_device, set_seed
from dataset_switching.scripts.train_dataset_switching import build_optimizer, rebuild_dataset_switching_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.run_name:
        config.setdefault("experiment", {})["run_name"] = args.run_name
    seed = int(config.get("seed", 7))
    set_seed(seed)
    device = choose_device(args.device)
    run_name = config.get("experiment", {}).get("run_name", f"independent_d2nn_{int(time.time())}")
    run_dir = make_run_dir(EXPERIMENT_ROOT, "dataset_switching", run_name)
    save_yaml(config, run_dir / "config.yaml")
    write_text(run_dir / "command.txt", " ".join(sys.argv))

    model_cfg = config.get("model", {})
    optics = config.get("optics", {})
    detector = config.get("detector", {})
    readout = config.get("readout", {})
    task_rows = []
    loader_summaries = {}
    total_optical = total_electronic = total_params = 0
    epochs = int(args.epochs or config.get("training", {}).get("epochs", 1))
    if args.smoke_test:
        epochs = min(epochs, 1)
    for idx, task in enumerate(config.get("training", {}).get("tasks", [])):
        task_name = str(task["name"]).lower()
        dataset_cfg = dict(task["dataset"])
        if args.smoke_test:
            dataset_cfg["smoke_test"] = True
            dataset_cfg.setdefault("smoke_train_size", 16)
            dataset_cfg.setdefault("smoke_test_size", 8)
            apply_smoke_loader_overrides(dataset_cfg)
        bundle = create_dataloaders(dataset_cfg, seed + idx)
        loader_summaries[task_name] = loader_summary_from_loaders(bundle.train_loader, bundle.val_loader, bundle.test_loader, dataset_cfg)
        print_loader_summary(loader_summaries[task_name], prefix=f"loader/{task_name}")
        model = GeneralD2NNClassifier(
            num_classes=bundle.num_classes,
            canvas_size=int(model_cfg.get("canvas_size", 1000)),
            input_size=int(model_cfg.get("input_size", 134)),
            d2nn_phase_grid_size=int(model_cfg.get("d2nn_phase_grid_size", 256)),
            num_layers=int(model_cfg.get("d2nn_num_layers", 5)),
            wavelength_m=float(optics.get("wavelength_m", 532e-9)),
            pixel_size_m=float(optics.get("pixel_size_m", 8e-6)),
            distances_m=optics.get("distances_m", {}),
            phase_param=optics.get("phase_param", "unconstrained"),
            phase_init=optics.get("expert_phase_init", "identity"),
            init_std=float(optics.get("expert_init_std", 0.02)),
            global_fc_phase_mode=optics.get("global_fc_phase_mode", "center_window"),
            global_fc_phase_size=optics.get("global_fc_phase_size"),
            global_fc_padding_mode=optics.get("global_fc_padding_mode", "transparent"),
            global_fc_phase_dropout_mode="none",
            global_fc_phase_dropout_p=0.0,
            detector_size=int(detector.get("detector_size", 32)),
            detector_layout=detector.get("layout", "grid"),
            normalize_detector_energy=bool(readout.get("normalize_detector_energy", True)),
            readout_type=readout.get("type", "mlp"),
            readout_hidden_dim=int(readout.get("hidden_dim", 64)),
            readout_activation=readout.get("activation", "gelu"),
            readout_input_norm=readout.get("input_norm", "layernorm"),
            readout_dropout=float(readout.get("dropout", 0.1)),
            evanescent_mode=optics.get("evanescent_mode", "zero"),
        ).to(device)
        opt = build_optimizer(model, config)
        criterion = torch.nn.CrossEntropyLoss()
        for _epoch in range(epochs):
            model.train()
            for images, targets in bundle.train_loader:
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(images.to(device)), targets.to(device))
                loss.backward()
                opt.step()
        test = evaluate(model, bundle.test_loader, criterion, device, max_batches=1 if args.smoke_test else None)
        task_run_dir = run_dir / task_name
        save_checkpoint(task_run_dir / "last.pt", model, opt, epochs, test, config)
        optical = int(model.optical_parameter_count())
        electronic = int(model.electronic_parameter_count())
        total = int(sum(p.numel() for p in model.parameters()))
        total_optical += optical
        total_electronic += electronic
        total_params += total
        task_rows.append(
            {
                "run_id": run_name,
                "task_name": task_name,
                "dataset_name": task_name,
                "independent_run_dir": str(task_run_dir),
                "test_acc": test["acc"],
                "test_loss": test["loss"],
                "optical_parameter_count": optical,
                "electronic_parameter_count": electronic,
                "total_parameter_count": total,
                "is_upper_bound": False,
            }
        )
    reference = model_cfg.get("moe_reference_optical_params") or ""
    for row in task_rows:
        row["total_independent_optical_params"] = total_optical
        row["total_independent_electronic_params"] = total_electronic
        row["total_independent_params"] = total_params
        row["moe_reference_optical_params"] = reference
        row["param_ratio_to_moe"] = float(total_optical) / float(reference) if reference not in {"", None, 0} else ""
    write_rows(run_dir / "metrics" / "independent_baseline.csv", task_rows)
    save_json(loader_summaries, run_dir / "loader_summary.json")
    save_json(task_rows, run_dir / "summary_for_master" / "independent_baseline_rows.json")
    runs_row = {
        "run_id": run_name,
        "model_type": "independent_d2nn",
        "independent_baseline_is_upper_bound": False,
        "tasks": ",".join(row["task_name"] for row in task_rows),
        "loader_summary": loader_summaries,
        "run_dir": str(run_dir),
    }
    save_json({"run_id": run_name, "independent_baseline_is_upper_bound": False, "rows": task_rows, "loader_summary": loader_summaries}, run_dir / "summary.json")
    save_json(runs_row, run_dir / "summary_for_master" / "runs_rows.json")
    if bool(config.get("reporting", {}).get("rebuild_master_tables_after_run", True)):
        rebuild_dataset_switching_tables(EXPERIMENT_ROOT / "dataset_switching" / "runs", EXPERIMENT_ROOT / "dataset_switching" / "results")
    print(f"saved independent baseline to {run_dir}")


if __name__ == "__main__":
    main()
