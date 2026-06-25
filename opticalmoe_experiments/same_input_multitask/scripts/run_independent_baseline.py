import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from common.utils.config import deep_update, load_yaml, save_json
from common.utils.filesystem import make_run_dir, write_text
from same_input_multitask.scripts.train_same_input_multitask import run_training


BASE_SHARED_D2NN_CONFIG = {
    "seed": 7,
    "device": "auto",
    "experiment": {"family": "same_input_multitask", "run_name": "independent_d2nn_task"},
    "model": {
        "type": "shared_d2nn",
        "input_size": 134,
        "canvas_size": 1000,
        "d2nn_phase_grid_size": 256,
        "d2nn_num_layers": 5,
        "prompt_type": "none",
        "routing_type": "none",
    },
    "layout": {"canvas_height": 1000, "input_size": 134},
    "optics": {
        "wavelength_m": 5.32e-7,
        "pixel_size_m": 8.0e-6,
        "num_layers": 5,
        "evanescent_mode": "zero",
        "global_fc_phase_mode": "center_window",
        "global_fc_phase_size": 600,
        "global_fc_padding_mode": "transparent",
        "phase_param": "unconstrained",
        "expert_phase_init": "identity",
        "expert_init_std": 0.02,
        "distances_m": {
            "input_to_prompt": 0.20,
            "inter_layer": 0.05,
            "layer5_to_fc": 0.05,
            "fc_to_detector": 0.05,
        },
    },
    "detector": {"detector_size": 32, "layout": "grid"},
    "readout": {
        "type": "mlp",
        "normalize_detector_energy": True,
        "logit_scale": 10.0,
        "input_norm": "layernorm",
        "norm_affine": True,
        "hidden_dim": 64,
        "hidden_layers": 1,
        "activation": "gelu",
        "dropout": 0.1,
    },
    "regularization": {
        "phase_dropout": {
            "enabled": True,
            "mode": "block_phase_bypass",
            "expert_p": 0.05,
            "global_fc_p": 0.0,
            "block_size": 8,
            "batch_shared": True,
            "apply_to_experts": True,
            "apply_to_global_fc": False,
            "start_epoch": 10,
        }
    },
    "optimizer": {"type": "adamw", "lr": 0.001, "weight_decay": 0.0005},
    "training": {
        "mode": "same_input_multitask",
        "batch_mode": "paired_same_input",
        "epochs": 200,
        "print_freq": 50,
        "tasks": ["shape"],
        "loss_weights": {"shape": 1.0},
        "evaluation": {"max_val_batches": None, "max_test_batches": None},
    },
    "dataset": {
        "name": "dsprites",
        "root": "./data",
        "input_size": 134,
        "batch_size": 64,
        "num_workers": 16,
        "pin_memory": "auto",
        "persistent_workers": True,
        "prefetch_factor": 4,
        "download": True,
        "val_split": 0.1,
        "test_split": 0.1,
        "seed": 7,
        "smoke_test": False,
        "smoke_train_size": 64,
        "smoke_test_size": 32,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_test_samples": None,
        "sampling_protocol": {
            "enabled": True,
            "total_size": 12000,
            "train_test_ratio": [4, 1],
            "class_balanced": False,
            "seed_offset": 0,
        },
    },
    "visualization": {"enabled": True, "save_interval_epochs": 10, "num_samples": 4, "dpi": 150},
    "reporting": {"rebuild_master_tables_after_run": False},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default="independent_d2nn_dsprites_same_input_seed7")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    args = parser.parse_args()

    config = deep_update(BASE_SHARED_D2NN_CONFIG, load_yaml(args.config))
    tasks = [str(task).lower() for task in config.get("training", {}).get("tasks", ["shape", "scale"])]
    parent_dir = make_run_dir(EXPERIMENT_ROOT, "same_input_multitask", args.run_name)
    write_text(parent_dir / "README.txt", "Independent D2NN baseline: one D2NN per task. This is not an upper bound.\n")
    child_summaries = []
    loader_summaries = {}
    for task in tasks:
        child_config = deep_update(BASE_SHARED_D2NN_CONFIG, config)
        child_config["model"]["type"] = "shared_d2nn"
        child_config["training"]["tasks"] = [task]
        child_config["training"]["loss_weights"] = {task: 1.0}
        child_config.setdefault("experiment", {})["run_name"] = f"{args.run_name}_{task}"
        child_config.setdefault("reporting", {})["rebuild_master_tables_after_run"] = False
        child_args = SimpleNamespace(
            run_name=f"{args.run_name}_{task}",
            device=args.device,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
            disable_visualization=args.disable_visualization,
        )
        run_dir = run_training(child_config, child_args)
        loader_path = Path(run_dir) / "loader_summary.json"
        loader_summary = json.loads(loader_path.read_text(encoding="utf-8")) if loader_path.exists() else {}
        loader_summaries[task] = loader_summary
        child_summaries.append({"task_name": task, "independent_run_dir": str(run_dir), "loader_summary": loader_summary})
    parent_summary = {
        "run_id": args.run_name,
        "baseline_type": "independent_d2nn",
        "not_upper_bound": True,
        "tasks": tasks,
        "children": child_summaries,
        "loader_summary": loader_summaries,
    }
    save_json(parent_summary, parent_dir / "summary.json")
    save_json(loader_summaries, parent_dir / "loader_summary.json")
    save_json(
        {
            "run_id": args.run_name,
            "model_type": "independent_d2nn",
            "tasks": ",".join(tasks),
            "loader_summary": loader_summaries,
            "run_dir": str(parent_dir),
        },
        parent_dir / "summary_for_master" / "runs_rows.json",
    )
    print(f"saved independent baseline summary to {parent_dir}")


if __name__ == "__main__":
    main()
