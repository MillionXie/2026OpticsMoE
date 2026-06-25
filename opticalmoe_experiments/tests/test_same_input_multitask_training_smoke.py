import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import same_input_multitask.scripts.train_same_input_multitask as train_script


class PairedToyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.rand(1, 16, 16), {"shape": torch.tensor(index % 3), "scale": torch.tensor(index % 6)}


class Args:
    run_name = "same_input_smoke"
    epochs = 1
    device = "cpu"
    smoke_test = True
    disable_visualization = True


def _fake_loaders(config, seed):
    loader = DataLoader(PairedToyDataset(), batch_size=2)
    return loader, loader, loader, {"shape": 3, "scale": 6}, ["shape", "scale"]


def _config():
    return {
        "seed": 7,
        "device": "cpu",
        "experiment": {"run_name": "same_input_smoke"},
        "model": {"type": "learnable_route_moe", "num_experts": 9},
        "layout": {"canvas_height": 128, "input_size": 16, "expert_size": 12, "expert_pitch": 30, "padding": 19, "prompt_aperture_size": 90},
        "optics": {"num_layers": 1, "focal_length_m": 0.01, "distances_m": {"input_to_prompt": 0.01, "prompt_to_expert": 0.01, "inter_layer": 0.01, "layer5_to_fc": 0.01, "fc_to_detector": 0.01}},
        "prompt": {"mode": "complex_order_router", "train_amplitudes": True, "train_phase_biases": True},
        "detector": {"detector_size": 4, "layout": "grid"},
        "readout": {"type": "linear", "normalize_detector_energy": True},
        "regularization": {"phase_dropout": {"enabled": False}},
        "optimizer": {"type": "adamw", "lr": 0.001, "weight_decay": 0.0},
        "training": {
            "mode": "same_input_multitask",
            "epochs": 1,
            "print_freq": 0,
            "tasks": ["shape", "scale"],
            "loss_weights": {"shape": 1.0, "scale": 1.0},
            "evaluation": {"max_val_batches": 1, "max_test_batches": 1},
        },
        "visualization": {"enabled": False, "num_samples": 2, "save_interval_epochs": 1},
        "reporting": {"rebuild_master_tables_after_run": True},
        "dataset": {"smoke_test": True},
    }


def test_same_input_training_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(train_script, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(train_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(train_script, "create_same_input_multitask_dataloaders", _fake_loaders)
    run_dir = train_script.run_training(_config(), Args())
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "metrics" / "task_metrics.csv").exists()
    assert (run_dir / "metrics" / "same_input_task_switching.csv").exists()
    assert (run_dir / "metrics" / "prompt_swap_matrix.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary_for_master" / "scaling_results_rows.json").exists()
