import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset_switching.scripts.train_dataset_switching as train_script
from test_dataset_switching_model import tiny_config


class Args:
    run_name = "dataset_switching_smoke"
    epochs = 1
    device = "cpu"
    smoke_test = True
    disable_visualization = True


def _fake_loaders(config, seed, smoke_test):
    loaders = {}
    nums = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    names = {}
    for task, classes in nums.items():
        x = torch.rand(2, 1, 16, 16)
        y = torch.tensor([0, min(1, classes - 1)])
        loaders[task] = DataLoader(TensorDataset(x, y), batch_size=2)
        names[task] = [str(i) for i in range(classes)]
    return loaders, loaders, loaders, nums, names


def test_dataset_switching_training_smoke(monkeypatch, tmp_path):
    cfg = tiny_config("learnable_route_moe")
    cfg["experiment"] = {"run_name": "dataset_switching_smoke"}
    cfg["training"]["epochs"] = 1
    cfg["training"]["multitask"].update({"steps_per_epoch": 1, "loss_weights": {"mnist": 1.0, "fashionmnist": 1.0, "emnist_letters": 1.0}})
    cfg["training"]["evaluation"] = {"max_val_batches": 1, "max_test_batches": 1}
    cfg["visualization"] = {"enabled": False}
    cfg["optimizer"] = {"type": "adamw", "lr": 0.001, "weight_decay": 0.0}
    monkeypatch.setattr(train_script, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(train_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(train_script, "create_task_loaders", _fake_loaders)
    run_dir = train_script.run_training(cfg, Args())
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "metrics" / "task_metrics.csv").exists()
    assert (run_dir / "metrics" / "prompt_swap_matrix.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary_for_master" / "prompt_swap_rows.json").exists()
