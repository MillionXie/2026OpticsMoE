import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.datasets import DataBundle
from common.utils.config import save_yaml
from dataset_switching.scripts.train_dataset_switching import build_model
from test_dataset_switching_model import tiny_config
from transfer_adaptation.scripts import train_transfer_prompt as train_script


class Args:
    run_name = "transfer_usps_smoke"
    epochs = 1
    device = "cpu"
    smoke_test = True
    disable_visualization = True


def _source_checkpoint(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    cfg = tiny_config("learnable_route_moe")
    cfg["model"]["type"] = "learnable_route_moe"
    save_yaml(cfg, source_dir / "source_config.yaml")
    tasks = ["mnist", "fashionmnist", "emnist_letters"]
    nums = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    model = build_model(cfg, tasks, nums)
    torch.save({"model_state_dict": model.state_dict()}, source_dir / "source_best.pt")
    return source_dir


def _loader(num_classes=10):
    x = torch.rand(2, 1, 16, 16)
    y = torch.tensor([0, min(1, num_classes - 1)])
    return DataLoader(TensorDataset(x, y), batch_size=2)


def _fake_target_loaders(config, seed, smoke_test=False):
    loader = _loader(10)
    bundle = DataBundle(loader, loader, loader, 10, [str(i) for i in range(10)])
    summary = {"train_samples": 2, "val_samples": 2, "test_samples": 2}
    return bundle, summary, dict(config["target"]["dataset"])


def _fake_source_loaders(source_config, seed, smoke_test=False):
    loaders = {
        "mnist": _loader(10),
        "fashionmnist": _loader(10),
        "emnist_letters": _loader(26),
    }
    nums = {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26}
    names = {task: [str(i) for i in range(classes)] for task, classes in nums.items()}
    summaries = {task: {"test_samples": 2} for task in nums}
    return loaders, loaders, loaders, nums, names, summaries


def test_transfer_prompt_only_training_smoke(monkeypatch, tmp_path):
    source_dir = _source_checkpoint(tmp_path)
    monkeypatch.setattr(train_script.tu, "TRANSFER_ROOT", tmp_path / "transfer_adaptation")
    monkeypatch.setattr(train_script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(train_script.tu, "create_target_loaders", _fake_target_loaders)
    monkeypatch.setattr(train_script.tu, "create_source_task_loaders", _fake_source_loaders)
    cfg = {
        "seed": 7,
        "device": "cpu",
        "experiment": {"run_name": "transfer_usps_smoke", "print_freq": 0},
        "source": {
            "checkpoint_dir": str(source_dir),
            "checkpoint_name": "source_best.pt",
            "config_name": "source_config.yaml",
            "architecture_report_name": "source_architecture_report.json",
            "expected_source_tasks": ["mnist", "fashionmnist", "emnist_letters"],
        },
        "target": {
            "task_name": "usps",
            "dataset": {
                "name": "usps",
                "sampling_protocol": {"enabled": True, "total_size": 8, "train_test_ratio": [4, 1]},
            },
        },
        "transfer": {
            "method": "prompt_only",
            "init_from_source_prompt": "mnist",
            "train_target_prompt": True,
            "train_target_readout": False,
            "freeze_target_readout": True,
            "allow_extra_trainable_params": False,
        },
        "target_head": {
            "readout_type": "optical_only",
            "input_norm": "none",
            "norm_affine": False,
        },
        "optimizer": {"type": "adamw", "lr": 0.01, "weight_decay": 0.0},
        "training": {"epochs": 1, "print_freq": 0, "evaluation": {"max_val_batches": 1, "max_test_batches": 1}},
        "visualization": {"enabled": False, "num_samples": 2},
        "reporting": {"rebuild_master_tables_after_run": True},
    }
    run_dir = train_script.run_training(cfg, Args())
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "metrics" / "epoch_metrics.csv").exists()
    assert (run_dir / "metrics" / "final_target_metrics.json").exists()
    assert (run_dir / "metrics" / "target_prompt_swap.csv").exists()
    assert (run_dir / "metrics" / "source_retention.csv").exists()
    assert (run_dir / "diagnostics" / "prompt_similarity.csv").exists()
    assert (run_dir / "diagnostics" / "expert_usage.csv").exists()
    assert (run_dir / "parameter_freeze" / "trainable_parameter_names.txt").exists()
    freeze_text = (run_dir / "parameter_freeze" / "trainable_parameter_names.txt").read_text(encoding="utf-8")
    assert "prompt_bank.prompts.usps.amplitude_logits" in freeze_text
    assert (tmp_path / "transfer_adaptation" / "results" / "master_transfer_final_metrics.csv").exists()

