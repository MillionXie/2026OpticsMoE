import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.utils.config import save_yaml
from foundation_distillation.scripts import train_lenet_supervised as train_script


def _config():
    return {
        "seed": 7,
        "device": "cpu",
        "experiment": {"variant": "lenet_supervised", "run_name": "supervised_smoke", "print_freq": 0},
        "dataset": {"name": "cifar10", "batch_size": 2, "num_workers": 0, "pin_memory": False, "smoke_batch_size": 1},
        "student": {"model_type": "supervised_lenet", "feature_dim": 900},
        "lenet": {
            "input_channels": 1, "channels": [4, 8, 16], "output_feature_dim": 900,
            "conv_dropout2d": 0.1, "feature_dropout": 0.2,
        },
        "feature_preprocess": {"norm": "layernorm", "norm_affine": True, "activation": "gelu"},
        "classifier": {
            "input": "lenet_feature", "input_dim": 900,
            "hidden_layers": 1, "hidden_dim": 8, "activation": "gelu", "dropout": 0.2,
        },
        "optimizer": {"type": "adamw", "lr": 0.001, "weight_decay": 0.0},
        "training": {"epochs": 1, "print_freq": 0, "evaluation": {"max_val_batches": 1, "max_test_batches": 1}},
        "reporting": {"rebuild_master_tables_after_run": True},
    }


def test_supervised_lenet_smoke_does_not_use_teacher_cache(tmp_path, monkeypatch):
    dataset = TensorDataset(torch.rand(6, 1, 32, 32), torch.arange(6) % 3)
    loader = DataLoader(dataset, batch_size=2)
    bundle = SimpleNamespace(
        train_loader=loader, val_loader=loader, test_loader=loader,
        num_classes=3, class_names=["a", "b", "c"],
    )
    monkeypatch.setattr(train_script, "create_dataloaders", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(train_script, "EXPERIMENTS_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    save_yaml(_config(), config_path)
    monkeypatch.setattr(
        sys, "argv",
        ["train", "--config", str(config_path), "--run_name", "supervised_smoke", "--epochs", "1", "--smoke_test", "--device", "cpu"],
    )
    train_script.main()
    run_dir = tmp_path / "foundation_distillation" / "runs" / "supervised_smoke"
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "metrics" / "epoch_metrics.csv").is_file()
    final = json.loads((run_dir / "metrics" / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["experiment_variant"] == "lenet_supervised"
    assert final["teacher_type"] == "none"
    assert final["feature_distill_weight"] == 0.0
    assert final["projector_parameter_count"] == 0
    assert final["optical_parameter_count"] == 0
    assert (run_dir / "figures" / "training_curves.png").stat().st_size > 0
