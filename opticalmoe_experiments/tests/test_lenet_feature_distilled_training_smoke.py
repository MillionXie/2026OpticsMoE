import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.utils.config import save_yaml
from foundation_distillation.scripts import train_lenet_feature_distilled as train_script


def _config(cache_dir):
    return {
        "seed": 7,
        "device": "cpu",
        "experiment": {"variant": "lenet_feature_distillation", "run_name": "lenet_smoke", "print_freq": 0},
        "dataset": {"name": "cifar10", "batch_size": 2, "num_workers": 0, "pin_memory": False, "smoke_batch_size": 1},
        "teacher": {"type": "clip_image_encoder", "model_name": "mock", "input_mode": "grayscale_replicated_rgb"},
        "teacher_cache": {"cache_dir": str(cache_dir), "require_metadata_match": False},
        "student": {"model_type": "feature_distilled_lenet", "feature_dim": 900},
        "lenet": {"input_channels": 1, "channels": [4, 8, 16], "activation": "gelu", "pooling": "avg", "adaptive_pool_size": 5, "output_feature_dim": 900},
        "feature_preprocess": {"norm": "layernorm", "norm_affine": True, "activation": "gelu"},
        "projector": {"type": "mlp", "input_dim": 900, "output_dim": "auto_teacher_dim", "hidden_layers": 1, "hidden_dim": 16, "dropout": 0.0, "output_l2_normalize": True},
        "classifier": {"input": "semantic_feature", "input_dim": "auto_teacher_dim", "hidden_layers": 1, "hidden_dim": 8, "activation": "gelu", "dropout": 0.0},
        "loss": {"ce_weight": 1.0, "feature_distill_weight": 0.5, "leak_loss_weight": 0.0},
        "optimizer": {"type": "adamw", "lr": 0.001, "weight_decay": 0.0},
        "training": {"epochs": 1, "print_freq": 0, "evaluation": {"max_val_batches": 1, "max_test_batches": 1}},
        "reporting": {"rebuild_master_tables_after_run": True},
    }


def test_lenet_feature_distillation_smoke(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "metadata.json").write_text("{}", encoding="utf-8")
    dataset = TensorDataset(
        torch.rand(6, 1, 32, 32), torch.arange(6) % 3, torch.randn(6, 12), torch.arange(6)
    )
    loader = DataLoader(dataset, batch_size=2)
    bundle = SimpleNamespace(
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        num_classes=3,
        class_names=["a", "b", "c"],
        teacher_feature_dim=12,
        teacher_metadata={"teacher_type": "clip_image_encoder", "teacher_feature_dim": 12},
    )
    monkeypatch.setattr(train_script, "create_cached_distillation_loaders", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(train_script, "EXPERIMENTS_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    save_yaml(_config(cache_dir), config_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", str(config_path), "--run_name", "lenet_smoke", "--epochs", "1", "--smoke_test", "--device", "cpu"],
    )
    train_script.main()
    run_dir = tmp_path / "foundation_distillation" / "runs" / "lenet_smoke"
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "metrics" / "epoch_metrics.csv").is_file()
    final = json.loads((run_dir / "metrics" / "final_metrics.json").read_text(encoding="utf-8"))
    assert final["student_model_type"] == "feature_distilled_lenet"
    assert final["student_feature_dim"] == 900
    assert final["optical_parameter_count"] == 0
    assert final["lenet_parameter_count"] > 0
    assert (run_dir / "figures" / "training_curves.png").stat().st_size > 0
    assert (run_dir / "summary.json").is_file()
    master = tmp_path / "foundation_distillation" / "results" / "master_distillation_final_metrics.csv"
    assert "feature_distilled_lenet" in master.read_text(encoding="utf-8")
