import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.utils.config import save_yaml
from foundation_distillation.scripts import train_teacher_feature_probe as probe_script


def test_matched_teacher_probe_smoke(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    metadata = {
        "teacher_type": "clip_image_encoder", "teacher_model_name": "mock", "feature_type": "image_embedding",
        "teacher_feature_dim": 12, "class_names": ["a", "b", "c"],
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for split, count in (("train", 12), ("val", 6), ("test", 6)):
        torch.save(
            {"features": torch.randn(count, 12), "labels": torch.arange(count) % 3, "indices": torch.arange(count), "split": split},
            cache_dir / f"{split}_features.pt",
        )
    config = {
        "seed": 7, "device": "cpu",
        "dataset": {"name": "cifar10", "batch_size": 3, "num_workers": 0, "pin_memory": False},
        "teacher": {"type": "clip_image_encoder", "model_name": "mock"},
        "teacher_cache": {"cache_dir": str(cache_dir)},
        "classifier": {"hidden_layers": 1, "hidden_dim": 128, "activation": "gelu", "dropout": 0.1},
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
        "training": {"epochs": 1},
    }
    config_path = tmp_path / "probe.yaml"
    save_yaml(config, config_path)
    monkeypatch.setattr(probe_script, "EXPERIMENTS_ROOT", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["probe", "--config", str(config_path), "--probe_type", "matched_mlp", "--run_name", "probe_smoke", "--epochs", "1", "--smoke_test", "--device", "cpu"]
    )
    probe_script.main()
    run_dir = tmp_path / "foundation_distillation" / "teacher_probe_runs" / "probe_smoke"
    final = json.loads((run_dir / "metrics" / "final_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert final["probe_type"] == "matched_mlp"
    assert final["teacher_feature_dim"] == 12
    assert summary["probe_classifier_config"]["hidden_dim"] == 128
    assert (run_dir / "figures" / "training_curves.png").stat().st_size > 0
    assert (tmp_path / "foundation_distillation" / "results" / "master_teacher_probe_final_metrics.csv").is_file()
