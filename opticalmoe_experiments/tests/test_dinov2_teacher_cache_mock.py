import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.foundation_distillation import IndexedGrayDataset
from common.utils.config import save_yaml
from foundation_distillation.scripts import build_teacher_feature_cache as cache_script
from foundation_distillation.scripts import train_feature_distilled_moe as train_script


class MockTeacher(torch.nn.Module):
    def forward(self, images):
        values = images.mean(dim=(-2, -1))
        return torch.nn.functional.normalize(torch.cat([values, values[:, :1]], dim=1), dim=-1)


def _indexed_dataset(count=6):
    return IndexedGrayDataset(TensorDataset(torch.rand(count, 1, 16, 16), torch.arange(count) % 3))


def test_mock_dinov2_cache_writes_backend_metadata(tmp_path, monkeypatch):
    datasets = SimpleNamespace(
        train_dataset=_indexed_dataset(),
        val_dataset=_indexed_dataset(4),
        test_dataset=_indexed_dataset(4),
        class_names=["0", "1", "2"],
    )
    monkeypatch.setattr(cache_script, "create_distillation_datasets", lambda *args, **kwargs: datasets)
    monkeypatch.setattr(
        cache_script,
        "load_frozen_image_teacher",
        lambda cfg, device: (
            MockTeacher(),
            {
                "teacher_type": "dinov2_image_encoder",
                "teacher_backend": "transformers",
                "teacher_model_name": "facebook/dinov2-small",
                "feature_type": "cls",
                "input_mode": "grayscale_replicated_rgb",
                "teacher_text_encoder_used": False,
                "teacher_image_size": 224,
                "teacher_image_mean": [0.485, 0.456, 0.406],
                "teacher_image_std": [0.229, 0.224, 0.225],
            },
        ),
    )
    config = {
        "seed": 7,
        "dataset": {"name": "cifar10", "batch_size": 2, "num_workers": 0, "input_size": 120},
        "teacher": {
            "type": "dinov2_image_encoder",
            "backend": "transformers",
            "model_name": "facebook/dinov2-small",
            "feature_type": "cls",
            "input_mode": "grayscale_replicated_rgb",
            "freeze": True,
        },
        "teacher_cache": {"cache_dir": str(tmp_path / "cache")},
    }
    metadata = cache_script.build_cache(config, torch.device("cpu"))
    assert metadata["teacher_type"] == "dinov2_image_encoder"
    assert metadata["teacher_backend"] == "transformers"
    assert metadata["feature_type"] == "cls"
    assert metadata["teacher_text_encoder_used"] is False
    assert metadata["features_are_l2_normalized"] is True
    assert metadata["teacher_feature_dim"] == 4
    for split in ("train", "val", "test"):
        payload = torch.load(tmp_path / "cache" / f"{split}_features.pt", weights_only=False)
        assert payload["split"] == split
        assert torch.allclose(payload["features"].norm(dim=-1), torch.ones(len(payload["features"])))


def _training_config(cache_dir):
    return {
        "seed": 7,
        "device": "cpu",
        "experiment": {"run_name": "dino_smoke", "print_freq": 0},
        "dataset": {"name": "cifar10", "batch_size": 2, "num_workers": 0, "pin_memory": False},
        "teacher": {"type": "dinov2_image_encoder", "backend": "transformers", "model_name": "facebook/dinov2-small", "feature_type": "cls", "input_mode": "grayscale_replicated_rgb", "freeze": True},
        "teacher_cache": {"cache_dir": str(cache_dir), "require_metadata_match": False},
        "student": {"model_type": "feature_distilled_optical_moe", "num_experts": 9},
        "layout": {"canvas_height": 96, "canvas_width": 96, "input_size": 16, "expert_size": 10, "expert_pitch": 24, "padding": 12, "prompt_aperture_size": 72},
        "optics": {"num_layers": 1, "global_fc_phase_size": 72, "distances_m": {key: 0.01 for key in ("input_to_prompt", "prompt_to_expert", "inter_layer", "layer5_to_fc", "fc_to_detector")}},
        "prompt": {},
        "feature_detector": {"grid_size": 4, "feature_dim": 16},
        "classifier": {"hidden_dim": 8, "hidden_layers": 1},
        "projector": {"hidden_dim": 8, "hidden_layers": 1},
        "loss": {"ce_weight": 1.0, "feature_distill_weight": 0.5},
        "optimizer": {"type": "adamw", "lr": 0.001},
        "regularization": {"phase_dropout": {"enabled": False}},
        "training": {"epochs": 1, "print_freq": 0, "evaluation": {"max_val_batches": 1, "max_test_batches": 1}},
        "visualization": {"enabled": False, "save_interval_epochs": 10, "num_samples": 2},
        "reporting": {"rebuild_master_tables_after_run": True},
    }


def test_dinov2_cache_reuses_training_and_projector_dimension(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "metadata.json").write_text("{}", encoding="utf-8")
    dataset = TensorDataset(torch.rand(6, 1, 16, 16), torch.arange(6) % 10, torch.randn(6, 12), torch.arange(6))
    loader = DataLoader(dataset, batch_size=2)
    bundle = SimpleNamespace(
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        num_classes=10,
        class_names=[str(i) for i in range(10)],
        teacher_feature_dim=12,
        teacher_metadata={
            "teacher_type": "dinov2_image_encoder",
            "teacher_backend": "transformers",
            "feature_type": "cls",
            "teacher_feature_dim": 12,
        },
    )
    monkeypatch.setattr(train_script, "create_cached_distillation_loaders", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(train_script, "EXPERIMENTS_ROOT", tmp_path)
    captured = {}
    real_builder = train_script.build_student

    def build_and_capture(config, num_classes, teacher_feature_dim):
        model = real_builder(config, num_classes, teacher_feature_dim)
        captured["teacher_feature_dim"] = model.teacher_feature_dim
        captured["projector_output_dim"] = model.projector[-1].out_features
        return model

    monkeypatch.setattr(train_script, "build_student", build_and_capture)
    config_path = tmp_path / "config.yaml"
    save_yaml(_training_config(cache_dir), config_path)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), "--run_name", "dino_smoke", "--epochs", "1", "--smoke_test", "--device", "cpu"])
    train_script.main()
    run_dir = tmp_path / "foundation_distillation" / "runs" / "dino_smoke"
    assert captured == {"teacher_feature_dim": 12, "projector_output_dim": 12}
    assert (run_dir / "metrics" / "epoch_metrics.csv").is_file()
    resolved = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    assert resolved["teacher"]["resolved_backend"] == "transformers"
    assert resolved["teacher"]["resolved_feature_type"] == "cls"
    architecture = json.loads((run_dir / "architecture_report.json").read_text(encoding="utf-8"))
    assert architecture["teacher_backend"] == "transformers"
    assert architecture["teacher_feature_dim"] == 12
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["teacher_type"] == "dinov2_image_encoder"
    assert summary["teacher_backend"] == "transformers"
    assert summary["feature_type"] == "cls"
    master = (tmp_path / "foundation_distillation" / "results" / "master_distillation_final_metrics.csv").read_text(encoding="utf-8")
    assert "dinov2_image_encoder" in master
