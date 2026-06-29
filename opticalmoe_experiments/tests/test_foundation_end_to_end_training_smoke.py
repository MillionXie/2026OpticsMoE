import sys
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.utils.config import save_yaml
from foundation_distillation.scripts import train_end_to_end_moe as train_script


def _config():
    return {
        "seed": 7,
        "device": "cpu",
        "experiment": {"run_name": "end_to_end_smoke", "print_freq": 0},
        "dataset": {"name": "cifar10", "batch_size": 2, "smoke_batch_size": 1, "num_workers": 0, "pin_memory": False},
        "student": {"model_type": "end_to_end_optical_moe", "num_experts": 9},
        "layout": {"canvas_height": 96, "canvas_width": 96, "input_size": 16, "expert_size": 10, "expert_pitch": 24, "padding": 12, "prompt_aperture_size": 72},
        "optics": {"num_layers": 1, "global_fc_phase_size": 72, "distances_m": {key: 0.01 for key in ("input_to_prompt", "prompt_to_expert", "inter_layer", "layer5_to_fc", "fc_to_detector")}},
        "prompt": {},
        "feature_detector": {"grid_size": 4, "feature_dim": 16},
        "classifier": {"hidden_dim": 8, "hidden_layers": 1},
        "loss": {"type": "cross_entropy"},
        "optimizer": {"type": "adamw", "lr": 0.001},
        "regularization": {"phase_dropout": {"enabled": False}},
        "training": {"epochs": 1, "print_freq": 0, "evaluation": {"max_val_batches": 1, "max_test_batches": 1}},
        "visualization": {"enabled": False, "save_interval_epochs": 10, "num_samples": 2},
        "reporting": {"rebuild_master_tables_after_run": True},
    }


def test_end_to_end_smoke_saves_comparable_outputs(tmp_path, monkeypatch):
    dataset = TensorDataset(torch.rand(6, 1, 16, 16), torch.arange(6) % 10)
    loader = DataLoader(dataset, batch_size=2)
    bundle = SimpleNamespace(
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        num_classes=10,
        class_names=[str(index) for index in range(10)],
    )
    monkeypatch.setattr(train_script, "create_dataloaders", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(train_script, "EXPERIMENTS_ROOT", tmp_path)
    config_path = tmp_path / "config.yaml"
    save_yaml(_config(), config_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--config", str(config_path), "--run_name", "end_to_end_smoke", "--epochs", "1", "--smoke_test", "--device", "cpu"],
    )
    train_script.main()
    run_dir = tmp_path / "foundation_distillation" / "runs" / "end_to_end_smoke"
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "metrics" / "epoch_metrics.csv").is_file()
    assert (run_dir / "figures" / "training_curves.png").stat().st_size > 0
    assert (run_dir / "summary.json").is_file()
    architecture = (run_dir / "architecture_report.json").read_text(encoding="utf-8")
    assert '"projector_parameter_count": 0' in architecture
    assert '"feature_distillation_used": false' in architecture

