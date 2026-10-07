from __future__ import annotations

from pathlib import Path

import yaml

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_vtab1k_fa.settings import (
    load_settings,
)


def test_settings_lock_equal_budgets_and_batch64(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    stem = tmp_path / "stem.pt"
    source.write_bytes(b"source")
    stem.write_bytes(b"stem")
    config = {
        "selection": {"task": "cifar100", "method": "bp", "seed": 2026},
        "paths": {
            "source_backbone": str(source),
            "source_backbone_sha256": "a" * 64,
            "stem_checkpoint": str(stem),
            "data_root": str(tmp_path / "data"),
            "output_root": str(tmp_path / "runs"),
        },
        "model": {"head_hidden_dim": 256},
        "optimizer": {
            "phase_learning_rate": 0.003,
            "adapter_learning_rate": 0.0002,
            "residual_learning_rate": 0.0002,
            "head_learning_rate": 0.001,
            "electronic_weight_decay": 0.0005,
            "warmup_epochs": 5,
            "minimum_learning_rate_ratio": 0.05,
        },
        "training": {
            "head_only_epochs": 50,
            "adaptation_epochs": 50,
            "use_amp": True,
            "checkpoint_interval_epochs": 50,
        },
        "dataloader": {
            "train_batch_size": 64,
            "evaluation_batch_size": 128,
            "num_workers": 8,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = load_settings(path, task="eurosat", method="fa_pretrained")
    assert settings.num_classes == 10
    assert settings.train_batch_size == 64
    assert settings.head_only_epochs == settings.adaptation_epochs == 50
    assert settings.run_dir.name == "seed_2026"
    assert settings.common_checkpoint.parts[-4:-1] == ("noft", "seed_2026", "checkpoints")
