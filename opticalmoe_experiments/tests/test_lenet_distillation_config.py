from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_cifar10_clip_lenet_distillation_config():
    path = ROOT / "foundation_distillation" / "configs" / "cifar10_gray_clip_vitb32_feature_distill_lenet.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["experiment"]["variant"] == "lenet_feature_distillation"
    assert config["student"]["model_type"] == "feature_distilled_lenet"
    assert config["student"]["feature_dim"] == 900
    assert config["lenet"]["output_feature_dim"] == 900
    assert config["lenet"]["conv_dropout2d"] == 0.1
    assert config["lenet"]["feature_dropout"] == 0.2
    assert config["dataset"]["grayscale"] is True
    assert config["projector"]["input_dim"] == 900
    assert config["projector"]["output_dim"] == "auto_teacher_dim"
    assert config["classifier"]["input"] == "semantic_feature"
    assert config["classifier"]["input_dim"] == "auto_teacher_dim"
    assert config["projector"]["dropout"] == 0.2
    assert config["classifier"]["dropout"] == 0.2
    assert "clip_vit_b32/cifar10_gray" in config["teacher_cache"]["cache_dir"]
