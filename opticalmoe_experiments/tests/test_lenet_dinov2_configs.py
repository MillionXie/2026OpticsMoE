from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_cifar10_lenet_dinov2_small_and_base_configs():
    expected = {
        "cifar10_gray_dinov2_vits14_feature_distill_lenet.yaml": ("facebook/dinov2-small", "dinov2_small"),
        "cifar10_gray_dinov2_vitb14_feature_distill_lenet.yaml": ("facebook/dinov2-base", "dinov2_base"),
    }
    for filename, (model_name, cache_token) in expected.items():
        path = ROOT / "foundation_distillation" / "configs" / filename
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["teacher"]["type"] == "dinov2_image_encoder"
        assert config["teacher"]["model_name"] == model_name
        assert cache_token in config["teacher_cache"]["cache_dir"]
        assert config["student"]["model_type"] == "feature_distilled_lenet"
        assert config["lenet"]["output_feature_dim"] == 900
        assert config["lenet"]["conv_dropout2d"] == 0.1
        assert config["lenet"]["feature_dropout"] == 0.2
        assert config["projector"]["dropout"] == 0.2
        assert config["classifier"]["dropout"] == 0.2
