from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dinov2_base_configs_exist_and_use_separate_caches():
    for dataset in ("cifar10", "imagenette"):
        path = ROOT / "foundation_distillation" / "configs" / f"{dataset}_gray_dinov2_vitb14_feature_distill_moe.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["teacher"]["model_name"] == "facebook/dinov2-base"
        assert config["teacher"]["feature_type"] == "cls"
        assert "dinov2_base" in config["teacher_cache"]["cache_dir"]
