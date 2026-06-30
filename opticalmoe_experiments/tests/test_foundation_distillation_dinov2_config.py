import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_dinov2_configs_use_fast_student_geometry():
    names = (
        "cifar10_gray_dinov2_vits14_feature_distill_moe.yaml",
        "imagenette_gray_dinov2_vits14_feature_distill_moe.yaml",
    )
    for name in names:
        path = ROOT / "foundation_distillation" / "configs" / name
        assert path.is_file()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        teacher = config["teacher"]
        assert teacher["type"] == "dinov2_image_encoder"
        assert teacher["backend"] == "transformers"
        assert teacher["model_name"] == "facebook/dinov2-small"
        assert teacher["feature_type"] == "cls"
        assert teacher["input_mode"] == "grayscale_replicated_rgb"
        assert "dinov2_small" in config["teacher_cache"]["cache_dir"]
        assert config["dataset"]["input_size"] == 120
        assert config["layout"]["geometry_profile"] == "fast120_520"
        assert config["layout"]["canvas_height"] == 520
        assert config["layout"]["expert_size"] == 120
        assert config["layout"]["expert_pitch"] == 150
        assert config["layout"]["prompt_aperture_size"] == 450
        assert config["optics"]["global_fc_phase_size"] == 450
