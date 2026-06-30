import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foundation_distillation.runtime import architecture_payload, build_student


def test_foundation_student_uses_fast_geometry_and_dynamic_detector_canvas():
    config_path = ROOT / "foundation_distillation" / "configs" / "cifar10_gray_clip_vitb32_feature_distill_moe.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["optics"]["num_layers"] = 1
    model = build_student(config, num_classes=10, teacher_feature_dim=32)
    with torch.inference_mode():
        logits, optical, projected, intermediates = model(
            torch.rand(1, 1, 120, 120), return_intermediates=True
        )
    assert logits.shape == (1, 10)
    assert optical.shape == (1, 256)
    assert projected.shape == (1, 32)
    assert intermediates["detector_intensity"].shape == (1, 520, 520)
    report = architecture_payload(model, config, "cifar10", config["teacher"])
    assert report["geometry_profile"] == "fast120_520"
    assert report["global_fc_parameter_count"] == 202500
    assert report["active_window_region"] == [35, 485, 35, 485]
