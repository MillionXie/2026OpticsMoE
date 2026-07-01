import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from foundation_distillation.scripts.train_feature_distilled_moe import _save_artifacts
from test_camera_crop_feature_detector import _model


def test_lightfield_artifacts_include_input_and_label(tmp_path):
    model = _model()
    fixed_batch = (
        torch.rand(1, 1, 16, 16),
        torch.tensor([1]),
        torch.randn(1, 8),
        torch.tensor([42]),
    )
    _save_artifacts(model, fixed_batch, tmp_path, "epoch_0000", True, ["a", "b", "c"], "mock")
    sample_dir = tmp_path / "figures" / "light_fields" / "epoch_0000" / "sample_000"
    assert (sample_dir / "input_student_gray.png").stat().st_size > 0
    assert (sample_dir / "input_amplitude.png").stat().st_size > 0
    assert (sample_dir / "input_teacher_gray_rgb.png").stat().st_size > 0
    assert "sample index: 42" in (sample_dir / "label.txt").read_text(encoding="utf-8")
