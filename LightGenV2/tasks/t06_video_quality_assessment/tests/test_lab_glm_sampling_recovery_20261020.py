"""Admission-only recovery preserves the strict GPU measurement contract."""
from pathlib import Path
from LightGenV2.tasks.t06_video_quality_assessment import lab_busy_resource_20261020 as lab
from LightGenV2.tasks.t06_video_quality_assessment import lab_strict_fp32_accuracy_20261020 as accuracy


def test_only_missing_glm_fp32_is_admitted():
    assert lab.GLM_RECOVERY == 'glm_sampling_recovery01'
    assert lab.point_plan(lab.GLM_RECOVERY) == [
        ('fp32', 'glm', 'mps', 100),
        ('fp32', 'glm', 'mps', 50),
        ('fp32', 'glm', 'temporal', 100),
    ]
    assert len(lab.point_plan('duration_1155')) == 24


def test_recovery_preserves_failure_and_requires_finished_accuracy():
    text = Path(lab.__file__).read_text()
    assert "'Power observation older than250ms'" in text
    assert "'accuracy_fp32/suite_status.json'" in text
    assert 'Sampling failure or prior accuracy still active' in text
    assert 'GLM_RECOVERY' in Path(accuracy.__file__).read_text()
    assert 'GLM_RECOVERY' in text[text.index('args.validate_pacing='):text.index("identity['pacing_calibration']")]
