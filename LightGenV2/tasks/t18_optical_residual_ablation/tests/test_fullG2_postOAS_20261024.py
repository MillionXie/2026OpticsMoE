import ast
from pathlib import Path


def test_original_five_capture_head_fusion_only_finite_continue():
    p=Path(__file__).resolve().parents[1]/'lab_caltech20_fullG2_postOAS_20261024.py';s=p.read_text();ast.parse(s)
    assert 'm.allowed_keys(model,5)' in s and '==5150' in s and "for stage in base.STAGES:" in s
    assert "audit['matched_samples']==2011" in s and "sp['prefix_SHA256'][stage+'/'+role]" in s
    assert "epochs=12,seconds=120" in s and '>=240' in s and '>=225' in s
    assert "base.fit_suffix(" in s and 'new_inference_layers=0' in s and 'TEST_gradient=False' in s and '.backward(' not in s


def test_renewed_budget_has_separate_run_and_preserves_archived_deadline():
    s=(Path(__file__).resolve().parents[1]/'lab_caltech20_fullG2_postOAS_20261024.py').read_text()
    assert "a.out.name=='fullG2_joint_renewed_20261024'" in s
    assert "epochs=80,seconds=600" in s and "if not a.renewed_offline and not a.teacher_after_local:" in s
    assert "2026-10-24T08:00:00+08:00" in s and "gradient_data='TRAIN only'" in s


def test_full_physical_TRAIN_teacher_is_bounded_and_no_inference_teacher():
    root=Path(__file__).resolve().parents[1];s=(root/'lab_caltech20_fullG2_postOAS_20261024.py').read_text();e=(root/'lab_caltech20_tucam_layerwise_20261023.py').read_text();ast.parse(s);ast.parse(e)
    assert "source=prefix/'fullG2_native_local_20261024'" in s and 'epochs=64,seconds=480' in s
    assert "assert prefix==5" in e and "not tr['TEST_used']" in e and "not tr['VALID_used']" in e
    assert "teacher_gradient_fraction_max=.5" in e and "inference_teacher=False" in e and "teacher[ix]" in e
