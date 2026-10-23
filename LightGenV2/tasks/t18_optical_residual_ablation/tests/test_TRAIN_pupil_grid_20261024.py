import ast
from pathlib import Path
ENTRY=Path(__file__).resolve().parents[1]/'lab_caltech20_TRAIN_pupil_grid_20261024.py'


def test_TRAIN_only_fixed_last_and_new_phase_not_fitted():
    s=ENTRY.read_text();ast.parse(s)
    assert "fit=[v for v in pack if v['split']=='fit']" in s and 'len(unseen)==40' in s
    body=s.split('for epoch in range(1,49):',1)[1].split('final={k:assess',1)[0]
    assert 'rows=[fit[i]' in body and 'unseen' not in body and 'held' not in body and 'TEST' not in body
    assert 'fixed_last_only_no_held_early_stopping' in s and 'NO_rejected_parameters' in s
    assert "('held','held_nominal_before','unseen_TRAIN_phase')" in s and "+.03" in s and '.95*baseline' in s


def test_native_initialization_and_no_hardware_or_classifier_writes():
    s=ENTRY.read_text();assert 'nominal_response_identity_maxabs' in s and 'finite_curve_phase_amplitude_gradients' in s
    assert "np.array_equal(pixels,measured[anchor['index']])" in s and "m.sha(input_path)==receipt['canonical_SHA256']" in s
    assert 'model_checkpoint_written=False' in s and 'LUT_changed=False' in s and 'classification_accuracy_updated=False' in s
    assert 'protected=={q:m.sha(q) for q in protected}' in s
    for forbidden in ('torch.save(', 'Bench(', 'capture.acquire(', 'data[\'test\']', 'model.cuda('):assert forbidden not in s


def test_phase_sign_counterfactual_is_read_only_not_a_LUT_change():
    s=ENTRY.read_text();body=s.split("if getattr(a,'mode','fit')=='handedness':",1)[1].split("m.save(a.out/'protocol.json',dict(source_git",2)[0]
    assert 'sign=-1.' in body and 'opposite={k:assess(v)' in body and 'opt.step' not in body
    assert 'unique_phase_LUT_direction_identified=False' in s and 'actual_phase_encoding_unchanged=True' in s
