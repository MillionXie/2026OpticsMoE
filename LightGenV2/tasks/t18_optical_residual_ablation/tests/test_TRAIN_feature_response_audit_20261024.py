import ast
from pathlib import Path

ENTRY=Path(__file__).resolve().parents[1]/'lab_caltech20_TRAIN_feature_response_audit_20261024.py'


def test_read_only_TRAIN_response_no_classification_or_camera_updates():
    s=ENTRY.read_text();ast.parse(s)
    assert "row['condition']" not in s or "'nominal_before'" in s
    assert "v['condition']=='nominal_before'" in s and "len(rows)==40" in s
    assert "data['test']" not in s and "data['train'][1]" not in s
    for bad in ('capture.acquire(','Bench(','loss.backward(','torch.optim','saved_model(','torch.save(','model.load_state_dict('):assert bad not in s
    assert 'not_classification_accuracy=True' in s and 'unique_physical_cause_identified=False' in s


def test_real_canonical_and_old_recent_comparison():
    s=ENTRY.read_text()
    assert "np.array_equal(encoded,expected)" in s and "begin=row['index']//8*8;ix=np.arange(begin,begin+8)" in s
    assert "input_identity_failure.json" in s
    assert "m.sha(recent_path)==row['crop_SHA256']" in s and "m.sha(canonical)==row['canonical_SHA256']" in s
    assert "same_input_old_vs_recent=metric(recorded,recent)" in s
    assert "model.nonlinear(predicted[None],'global')" in s and 'final256_photometric_vs_old_PCC' in s
    assert "base.identities(m,paths)==lineage" in s and "assert len(selected)==20" in s
