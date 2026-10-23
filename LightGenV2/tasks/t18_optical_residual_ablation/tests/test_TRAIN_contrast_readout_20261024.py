import ast,hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parents[1];ENTRY=HERE/'lab_caltech20_TRAIN_contrast_readout_20261024.py'


def test_source_and_fixed_completed_native_mask():
    s=ENTRY.read_text();ast.parse(s)
    for name in ('lab_caltech20_TRAIN_photometric_readout_20261024.py','lab_caltech20_TRAIN_native_mask_20261024.py'):assert hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() in s
    assert "r['selected_predictive_correct']==258" in s and "r['actual_epochs']==24" in s and "local_SHA_backup.json" in s
    assert 'model_factory=factory,score_function=scorer,gauge_regularization=True' in s
    assert 'outer_steps=20,seconds=900' in s and 'new_CCD=0,G2_not_started=True' in s


def test_original_fitter_default_retained_and_optional_contrast_penalty():
    s=(HERE/'lab_caltech20_TRAIN_photometric_readout_20261024.py').read_text()
    assert 'gauge_regularization=False' in s and 'factory=model_factory or mask.make_predictor;scorer=score_function or mask.predictive_score' in s
    assert 'norm=contrast_norm if gauge_regularization else raw_norm' in s
    assert 'best_head={k:v.detach().clone()' in s and 'if gauge_regularization:w=w-w.mean(0,keepdims=True);b=b-b.mean()' in s
    assert 'class_contrast_regularization=gauge_regularization' in s
    for bad in ('engine.mask=', 'mask.make_predictor=', 'torch.nn.Sequential'):assert bad not in ENTRY.read_text()
