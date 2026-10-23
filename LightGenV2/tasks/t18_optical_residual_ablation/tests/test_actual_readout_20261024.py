import ast
from pathlib import Path
ENTRY=Path(__file__).resolve().parents[1]/'lab_caltech20_actual_readout_20261024.py'


def test_TRAIN_objective_only_and_native_real_data_cache_identity():
    s=ENTRY.read_text();ast.parse(s);closure=s.split('def closure():',1)[1].split('opt.step(closure)',1)[0]
    assert 'train64,labels,weights' in closure and 'test' not in closure.lower()
    assert "stats['feature_cache_SHA256']" in s and "np.array_equal(z['ids'],data['train'][2])" in s
    assert 'max(error,train_error)<=3e-6' in s and "regularization=2*.1*r['results']['best']['train']['nll']" in s
    assert 'head.weight.sub_(head.weight.mean(0,keepdim=True))' in s and 'best_head=raw_parent' in s
    assert 'range(1,1 if ridge or proximal else 81)' in s and 'max_eval=8' in s


def test_only_original5140_and_native_full_CPU_backup():
    s=ENTRY.read_text();assert "keys={'linear.weight','linear.bias'}" in s
    assert "for label in ('best','last')" in s and "for role in ('train','test')" in s
    assert 'loop.validate_state(m,native,original,keys)' in s and 'base.identities(m,paths)==lineage' in s
    assert 'SDK_used=False,GPU_used=False' in s and 'all_non_readout_frozen=True' in s
    assert 'new_inference_layers=0' in s and 'class_contrast_regularization=not(ridge or proximal)' in s
    assert "[v['source_id'] for v in records]" in s and 'TEST_used_for_regularization=False' in s
